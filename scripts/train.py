# scripts/train.py
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any

import torch
import yaml
from safetensors.torch import load_file, save_file
from torch.optim import AdamW

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.jev_set.data_curation import permute_and_curriculum_subsample_options
from src.jev_set.losses import JEVCompositeDecisionLoss
from src.jev_set.modeling_jev_set import (
    QUESTION_TYPE_TO_ID,
    JEVSetConfig,
    JEVSetForDecisionIndex,
    forward_single_request,
)
from src.jev_set.packing import DeterministicWhitespaceTokenizer, build_tree_masked_packs


def detect_colab_gpu_profile(stage_cfg: dict[str, Any]) -> dict[str, Any]:
    """Automatically selects the optimal Colab Pro GPU profile (A100 80GB/40GB, L4 24GB, T4 16GB)."""
    profiles = stage_cfg.get("colab_pro_profiles", {})
    default_profile = {
        "per_device_requests": 1,
        "gradient_accumulation_steps": 32,
        "effective_batch_requests": 32,
        "gradient_checkpointing": True,
        "tf32": False,
    }
    if not torch.cuda.is_available() or not profiles:
        return default_profile

    gpu_name = torch.cuda.get_device_name(0).lower()
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    if ("a100" in gpu_name or "h100" in gpu_name) and vram_gb >= 70:
        return profiles.get("a100_80gb", default_profile)
    if "a100" in gpu_name or vram_gb >= 35:
        return profiles.get("a100_40gb", default_profile)
    if "l4" in gpu_name or vram_gb >= 20:
        return profiles.get("l4_24gb", default_profile)
    return profiles.get("t4_16gb", default_profile)


def set_backbone_layer_freeze(model: JEVSetForDecisionIndex, freeze_bottom_layers: int = 14) -> None:
    """Freezes token embeddings and bottom `freeze_bottom_layers` transformer blocks during Stage 1 warmup."""
    for p in model.backbone.embeddings.parameters():
        p.requires_grad = freeze_bottom_layers <= 0
    for idx, layer in enumerate(model.backbone.layers):
        requires_grad = idx >= freeze_bottom_layers
        for p in layer.parameters():
            p.requires_grad = requires_grad


def build_llrd_optimizer(
    model: JEVSetForDecisionIndex,
    backbone_lr: float = 3e-5,
    head_lr: float = 1.5e-4,
    weight_decay: float = 0.01,
    layer_decay: float = 0.95,
) -> AdamW:
    no_decay = (
        "bias",
        "norm",
        "set_gate",
        "log_temperature",
        "cardinality_slope",
        "multi_label_bias",
    )
    param_groups: list[dict[str, Any]] = []

    head_decay = [
        p
        for n, p in model.head.named_parameters()
        if p.requires_grad and not any(nd in n for nd in no_decay)
    ]
    head_nodecay = [
        p
        for n, p in model.head.named_parameters()
        if p.requires_grad and any(nd in n for nd in no_decay)
    ]
    if head_decay:
        param_groups.append({"params": head_decay, "lr": head_lr, "weight_decay": weight_decay})
    if head_nodecay:
        param_groups.append({"params": head_nodecay, "lr": head_lr, "weight_decay": 0.0})

    num_layers = model.config.num_hidden_layers
    for name, param in model.backbone.named_parameters():
        if not param.requires_grad:
            continue
        lr = backbone_lr * (layer_decay ** num_layers)
        if name.startswith("layers."):
            layer_idx = int(name.split(".")[1])
            lr = backbone_lr * (layer_decay ** (num_layers - 1 - layer_idx))
        elif "final_norm" in name:
            lr = backbone_lr
        wd = 0.0 if any(nd in name for nd in no_decay) else weight_decay
        param_groups.append({"params": [param], "lr": lr, "weight_decay": wd})

    use_fused = torch.cuda.is_available()
    return AdamW(param_groups, betas=(0.9, 0.98), eps=1e-6, fused=use_fused)


def run_training_stage(
    config_path: Path,
    max_steps: int = 10,
    load_pretrained: bool = False,
    num_layers: int = 22,
    resume: bool = False,
    output_dir_override: Path | None = None,
    save_every_steps: int = 100,
    require_real_data: bool = False,
) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as f:
        stage_cfg = yaml.safe_load(f)

    hw_profile = detect_colab_gpu_profile(stage_cfg)
    if hw_profile.get("tf32", False) and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = output_dir_override or Path(stage_cfg.get("output_dir", "checkpoints/jev-set-150m-stage"))
    out_dir.mkdir(parents=True, exist_ok=True)
    state_file = out_dir / "trainer_state.json"
    ckpt_file = out_dir / "model.safetensors"

    init_ckpt = str(stage_cfg.get("init_checkpoint", "cross-encoder/ettin-reranker-150m-v1"))
    ckpt_p = Path(init_ckpt)
    has_local_ckpt = (ckpt_p.is_file() and ckpt_p.suffix == ".safetensors") or (
        ckpt_p.is_dir() and (ckpt_p / "model.safetensors").exists()
    )
    model_cfg = JEVSetConfig(
        backbone_name_or_path=init_ckpt,
        num_hidden_layers=num_layers,
    )
    model = JEVSetForDecisionIndex(
        model_cfg,
        load_pretrained=(load_pretrained or has_local_ckpt),
    ).to(device)

    start_step = 1
    losses: list[float] = []
    resumed_from_checkpoint = False
    if resume and ckpt_file.exists() and state_file.exists():
        saved_meta = json.loads(state_file.read_text(encoding="utf-8"))
        saved_tensors = load_file(str(ckpt_file))
        if set(saved_tensors.keys()) == set(model.state_dict().keys()):
            model.load_state_dict(saved_tensors, strict=True)
            start_step = int(saved_meta.get("completed_steps", 0)) + 1
            losses = [float(x) for x in saved_meta.get("losses", [])]
            resumed_from_checkpoint = True

    model.train()

    train_cfg = stage_cfg.get("training", {})
    loss_cfg = stage_cfg.get("loss", {})
    data_cfg = stage_cfg.get("data", {})
    freeze_layers = min(int(train_cfg.get("freeze_bottom_layers", 0)), max(0, num_layers - 1))
    unfreeze_step = int(train_cfg.get("unfreeze_at_step", 0))
    if freeze_layers > 0 and (unfreeze_step == 0 or start_step < unfreeze_step):
        set_backbone_layer_freeze(model, freeze_bottom_layers=freeze_layers)

    optimizer = build_llrd_optimizer(
        model=model,
        backbone_lr=float(train_cfg.get("backbone_lr", 3e-5)),
        head_lr=float(train_cfg.get("head_lr", 1.5e-4)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        layer_decay=float(train_cfg.get("llrd_decay", 0.95)),
    )
    criterion = JEVCompositeDecisionLoss(
        kd_alpha=float(loss_cfg.get("kd_alpha", 0.0)),
        kd_temperature=float(loss_cfg.get("kd_temperature", 2.0)),
        margin=float(loss_cfg.get("margin", 1.0)),
        margin_weight=float(loss_cfg.get("margin_weight", 0.15)),
        zlpr_weight=float(loss_cfg.get("zlpr_weight", 0.35)),
    )

    tokenizer: Any = DeterministicWhitespaceTokenizer()
    if load_pretrained:
        try:
            from transformers import AutoTokenizer

            has_local_tok = ckpt_p.is_dir() and (
                (ckpt_p / "tokenizer.json").exists() or (ckpt_p / "tokenizer_config.json").exists()
            )
            tok_source = (
                init_ckpt
                if (has_local_tok or not ckpt_p.exists())
                else "cross-encoder/ettin-reranker-150m-v1"
            )
            tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
        except Exception:
            tokenizer = DeterministicWhitespaceTokenizer()

    train_path = Path(data_cfg.get("train_path", "data/processed/stage1_ood_train.jsonl"))
    rows: list[dict[str, Any]] = []
    if train_path.exists():
        with train_path.open("r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rows.append(json.loads(line))
    had_no_rows = not rows
    if not rows:
        rows = [
            {
                "id": "fallback_0",
                "group_id": "grp_0",
                "state": "User requests an API call for weather forecast.",
                "instructions": "Select the matching function signature.",
                "options": ["get_weather(city)", "send_email(to)", "delete_file(path)", "run_sql(query)"],
                "question_type": "choice",
                "gold_indices": [0],
                "difficulty_weight": 1.2,
            }
        ]

    if require_real_data:
        if had_no_rows:
            raise RuntimeError(f"--require-real-data: training file {train_path} missing/empty (would use hard-coded fallback row).")
        if isinstance(tokenizer, DeterministicWhitespaceTokenizer):
            raise RuntimeError("--require-real-data: real tokenizer failed to load (whitespace tokenizer fallback).")
        if not (load_pretrained or has_local_ckpt):
            raise RuntimeError("--require-real-data: pass --load-pretrained so real backbone weights are used.")
        n_real = sum(1 for r in rows if r.get("source") == "real")
        n_syn = sum(1 for r in rows if r.get("source") == "synthetic")
        n_unk = len(rows) - n_real - n_syn
        print(f"[real-data-check] {train_path}: real={n_real} synthetic={n_syn} untagged={n_unk}", flush=True)
        if n_real == 0 or n_syn > 0 or n_unk > 0:
            raise RuntimeError(
                f"--require-real-data violated in {train_path}: real={n_real} synthetic={n_syn} untagged={n_unk}. "
                "Rebuild data with --no-synthetic-fallback (stale smoke/synthetic JSONL on disk?)."
            )
    used_source_counts = {"real": 0, "synthetic": 0, "other": 0}

    if bool(data_cfg.get("cluster_by_group_id", False)):
        rows.sort(key=lambda r: (str(r.get("group_id", "")), str(r.get("id", ""))))

    dynamic_shuffle = bool(data_cfg.get("dynamic_option_shuffle", True))
    rng = random.Random(int(stage_cfg.get("seed", 42)) + start_step)
    grad_accum = max(1, int(hw_profile.get("gradient_accumulation_steps", 1)))
    ckpt_interval = max(1, int(train_cfg.get("save_every_steps", save_every_steps)))
    optimizer.zero_grad()

    def _save_stage_checkpoint(completed_step: int) -> None:
        save_file(model.state_dict(), str(ckpt_file))
        state_payload = {
            "stage": stage_cfg.get("stage", "unknown"),
            "completed_steps": completed_step,
            "losses": losses,
            "resumed": resumed_from_checkpoint,
        }
        state_file.write_text(json.dumps(state_payload, indent=2) + "\n", encoding="utf-8")

    for step in range(start_step, max_steps + 1):
        if freeze_layers > 0 and unfreeze_step > 0 and step == unfreeze_step:
            set_backbone_layer_freeze(model, freeze_bottom_layers=0)
            optimizer = build_llrd_optimizer(
                model=model,
                backbone_lr=float(train_cfg.get("backbone_lr", 3e-5)),
                head_lr=float(train_cfg.get("head_lr", 1.5e-4)),
                weight_decay=float(train_cfg.get("weight_decay", 0.01)),
                layer_decay=float(train_cfg.get("llrd_decay", 0.95)),
            )

        base_rec = rows[(step - 1) % len(rows)]
        used_source_counts[base_rec.get("source", "other") if base_rec.get("source") in ("real", "synthetic") else "other"] += 1
        if require_real_data and used_source_counts["synthetic"] > 0:
            raise RuntimeError(f"--require-real-data: synthetic row used at step {step} ({base_rec.get('id')}).")
        rec = permute_and_curriculum_subsample_options(base_rec, rng=rng) if dynamic_shuffle else base_rec
        raw_logits, cal_logits = forward_single_request(
            model=model,
            tokenizer=tokenizer,
            state=rec["state"],
            instructions=rec["instructions"],
            options=rec["options"],
            question_type=rec.get("question_type", "choice"),
            device=device,
        )
        t_probs = (
            torch.tensor(rec["teacher_probs"], dtype=torch.float32, device=device)
            if "teacher_probs" in rec and rec["teacher_probs"] is not None
            else None
        )
        loss = criterion(
            raw_logits=raw_logits,
            calibrated_logits=cal_logits,
            question_type=rec.get("question_type", "choice"),
            gold_indices=rec.get("gold_indices", [0]),
            teacher_probs=t_probs,
            difficulty_weight=float(rec.get("difficulty_weight", 1.0)),
        )
        (loss / grad_accum).backward()
        losses.append(float(loss.detach().item()))

        if step % grad_accum == 0 or step == max_steps:
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg.get("max_grad_norm", 1.0)))
            optimizer.step()
            optimizer.zero_grad()

        if step % ckpt_interval == 0 or step == max_steps:
            _save_stage_checkpoint(step)
            print(f"[real-data-check] step {step}: rows used by source = {used_source_counts}", flush=True)

    final_completed = max(max_steps, start_step - 1)
    if start_step > max_steps:
        _save_stage_checkpoint(final_completed)

    return {
        "stage": stage_cfg.get("stage", "unknown"),
        "steps": final_completed,
        "resumed_from_checkpoint": resumed_from_checkpoint,
        "mean_loss": sum(losses) / max(1, len(losses)),
        "checkpoint": str(ckpt_file),
        "hardware_profile": hw_profile,
        "rows_used_by_source": used_source_counts,
        "train_rows": len(rows),
        "real_tokenizer": not isinstance(tokenizer, DeterministicWhitespaceTokenizer),
    }


def run_ablation_experiment(exp_id: str = "all") -> dict[str, Any]:
    """
    Executes the Experiment A-H Ablation Matrix from Section 12 of PLAN.md and evaluates
    each decision gate using the 10,000-sample paired bootstrap evaluator.
    """
    from scripts.evaluate_official import compute_local_full_val_and_bootstrap

    ablation_specs: dict[str, dict[str, Any]] = {
        "exp_a": {
            "name": "Exp A: Backbone Gate (ettin-reranker-150m-v1 vs ettin-encoder-150m vs ModernBERT-base)",
            "gate_criterion": "Delta Full_val >= +1.5",
            "s_cand": [30.8, 31.4, 31.1, 31.5, 31.2],
            "o_cand": [9.3, 9.5, 9.4, 9.6, 9.2],
            "s_base": [28.2, 28.5, 28.1, 28.4, 28.3],
            "o_base": [8.1, 8.2, 8.0, 8.3, 8.1],
        },
        "exp_b": {
            "name": "Exp B: Encoding Topology Gate (Tree-Masked Virtual-Pair vs Per-Option vs Flat Joint)",
            "gate_criterion": ">= 4x speedup over Per-Option & 0.0% permutation flip rate",
            "s_cand": [31.2, 31.0, 31.4, 31.1, 31.3],
            "o_cand": [9.4, 9.3, 9.5, 9.4, 9.4],
            "s_base": [29.5, 29.3, 29.6, 29.4, 29.5],
            "o_base": [8.6, 8.5, 8.7, 8.6, 8.6],
            "permutation_flip_rate": 0.0,
            "speedup_vs_per_option": 4.4,
        },
        "exp_c": {
            "name": "Exp C: EquivariantSetInteractionHead Gate (Learned gamma vs gamma=0 linear scorer)",
            "gate_criterion": "Delta Full_val >= +0.35 at p < 0.05",
            "s_cand": [31.2, 31.5, 31.0, 31.3, 31.0],
            "o_cand": [9.4, 9.5, 9.3, 9.4, 9.4],
            "s_base": [30.1, 30.3, 29.9, 30.2, 30.0],
            "o_base": [8.9, 9.0, 8.8, 8.9, 8.9],
        },
        "exp_d": {
            "name": "Exp D: Mixture Ratio Gate (55:45 OOD:Same-Skill & alpha=0.55 vs 30:70 / 75:25)",
            "gate_criterion": "Positive Delta Full_val with Delta O_val >= -0.15",
            "s_cand": [31.2, 31.1, 31.3, 31.2, 31.2],
            "o_cand": [9.4, 9.5, 9.3, 9.4, 9.4],
            "s_base": [30.4, 30.3, 30.5, 30.4, 30.4],
            "o_base": [8.5, 8.6, 8.4, 8.5, 8.5],
        },
        "exp_e": {
            "name": "Exp E: Multi-Teacher Distillation Gate (3-Teacher Ensemble vs Hard Labels)",
            "gate_criterion": "Delta Full_val >= +1.0",
            "s_cand": [31.2, 31.4, 31.1, 31.3, 31.0],
            "o_cand": [9.4, 9.5, 9.3, 9.4, 9.4],
            "s_base": [29.2, 29.4, 29.1, 29.3, 29.0],
            "o_base": [8.4, 8.5, 8.3, 8.4, 8.4],
        },
        "exp_f": {
            "name": "Exp F: Multi-Label ZLPR + Relative-Margin Gate (Delta_gap=1.35 vs raw BCE)",
            "gate_criterion": ">= +6.0 points on SATA-Bench + ACOS + BFCL",
            "s_cand": [31.2, 31.3, 31.1, 31.2, 31.2],
            "o_cand": [9.4, 9.4, 9.4, 9.4, 9.4],
            "s_base": [29.8, 29.9, 29.7, 29.8, 29.8],
            "o_base": [9.0, 9.0, 9.0, 9.0, 9.0],
            "multi_label_gain_points": 6.8,
        },
        "exp_g": {
            "name": "Exp G: Linked-Case & Gold-Star Gate (1.2x Gold + group_id Clustering vs Uniform)",
            "gate_criterion": ">= +0.8 points on Gold-Star benchmarks",
            "s_cand": [31.2, 31.4, 31.0, 31.3, 31.1],
            "o_cand": [9.4, 9.4, 9.4, 9.4, 9.4],
            "s_base": [30.1, 30.3, 29.9, 30.2, 30.0],
            "o_base": [9.3, 9.3, 9.3, 9.3, 9.3],
            "gold_star_gain_points": 1.1,
        },
        "exp_h": {
            "name": "Exp H: 3-Seed Model Soup + L-BFGS Cardinality Calibration T(K, t) vs Single Seed",
            "gate_criterion": "ECE <= 0.045, Brier <= 0.480, Delta O_val >= -0.15",
            "s_cand": [31.2, 31.3, 31.1, 31.2, 31.2],
            "o_cand": [9.4, 9.5, 9.3, 9.4, 9.4],
            "s_base": [30.6, 30.7, 30.5, 30.6, 30.6],
            "o_base": [9.1, 9.2, 9.0, 9.1, 9.1],
            "ece": 0.044,
            "brier": 0.472,
        },
    }

    target_keys = list(ablation_specs.keys()) if exp_id == "all" else [exp_id.lower()]
    results: dict[str, Any] = {}
    for k in target_keys:
        if k not in ablation_specs:
            raise ValueError(f"Unknown ablation experiment {k!r}; choose from {list(ablation_specs)} or 'all'")
        spec = ablation_specs[k]
        boot = compute_local_full_val_and_bootstrap(
            s_val_cases=spec["s_cand"],
            o_val_cases=spec["o_cand"],
            s_val_baseline=spec["s_base"],
            o_val_baseline=spec["o_base"],
            n_bootstrap=10_000,
        )
        results[k] = {
            "name": spec["name"],
            "gate_criterion": spec["gate_criterion"],
            "bootstrap": boot,
            "gate_passed": boot["promote_checkpoint"],
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Unified Stage 1/2/3 Trainer & Ablation Runner for JEV-Set-150M.")
    parser.add_argument("--config", type=Path, default=Path("configs/stage1_ood.yaml"))
    parser.add_argument("--max-steps", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=22)
    parser.add_argument("--load-pretrained", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Resume training stage from checkpoint if available.")
    parser.add_argument("--ablation", type=str, default=None, help="Run ablation experiment (exp_a..exp_h or all).")
    parser.add_argument("--require-real-data", action="store_true",
                        help="Fail loudly if any synthetic/untagged rows, fallback row or fallback tokenizer would be used.")
    parser.add_argument("--one-epoch", action="store_true",
                        help="Set max-steps to the number of rows in the stage train file (one pass, 1 request/step).")
    args = parser.parse_args()

    if args.ablation:
        print(
            "WARNING: --ablation reports HARD-CODED PLACEHOLDER score vectors fed to the bootstrap evaluator; "
            "no model is trained or evaluated. Use scripts/pilot_ablation.py for measured Exp A/B pilot numbers.",
            file=sys.stderr,
        )
        ablation_res = run_ablation_experiment(args.ablation)
        for v in ablation_res.values():
            v["placeholder_inputs"] = True
        print(json.dumps(ablation_res, indent=2))
        return

    max_steps = args.max_steps
    if args.one_epoch:
        with args.config.open("r", encoding="utf-8") as f:
            _cfg = yaml.safe_load(f)
        _tp = Path(_cfg.get("data", {}).get("train_path", "data/processed/stage1_ood_train.jsonl"))
        if not _tp.exists():
            raise FileNotFoundError(f"--one-epoch: {_tp} not found")
        max_steps = sum(1 for line in _tp.open("r", encoding="utf-8") if line.strip())

    summary = run_training_stage(
        config_path=args.config,
        max_steps=max_steps,
        load_pretrained=args.load_pretrained,
        num_layers=args.num_layers,
        resume=args.resume,
        require_real_data=args.require_real_data,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
