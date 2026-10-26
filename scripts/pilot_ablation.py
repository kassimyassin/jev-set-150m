# scripts/pilot_ablation.py
"""MEASURED pilot versions of Experiment A (backbone gate) and Experiment B (encoding topology gate).

Unlike `train.py --ablation`, which feeds hard-coded placeholder score vectors to the bootstrap evaluator,
this script actually runs models on data:

  Exp A: short fine-tune of JEV-Set with backbone = ettin-reranker-150m-v1 vs ettin-encoder-150m
         (same architecture), then paired bootstrap of per-row correctness on held-out rows.
         (ModernBERT-base has a different width and cannot be loaded into the 150M budget model.)
  Exp B: tree-masked shared-state virtual-pair packing vs K independent single-option passes:
         measured wall-clock speed-up + permutation flip rate (selected option text under shuffled order).

Pilot-scale numbers are NOISY and are NOT the PLAN.md gate verdicts; they check the machinery end-to-end.
"""
from __future__ import annotations

import argparse
import copy
import json
import random
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import numpy as np
import torch
import yaml
from safetensors.torch import load_file

from scripts.train import run_training_stage
from src.jev_set.modeling_jev_set import JEVSetConfig, JEVSetForDecisionIndex, forward_single_request
from src.jev_set.packing import DeterministicWhitespaceTokenizer

RERANKER = "cross-encoder/ettin-reranker-150m-v1"
ENCODER = "jhu-clsp/ettin-encoder-150m"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(l) for l in path.open("r", encoding="utf-8") if l.strip()]


def load_tokenizer(require_real: bool):
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(RERANKER)
    except Exception:
        if require_real:
            raise
        return DeterministicWhitespaceTokenizer()


def assert_real(rows: list[dict[str, Any]], label: str, require_real: bool) -> dict[str, int]:
    c = {"real": 0, "synthetic": 0, "untagged": 0}
    for r in rows:
        c[r.get("source", "untagged") if r.get("source") in ("real", "synthetic") else "untagged"] += 1
    print(f"[real-data-check] {label}: {c}", flush=True)
    if require_real and (c["real"] == 0 or c["synthetic"] or c["untagged"]):
        raise RuntimeError(f"--require-real-data violated for {label}: {c}")
    return c


@torch.no_grad()
def predict(model, tok, row, device, options=None) -> list[float]:
    opts = options if options is not None else row["options"]
    _, cal = forward_single_request(
        model=model, tokenizer=tok, state=row["state"], instructions=row["instructions"],
        options=opts, question_type=row.get("question_type", "choice"), device=device,
    )
    return [float(x) for x in cal.detach().float().reshape(-1).tolist()]


def paired_bootstrap(a: np.ndarray, b: np.ndarray, n: int = 10_000, seed: int = 0) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    d = a - b
    idx = rng.integers(0, len(d), size=(n, len(d)))
    means = d[idx].mean(axis=1)
    return {
        "delta_acc_points": float(100 * d.mean()),
        "ci95_lo": float(100 * np.percentile(means, 2.5)),
        "ci95_hi": float(100 * np.percentile(means, 97.5)),
        "p_delta_le_0": float((means <= 0).mean()),
        "n_rows": int(len(d)),
    }


def run_exp_b(rows, tok, device, n_rows: int, seed: int) -> dict[str, Any]:
    model = JEVSetForDecisionIndex(JEVSetConfig(backbone_name_or_path=RERANKER), load_pretrained=True).to(device).eval()
    rng = random.Random(seed)
    multi = [r for r in rows if len(r["options"]) >= 3][:n_rows]
    if not multi:
        raise RuntimeError("Exp B: no rows with K>=3")
    flips = 0
    t_packed = t_per_option = 0.0
    for r in multi:
        t0 = time.perf_counter()
        s = predict(model, tok, r, device)
        t_packed += time.perf_counter() - t0
        t0 = time.perf_counter()
        for o in r["options"]:
            predict(model, tok, r, device, options=[o])
        t_per_option += time.perf_counter() - t0
        perm = list(range(len(r["options"])))
        rng.shuffle(perm)
        s2 = predict(model, tok, r, device, options=[r["options"][i] for i in perm])
        best1 = r["options"][int(np.argmax(s))]
        best2 = r["options"][perm[int(np.argmax(s2))]]
        flips += int(best1 != best2)
    return {
        "name": "Exp B (measured pilot): tree-masked packing vs per-option passes",
        "rows": len(multi),
        "permutation_flip_rate": flips / len(multi),
        "speedup_vs_per_option": t_per_option / max(t_packed, 1e-9),
        "gate_criterion": ">=4x speedup & 0.0% flip rate (pilot is indicative only)",
        "device": str(device),
    }


def run_exp_a(train_rows_path: Path, eval_rows, tok, device, steps: int, base_cfg: Path, seed: int) -> dict[str, Any]:
    cfg = yaml.safe_load(base_cfg.read_text(encoding="utf-8"))
    correct: dict[str, np.ndarray] = {}
    tmp = Path(tempfile.mkdtemp(prefix="pilot_expA_"))
    for tag, backbone in (("reranker", RERANKER), ("encoder", ENCODER)):
        c = copy.deepcopy(cfg)
        c["init_checkpoint"] = backbone
        c["output_dir"] = str(tmp / tag)
        c.setdefault("data", {})["train_path"] = str(train_rows_path)
        cfg_path = tmp / f"{tag}.yaml"
        cfg_path.write_text(yaml.safe_dump(c), encoding="utf-8")
        summary = run_training_stage(
            config_path=cfg_path, max_steps=steps, load_pretrained=True, require_real_data=True,
            output_dir_override=tmp / tag,
        )
        print(f"[expA] {tag}: mean_loss={summary['mean_loss']:.4f} sources={summary['rows_used_by_source']}", flush=True)
        model = JEVSetForDecisionIndex(JEVSetConfig(backbone_name_or_path=backbone), load_pretrained=False).to(device)
        model.load_state_dict(load_file(str(tmp / tag / "model.safetensors")), strict=True)
        model.eval()
        flags = []
        for r in eval_rows:
            s = predict(model, tok, r, device)
            flags.append(float(int(np.argmax(s)) in set(r["gold_indices"])))
        correct[tag] = np.array(flags)
        del model
    return {
        "name": "Exp A (measured pilot): backbone gate, reranker-150m vs encoder-150m, short fine-tune",
        "train_steps": steps,
        "acc_reranker": float(correct["reranker"].mean()),
        "acc_encoder": float(correct["encoder"].mean()),
        "bootstrap_reranker_minus_encoder": paired_bootstrap(correct["reranker"], correct["encoder"]),
        "gate_criterion": "Delta >= +1.5 points (pilot is indicative only; tiny n)",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", nargs="+", default=["a", "b"], choices=["a", "b"])
    ap.add_argument("--train-jsonl", type=Path, default=Path("data/processed/stage1_ood_train.jsonl"))
    ap.add_argument("--eval-jsonl", type=Path, default=Path("data/processed/s_val_5fold.jsonl"))
    ap.add_argument("--config", type=Path, default=Path("configs/stage1_ood.yaml"))
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--eval-rows", type=int, default=64)
    ap.add_argument("--require-real-data", action="store_true")
    ap.add_argument("--out", type=Path, default=Path("data/processed/pilot_ablation_results.json"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = load_tokenizer(args.require_real_data)
    eval_rows = read_jsonl(args.eval_jsonl)[: args.eval_rows]
    counts = assert_real(eval_rows, f"eval rows {args.eval_jsonl}", args.require_real_data)
    results: dict[str, Any] = {"eval_row_sources": counts, "device": str(device)}
    if "b" in args.exp:
        results["exp_b"] = run_exp_b(eval_rows, tok, device, n_rows=min(len(eval_rows), 16), seed=args.seed)
    if "a" in args.exp:
        results["exp_a"] = run_exp_a(args.train_jsonl, eval_rows, tok, device, args.steps, args.config, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
