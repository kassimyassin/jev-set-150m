# scripts/verify_real_weights.py
"""Downloads the real cross-encoder/ettin-reranker-150m-v1 weights and strictly verifies loading.

Checks: 0 missing/unexpected backbone keys, 149,014,272 backbone params, 4_Dense scorer loaded
(matches the file), backbone outputs == transformers.ModernBertModel, real tokenizer forward via JEVSetEngine.
Exit code != 0 on any failure.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT, _ROOT / "src"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torch
from huggingface_hub import hf_hub_download, snapshot_download
from safetensors.torch import load_file
from transformers import AutoModel, AutoTokenizer

from src.jev_set.engine import JEVSetEngine
from src.jev_set.modeling_jev_set import JEVSetConfig, JEVSetForDecisionIndex

REPO = "cross-encoder/ettin-reranker-150m-v1"
EXPECTED_BACKBONE = 149_014_272


def check(cond: bool, msg: str) -> None:
    print(("PASS " if cond else "FAIL ") + msg)
    if not cond:
        sys.exit(1)


def main() -> None:
    local = snapshot_download(
        REPO,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "4_Dense/*", "tokenizer*"],
        ignore_patterns=["onnx/*", "openvino/*"],
    )
    print("snapshot:", local)
    cfg = JEVSetConfig(backbone_name_or_path=REPO)
    model = JEVSetForDecisionIndex(cfg, load_pretrained=True).eval()

    # Re-run the loader to obtain key accounting (constructor discards it).
    stats = model.load_pretrained_weights(REPO)
    print("load stats:", stats)
    check(stats["missing_keys"] == 0, "0 missing backbone keys")
    check(stats["unexpected_keys"] == 0, "0 unexpected backbone keys")
    n_bb = sum(p.numel() for p in model.backbone.parameters())
    check(n_bb == EXPECTED_BACKBONE, f"backbone params == {EXPECTED_BACKBONE:,} (got {n_bb:,})")

    dense = Path(local) / "4_Dense" / "model.safetensors"
    check(dense.exists(), "4_Dense/model.safetensors exists in repo snapshot")
    ds = load_file(str(dense))
    w = ds.get("linear.weight", ds.get("weight"))
    b = ds.get("linear.bias", ds.get("bias"))
    print("4_Dense keys:", {k: tuple(v.shape) for k, v in ds.items()})
    check(w is not None and torch.allclose(model.head.base_scorer.weight.detach().float(), w.float()),
          "base_scorer.weight == 4_Dense weight")
    if b is not None:
        check(torch.allclose(model.head.base_scorer.bias.detach().float(), b.float()), "base_scorer.bias == 4_Dense bias")

    # Backbone equivalence vs transformers (fp32, CPU)
    tok = AutoTokenizer.from_pretrained(REPO)
    ref = AutoModel.from_pretrained(REPO, torch_dtype=torch.float32, attn_implementation="eager").eval()
    model.backbone.float()
    long_text = " ".join(f"Sentence number {i} discusses topic {i * 7 % 13} in some detail." for i in range(60))
    for label, texts in (
        ("short", ["Which option is best? A careful short sentence.", "Another example text"]),
        ("long(>128 tok)", [long_text, "Another example text"]),
    ):
        enc = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            ref_out = ref(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"]).last_hidden_state
            mine = model.backbone(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
            mine = getattr(mine, "last_hidden_state", mine)
        mask = enc["attention_mask"].bool()
        d = (ref_out - mine).abs()[mask]
        scale = ref_out.abs()[mask].max().item()
        diff, mean_d = d.max().item(), d.mean().item()
        print(f"[{label}] seq_len={enc['input_ids'].shape[1]} max|diff|={diff:.3e} (ref max |h|={scale:.1f}, rel={diff / scale:.2e}) mean|diff|={mean_d:.3e}")
        check(diff / scale < 2e-3 and mean_d < 1e-3, f"[{label}] backbone output matches transformers ModernBertModel")

    # Real tokenizer end-to-end through the engine
    eng = JEVSetEngine(checkpoint_dir=REPO, device="cpu", load_pretrained=True)
    check(type(eng.tokenizer).__name__ != "DeterministicWhitespaceTokenizer", f"real tokenizer used ({type(eng.tokenizer).__name__})")
    from src.jev_set.engine import EngineRequest

    req = EngineRequest(
        request_id="r0", state="The capital of France is a famous city on the Seine.",
        instructions="Pick the correct city.", options=("Paris", "Berlin", "Madrid", "Rome"),
        question_type="choice",
    )
    resp = eng.run(req)
    print("response:", resp)
    check(resp.status == "ok" and len(resp.option_scores) == 4, "engine forward pass ok with 4 scores")
    check(all(torch.isfinite(torch.tensor(s)) for s in resp.option_scores), "scores finite")
    print("ALL REAL-WEIGHT CHECKS PASSED")


if __name__ == "__main__":
    main()
