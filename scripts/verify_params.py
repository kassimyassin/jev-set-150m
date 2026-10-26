# scripts/verify_params.py
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.jev_set.modeling_jev_set import (
    EquivariantSetInteractionHead,
    JEVSetConfig,
    JEVSetForDecisionIndex,
)


def verify_exact_budget_and_equivariance() -> dict[str, int | float]:
    cfg = JEVSetConfig()
    model = JEVSetForDecisionIndex(cfg, load_pretrained=False).eval()
    head = model.head

    V, D, L, d_ff = (
        cfg.vocab_size,
        cfg.hidden_size,
        cfg.num_hidden_layers,
        cfg.intermediate_size,
    )
    emb_params = V * D + D
    layer0_params = (3 * D * D) + (D * D) + D + (2 * d_ff * D) + (D * d_ff)
    other_layers_params = (L - 1) * (
        D + (3 * D * D) + (D * D) + D + (2 * d_ff * D) + (D * d_ff)
    )
    final_norm_params = D
    analytical_backbone_params = (
        emb_params + layer0_params + other_layers_params + final_norm_params
    )

    backbone_params = sum(p.numel() for p in model.backbone.parameters())
    head_params = sum(p.numel() for p in head.parameters())
    total_params = backbone_params + head_params
    margin = cfg.hard_param_budget - total_params

    print(f"Backbone (ettin-reranker-150m-v1) : {backbone_params:>12,} params")
    print(f"Head (EquivariantSetInteraction)  : {head_params:>12,} params")
    for name, p in head.named_parameters():
        print(f"  {name:28s}: {str(tuple(p.shape)):15s} -> {p.numel():,}")
    print(f"TOTAL SERVED PARAMETERS           : {total_params:>12,} params")
    print(f"MARGIN UNDER 150,000,000 CAP      : {margin:>12,} params")

    assert backbone_params == analytical_backbone_params == 149_014_272, (
        f"Unexpected backbone params: {backbone_params}"
    )
    assert head_params == 922_633, f"Unexpected head params: {head_params}"
    assert total_params == 149_936_905, f"Unexpected total params: {total_params}"
    assert total_params <= 150_000_000, f"Budget violated: {total_params}"

    # Verify non-zero set-attention gate permutation equivariance across K=10 options
    torch.manual_seed(42)
    nn.init.constant_(head.set_gate, 0.5)
    h_ctx = torch.randn(1, D)
    h_opts = torch.randn(10, D)
    perm = torch.randperm(10)

    with torch.no_grad():
        raw_1, cal_1 = head(h_ctx, h_opts, qtype_id=0)
        raw_2, cal_2 = head(h_ctx, h_opts[perm], qtype_id=0)
        max_raw_diff = (raw_1[perm] - raw_2).abs().max().item()
        max_cal_diff = (cal_1[perm] - cal_2).abs().max().item()

    print(f"Permutation Equivariance Max Diff : {max_cal_diff:.3e}")
    assert max_raw_diff < 1e-5, f"Raw equivariance violated: {max_raw_diff}"
    assert max_cal_diff < 1e-5, f"Calibrated equivariance violated: {max_cal_diff}"
    print("SUCCESS: Budget (149,936,905 <= 150,000,000) and Equivariance verified.")

    return {
        "backbone_params": backbone_params,
        "head_params": head_params,
        "total_params": total_params,
        "margin": margin,
        "max_equivariance_diff": max_cal_diff,
    }


if __name__ == "__main__":
    verify_exact_budget_and_equivariance()
