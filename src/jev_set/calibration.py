# src/jev_set/calibration.py
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from .modeling_jev_set import resolve_regime_id


@dataclass
class TemperatureCalibrationResult:
    log_temperature: torch.Tensor
    cardinality_slope: torch.Tensor
    ece_before: float
    ece_after: float
    brier_before: float
    brier_after: float


def apply_cardinality_calibration(
    raw_logits: torch.Tensor,
    qtype_id: int,
    log_temperature: torch.Tensor,
    cardinality_slope: torch.Tensor,
    multi_label_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Applies cardinality-aware temperature scaling:
      T(K, r) = clamp(exp(log_temp[r]), 0.05, 10.0) * (1 + softplus(card_slope[r]) * log(max(K, 2) / 2))
    """
    K = raw_logits.numel()
    if K == 0:
        return raw_logits
    idx = qtype_id if qtype_id < log_temperature.numel() else resolve_regime_id(qtype_id)
    base_temp = torch.exp(log_temperature[idx]).clamp(min=0.05, max=10.0)
    slope = F.softplus(cardinality_slope[idx])
    eff_temp = base_temp * (1.0 + slope * math.log(max(K, 2) / 2.0))
    if resolve_regime_id(qtype_id) == 1 and multi_label_bias is not None:
        return (raw_logits - multi_label_bias) / eff_temp
    return raw_logits / eff_temp


def select_multi_label_indices(
    calibrated_logits: torch.Tensor,
    select_k: Optional[int] = None,
    threshold: float = 0.50,
    max_gap: float = 1.35,
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[float, ...]]:
    """
    Returns (selected_indices, ranked_indices, option_scores) for multi_label requests.
    Enforces non-empty selection fallback to top-1 if no option exceeds threshold.
    """
    cal_f64 = calibrated_logits.to(torch.float64).cpu()
    K = cal_f64.numel()
    if K == 0:
        return (), (), ()

    ranked_tensor = torch.argsort(-cal_f64, stable=True)
    ranked_indices = tuple(int(i) for i in ranked_tensor.tolist())
    probs = torch.sigmoid(cal_f64).clamp(1e-7, 1.0 - 1e-7)

    if select_k is not None and select_k > 0:
        selected_indices = ranked_indices[: min(select_k, K)]
    else:
        max_logit = cal_f64.max().item()
        chosen = [
            i
            for i in range(K)
            if probs[i].item() >= threshold and cal_f64[i].item() >= (max_logit - max_gap)
        ]
        if not chosen:
            chosen = [ranked_indices[0]]
        selected_indices = tuple(sorted(chosen))

    option_scores = tuple(float(p) for p in probs.tolist())
    return selected_indices, ranked_indices, option_scores


def compute_brier_score(
    probs_list: list[torch.Tensor],
    gold_indices: list[int],
) -> float:
    if not probs_list:
        return 0.0
    total = 0.0
    for probs, y in zip(probs_list, gold_indices):
        p64 = probs.to(torch.float64)
        one_hot = torch.zeros_like(p64)
        if 0 <= y < p64.numel():
            one_hot[y] = 1.0
        total += float(((p64 - one_hot) ** 2).sum().item())
    return total / len(probs_list)


def compute_ece(
    probs_list: list[torch.Tensor],
    gold_indices: list[int],
    num_bins: int = 15,
) -> float:
    if not probs_list:
        return 0.0
    confidences: list[float] = []
    accuracies: list[float] = []
    for probs, y in zip(probs_list, gold_indices):
        pred = int(torch.argmax(probs).item())
        conf = float(probs[pred].item())
        confidences.append(conf)
        accuracies.append(1.0 if pred == y else 0.0)

    conf_t = torch.tensor(confidences, dtype=torch.float64)
    acc_t = torch.tensor(accuracies, dtype=torch.float64)
    bin_boundaries = torch.linspace(0.0, 1.0, num_bins + 1, dtype=torch.float64)
    ece = 0.0
    n = float(len(confidences))
    for b_idx in range(num_bins):
        low, high = bin_boundaries[b_idx], bin_boundaries[b_idx + 1]
        if b_idx == 0:
            in_bin = (conf_t >= low) & (conf_t <= high)
        else:
            in_bin = (conf_t > low) & (conf_t <= high)
        count = int(in_bin.sum().item())
        if count > 0:
            avg_conf = float(conf_t[in_bin].mean().item())
            avg_acc = float(acc_t[in_bin].mean().item())
            ece += (count / n) * abs(avg_acc - avg_conf)
    return ece


def fit_cardinality_temperatures_lbfgs(
    oof_raw_logits: list[torch.Tensor],
    oof_gold_indices: list[int],
    oof_qtype_ids: list[int],
    num_slots: int = 3,
    max_iter: int = 200,
    lr: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fits cardinality-aware temperature parameters (log_temperature, cardinality_slope) via L-BFGS
    minimizing 0.70 * NLL + 0.30 * Brier + 1e-4 * L2.
    """
    log_temp = torch.zeros(num_slots, dtype=torch.float64, requires_grad=True)
    card_slope = torch.zeros(num_slots, dtype=torch.float64, requires_grad=True)
    if not oof_raw_logits:
        return log_temp.detach().float(), card_slope.detach().float()

    optimizer = torch.optim.LBFGS(
        [log_temp, card_slope],
        lr=lr,
        max_iter=max_iter,
        line_search_fn="strong_wolfe",
    )

    def closure() -> torch.Tensor:
        optimizer.zero_grad()
        total_loss = torch.zeros((), dtype=torch.float64)
        for z, y, qid in zip(oof_raw_logits, oof_gold_indices, oof_qtype_ids):
            z64 = z.to(torch.float64)
            K = z64.numel()
            slot = qid if qid < num_slots else resolve_regime_id(qid)
            t_base = torch.exp(log_temp[slot]).clamp(0.05, 10.0)
            slope = F.softplus(card_slope[slot])
            temp = t_base * (
                1.0 + slope * torch.log(torch.tensor(max(K, 2) / 2.0, dtype=torch.float64))
            )
            scaled = z64 / temp
            nll = F.cross_entropy(scaled.unsqueeze(0), torch.tensor([y], dtype=torch.long))
            probs = torch.softmax(scaled, dim=-1)
            one_hot = torch.zeros_like(probs)
            if 0 <= y < K:
                one_hot[y] = 1.0
            brier = ((probs - one_hot) ** 2).sum()
            total_loss = total_loss + (0.7 * nll + 0.3 * brier)
        reg = 1e-4 * ((log_temp ** 2).sum() + (card_slope ** 2).sum())
        loss = total_loss / len(oof_raw_logits) + reg
        loss.backward()
        return loss

    optimizer.step(closure)
    return log_temp.detach().float(), card_slope.detach().float()
