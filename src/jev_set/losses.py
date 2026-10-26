# src/jev_set/losses.py
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def zlpr_multi_label_loss(logits: torch.Tensor, multi_hot_targets: torch.Tensor) -> torch.Tensor:
    """
    Zero-bounded Log-sum-exp Pairwise Ranking (ZLPR) loss (Su et al., 2022) for multi_label tasks.
    Enforces that all positive options score above thresholding zero and all negative options score below zero.
    """
    pos_mask = multi_hot_targets > 0.5
    neg_mask = ~pos_mask
    pos_term = (
        torch.log1p(torch.exp(-logits[pos_mask]).sum())
        if pos_mask.any()
        else logits.new_zeros(())
    )
    neg_term = (
        torch.log1p(torch.exp(logits[neg_mask]).sum())
        if neg_mask.any()
        else logits.new_zeros(())
    )
    return pos_term + neg_term


def lambdarank_ndcg_loss(
    logits: torch.Tensor,
    relevance_grades: torch.Tensor,
    top_k: int = 10,
    sigma: float = 1.0,
) -> torch.Tensor:
    """
    Listwise LambdaRank loss weighted by |Delta nDCG@top_k| for retrieval/ranking benchmarks (BRIGHT, ToolRet).
    """
    K = logits.size(0)
    if K <= 1:
        return logits.sum() * 0.0

    rel = relevance_grades.float().to(logits.device)
    gains = (2.0 ** rel) - 1.0
    if gains.max() <= 0:
        return logits.sum() * 0.0

    # Ideal DCG@top_k
    sorted_gains, _ = torch.sort(gains, descending=True)
    k_eff = min(K, top_k)
    discounts = 1.0 / torch.log2(torch.arange(2, K + 2, dtype=torch.float32, device=logits.device))
    idcg = (sorted_gains[:k_eff] * discounts[:k_eff]).sum().clamp(min=1e-8)

    # Current predicted ranks
    pred_order = torch.argsort(logits, descending=True)
    ranks = torch.zeros(K, dtype=torch.long, device=logits.device)
    ranks[pred_order] = torch.arange(K, device=logits.device)
    rank_discounts = torch.where(
        ranks < top_k,
        discounts[ranks],
        torch.zeros_like(discounts),
    )

    gain_diff = gains.unsqueeze(1) - gains.unsqueeze(0)  # [K, K]
    disc_diff = rank_discounts.unsqueeze(1) - rank_discounts.unsqueeze(0)  # [K, K]
    delta_ndcg = ((gain_diff * disc_diff).abs() / idcg).detach()

    valid_pairs = gain_diff > 0
    if not valid_pairs.any():
        return logits.sum() * 0.0

    score_diff = logits.unsqueeze(1) - logits.unsqueeze(0)
    pairwise_loss = F.softplus(-sigma * score_diff) * delta_ndcg
    return pairwise_loss[valid_pairs].mean()


class JEVCompositeDecisionLoss(nn.Module):
    """
    Unified multi-regime loss combining:
      1. Cross-Entropy + Simplex KL Distillation (tau_KD = 2.0)
      2. Pairwise max-margin hinge loss (margin = 1.0)
      3. Multi-label BCE + ZLPR loss
      4. Listwise LambdaRank loss for nDCG@10 ranking tasks
    """

    def __init__(
        self,
        kd_alpha: float = 0.50,
        kd_temperature: float = 2.0,
        margin: float = 1.0,
        margin_weight: float = 0.15,
        zlpr_weight: float = 0.35,
        lambdarank_weight: float = 0.25,
    ) -> None:
        super().__init__()
        self.kd_alpha = kd_alpha
        self.kd_temperature = kd_temperature
        self.margin = margin
        self.margin_weight = margin_weight
        self.zlpr_weight = zlpr_weight
        self.lambdarank_weight = lambdarank_weight

    def forward(
        self,
        raw_logits: torch.Tensor,
        calibrated_logits: torch.Tensor,
        question_type: str,
        gold_indices: list[int],
        teacher_probs: Optional[torch.Tensor] = None,
        difficulty_weight: float = 1.0,
        relevance_grades: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        K = calibrated_logits.size(0)
        device = calibrated_logits.device
        if K == 0 or (not gold_indices and relevance_grades is None):
            return calibrated_logits.sum() * 0.0

        if question_type == "multi_label":
            target_vec = torch.zeros(K, dtype=torch.float32, device=device)
            for idx in gold_indices:
                if 0 <= idx < K:
                    target_vec[idx] = 1.0
            bce = F.binary_cross_entropy_with_logits(calibrated_logits, target_vec)
            zlpr = zlpr_multi_label_loss(calibrated_logits, target_vec)
            loss = bce + self.zlpr_weight * zlpr
            if teacher_probs is not None and self.kd_alpha > 0.0:
                t_probs = teacher_probs.to(device).float().clamp(1e-6, 1.0 - 1e-6)
                kd_bce = F.binary_cross_entropy_with_logits(calibrated_logits, t_probs)
                loss = (1.0 - self.kd_alpha) * loss + self.kd_alpha * kd_bce
            return difficulty_weight * loss

        gold_idx = gold_indices[0] if gold_indices else int(torch.argmax(relevance_grades).item())
        target_tensor = torch.tensor([gold_idx], dtype=torch.long, device=device)
        ce_loss = F.cross_entropy(calibrated_logits.unsqueeze(0), target_tensor)

        if K > 1:
            gold_score = raw_logits[gold_idx]
            mask = torch.ones(K, dtype=torch.bool, device=device)
            mask[gold_idx] = False
            hinge = F.relu(self.margin - (gold_score - raw_logits[mask])).mean()
        else:
            hinge = raw_logits.new_zeros(())

        loss = ce_loss + self.margin_weight * hinge

        if question_type == "ranking":
            if relevance_grades is None:
                rel = torch.zeros(K, dtype=torch.float32, device=device)
                for rank_pos, idx in enumerate(gold_indices):
                    if 0 <= idx < K:
                        rel[idx] = max(1.0, float(len(gold_indices) - rank_pos))
            else:
                rel = relevance_grades.to(device).float()
            loss = loss + self.lambdarank_weight * lambdarank_ndcg_loss(raw_logits, rel, top_k=10)

        if teacher_probs is not None and self.kd_alpha > 0.0:
            T = self.kd_temperature
            student_log_probs = F.log_softmax(calibrated_logits / T, dim=-1)
            teacher_target = teacher_probs.to(device).float().clamp(min=1e-7) ** (1.0 / T)
            teacher_target = teacher_target / teacher_target.sum()
            kl_loss = F.kl_div(student_log_probs, teacher_target, reduction="sum") * (T * T)
            loss = (1.0 - self.kd_alpha) * loss + self.kd_alpha * kl_loss

        return difficulty_weight * loss
