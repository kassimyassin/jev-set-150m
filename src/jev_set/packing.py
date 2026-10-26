# src/jev_set/packing.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class PackedOptionTreeBatch:
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask_2d: torch.Tensor
    ctx_span: tuple[int, int]
    option_spans: list[tuple[int, int]]


class DeterministicWhitespaceTokenizer:
    """
    Fast, zero-network deterministic tokenizer mapping words/tokens to ModernBERT/Ettin vocab IDs
    [4 .. 50280] with cls_token_id=50281, sep_token_id=50282, unk_token_id=0.
    Used for offline verification and unit tests when load_pretrained=False.
    """

    cls_token_id: int = 50281
    sep_token_id: int = 50282
    unk_token_id: int = 0
    pad_token_id: int = 1
    vocab_size: int = 50368

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        if not text or not text.strip():
            ids = [self.unk_token_id]
        else:
            ids = []
            for tok in text.strip().split():
                digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=4).digest()
                val = int.from_bytes(digest, "little")
                ids.append(4 + (val % 50276))
        if add_special_tokens:
            return [self.cls_token_id] + ids + [self.sep_token_id]
        return ids


def build_tree_masked_packs(
    tokenizer: Any,
    state: str,
    instructions: str,
    options: tuple[str, ...] | list[str],
    max_window_tokens: int = 8192,
    max_state_tokens: int = 6144,
    option_chunk_size: int = 64,
    device: torch.device | str = "cpu",
) -> list[list[PackedOptionTreeBatch]]:
    """
    Packs a decision request (state, instructions, K options) into tree-masked virtual-pair batches.
    - Never truncates state, instructions, or options.
    - Splits ultra-long states (> max_state_tokens) into overlapping windows anchored by the first
      512 state tokens.
    - Splits large candidate sets (K > option_chunk_size or exceeding max_window_tokens) into
      multiple tree-masked chunks.
    - Resets position_ids for every option segment to [P .. P + seg_len - 1] and constructs a 2D
      block-diagonal tree mask so option_k attends only to the shared state prefix and itself.
    """
    cls_id = getattr(tokenizer, "cls_token_id", None)
    if cls_id is None:
        cls_id = 50281
    sep_id = getattr(tokenizer, "sep_token_id", None)
    if sep_id is None:
        sep_id = 50282
    unk_id = getattr(tokenizer, "unk_token_id", None)
    if unk_id is None:
        unk_id = 0

    state_clean = state if state and state.strip() else " "
    inst_clean = instructions if instructions and instructions.strip() else " "

    full_state_ids = list(tokenizer.encode(state_clean, add_special_tokens=False))
    if not full_state_ids:
        full_state_ids = [unk_id]

    q_prefix_ids = list(
        tokenizer.encode(f"Question: {inst_clean}\nOption: ", add_special_tokens=False)
    )
    if not q_prefix_ids:
        q_prefix_ids = [unk_id]

    opt_token_lists: list[list[int]] = []
    for opt in options:
        opt_clean = opt if opt and opt.strip() else " "
        oids = list(tokenizer.encode(opt_clean, add_special_tokens=False))
        if not oids:
            oids = [unk_id]
        opt_token_lists.append(oids)

    max_seg_len = max(
        (len(q_prefix_ids) + len(oids) + 1 for oids in opt_token_lists),
        default=0,
    )
    effective_max_state = min(
        max_state_tokens,
        max(64, max_window_tokens - 2 - max_seg_len),
    )

    state_windows: list[list[int]] = []
    if len(full_state_ids) <= effective_max_state:
        state_windows.append(full_state_ids)
    else:
        anchor_len = min(512, max(1, effective_max_state // 4))
        head_ids = full_state_ids[:anchor_len]
        body_ids = full_state_ids[anchor_len:]
        win_len = max(1, effective_max_state - anchor_len)
        overlap = min(1024, max(0, win_len // 2))
        stride = max(1, win_len - overlap)
        for start in range(0, len(body_ids), stride):
            chunk = head_ids + body_ids[start : start + win_len]
            state_windows.append(chunk)
            if start + win_len >= len(body_ids):
                break

    all_window_packs: list[list[PackedOptionTreeBatch]] = []
    for state_win in state_windows:
        prefix_ids = [cls_id] + state_win + [sep_id]
        P = len(prefix_ids)
        ctx_span = (1, P - 1)

        window_chunks: list[PackedOptionTreeBatch] = []
        curr_opt_indices: list[int] = []
        curr_tokens_budget = P

        def flush_chunk(opt_idx_list: list[int]) -> PackedOptionTreeBatch:
            seq_ids = list(prefix_ids)
            pos_ids = list(range(P))
            opt_spans: list[tuple[int, int]] = []
            seg_bounds: list[tuple[int, int]] = []

            for idx in opt_idx_list:
                seg_start = len(seq_ids)
                oids = opt_token_lists[idx]
                seg_tokens = q_prefix_ids + oids + [sep_id]
                opt_start = seg_start + len(q_prefix_ids)
                opt_end = seg_start + len(q_prefix_ids) + len(oids)
                seq_ids.extend(seg_tokens)
                pos_ids.extend(range(P, P + len(seg_tokens)))
                opt_spans.append((opt_start, opt_end))
                seg_bounds.append((seg_start, len(seq_ids)))

            L_tot = len(seq_ids)
            mask_2d = torch.zeros((L_tot, L_tot), dtype=torch.bool, device=device)
            mask_2d[:P, :P] = True
            for s_start, s_end in seg_bounds:
                mask_2d[s_start:s_end, :P] = True
                mask_2d[s_start:s_end, s_start:s_end] = True

            return PackedOptionTreeBatch(
                input_ids=torch.tensor([seq_ids], dtype=torch.long, device=device),
                position_ids=torch.tensor([pos_ids], dtype=torch.long, device=device),
                attention_mask_2d=mask_2d.unsqueeze(0).unsqueeze(0),
                ctx_span=ctx_span,
                option_spans=opt_spans,
            )

        if not opt_token_lists:
            window_chunks.append(flush_chunk([]))
        else:
            for k_idx, oids in enumerate(opt_token_lists):
                seg_len = len(q_prefix_ids) + len(oids) + 1
                if curr_opt_indices and (
                    curr_tokens_budget + seg_len > max_window_tokens
                    or len(curr_opt_indices) >= option_chunk_size
                ):
                    window_chunks.append(flush_chunk(curr_opt_indices))
                    curr_opt_indices = []
                    curr_tokens_budget = P
                curr_opt_indices.append(k_idx)
                curr_tokens_budget += seg_len

            if curr_opt_indices:
                window_chunks.append(flush_chunk(curr_opt_indices))

        all_window_packs.append(window_chunks)

    return all_window_packs
