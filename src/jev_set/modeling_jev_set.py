# src/jev_set/modeling_jev_set.py
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

from .packing import build_tree_masked_packs

QUESTION_TYPE_TO_ID: dict[str, int] = {
    "choice": 0,
    "multi_label": 1,
    "ranking": 2,
    "Preference": 3,
    "Match": 4,
}

# Maps the 5 benchmark question_type IDs onto the 3 canonical mathematical decision regimes:
#   0 -> Simplex selection (choice, Preference, Match)
#   1 -> Subset selection (multi_label)
#   2 -> Listwise ordering (ranking)
QUESTION_ID_TO_REGIME: tuple[int, ...] = (0, 1, 2, 0, 0)


def resolve_question_type_id(question_type: str | int) -> int:
    if isinstance(question_type, int):
        return max(0, min(question_type, len(QUESTION_ID_TO_REGIME) - 1))
    if question_type in QUESTION_TYPE_TO_ID:
        return QUESTION_TYPE_TO_ID[question_type]
    lowered = question_type.strip().lower()
    alias_map = {
        "choice": 0,
        "multi_label": 1,
        "multilabel": 1,
        "noul": 1,
        "ranking": 2,
        "score": 2,
        "preference": 3,
        "match": 4,
    }
    return alias_map.get(lowered, 0)


def resolve_regime_id(qtype_id: int) -> int:
    if 0 <= qtype_id < len(QUESTION_ID_TO_REGIME):
        return QUESTION_ID_TO_REGIME[qtype_id]
    return qtype_id % 3


@dataclass
class JEVSetConfig:
    backbone_name_or_path: str = "cross-encoder/ettin-reranker-150m-v1"
    vocab_size: int = 50368
    hidden_size: int = 768
    num_hidden_layers: int = 22
    num_attention_heads: int = 12
    intermediate_size: int = 1152
    set_hidden_size: int = 256
    set_num_heads: int = 8
    mlp_width: int = 256
    num_question_types: int = 5
    num_decision_regimes: int = 3
    dropout: float = 0.0
    max_window_tokens: int = 8192
    max_state_tokens: int = 6144
    option_chunk_size: int = 64
    hard_param_budget: int = 150_000_000
    rope_theta: float = 160000.0
    # Ettin/ModernBERT alternating attention: layer i is global iff i % global_attn_every_n_layers == 0,
    # otherwise local (bidirectional sliding window, |pos_i - pos_j| <= local_attention // 2).
    global_attn_every_n_layers: int = 3
    local_attention: int = 128


@dataclass
class BackboneOutput:
    last_hidden_state: torch.Tensor


def _apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies Rotary Position Embeddings (RoPE) to query and key tensors [B, H, L, D_h]."""
    half_d = q.shape[-1] // 2
    q1, q2 = q[..., :half_d], q[..., half_d:]
    k1, k2 = k[..., :half_d], k[..., half_d:]
    q_rot = torch.cat([-q2, q1], dim=-1)
    k_rot = torch.cat([-k2, k1], dim=-1)
    return (q * cos) + (q_rot * sin), (k * cos) + (k_rot * sin)


class EttinEmbeddings(nn.Module):
    """ModernBERT / Ettin token embeddings + bias-free LayerNorm (38,683,392 params)."""

    def __init__(self, config: JEVSetConfig) -> None:
        super().__init__()
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.norm(self.tok_embeddings(input_ids))


class EttinAttention(nn.Module):
    """ModernBERT / Ettin fused Wqkv + Wo attention with tree-mask & virtual-pair RoPE support."""

    def __init__(self, config: JEVSetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.Wqkv = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=False)
        self.Wo = nn.Linear(config.hidden_size, config.hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask_4d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = hidden_states.shape
        qkv = self.Wqkv(hidden_states)
        qkv = qkv.view(bsz, seq_len, 3, self.num_heads, self.head_dim)
        q = qkv[:, :, 0].transpose(1, 2)  # [B, H, L, D_h]
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        q, k = _apply_rotary_pos_emb(q, k, cos, sin)
        attn_out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask_4d,
            dropout_p=0.0,
            is_causal=False,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(bsz, seq_len, self.hidden_size)
        return self.Wo(attn_out)


class EttinGeGLUMLP(nn.Module):
    """ModernBERT / Ettin GeGLU feedforward network: Wi (768 -> 2304) and Wo (1152 -> 768)."""

    def __init__(self, config: JEVSetConfig) -> None:
        super().__init__()
        self.Wi = nn.Linear(config.hidden_size, 2 * config.intermediate_size, bias=False)
        self.Wo = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate, val = self.Wi(hidden_states).chunk(2, dim=-1)
        return self.Wo(F.gelu(gate) * val)


class EttinTransformerLayer(nn.Module):
    """
    Single ModernBERT / Ettin layer.
    Layer 0 omits pre-attention LayerNorm (5,014,272 params);
    Layers 1..21 include pre-attention LayerNorm (5,015,040 params each).
    """

    def __init__(self, config: JEVSetConfig, layer_idx: int) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        if layer_idx > 0:
            self.attn_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        else:
            self.attn_norm = None
        self.attn = EttinAttention(config)
        self.mlp_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.mlp = EttinGeGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask_4d: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        normed_attn_in = self.attn_norm(hidden_states) if self.attn_norm is not None else hidden_states
        hidden_states = hidden_states + self.attn(normed_attn_in, cos, sin, attn_mask_4d)
        hidden_states = hidden_states + self.mlp(self.mlp_norm(hidden_states))
        return hidden_states


class EttinBackbone(nn.Module):
    """
    Exact 22-layer ModernBERT / Ettin encoder backbone (149,014,272 parameters).
    Supports 2D/4D block-diagonal tree attention masks and reset virtual-pair RoPE positions.
    """

    def __init__(self, config: JEVSetConfig) -> None:
        super().__init__()
        self.config = config
        self.embeddings = EttinEmbeddings(config)
        self.layers = nn.ModuleList(
            [EttinTransformerLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.final_norm = nn.LayerNorm(config.hidden_size, eps=1e-5, bias=False)
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.rope_theta = config.rope_theta

    def _compute_rope(
        self,
        position_ids: torch.Tensor,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        device = position_ids.device
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, device=device, dtype=torch.float32) / self.head_dim)
        )
        freqs = torch.einsum("bl,d->bld", position_ids.float(), inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1).unsqueeze(1)  # [B, 1, L, D_h]
        return emb.cos().to(dtype), emb.sin().to(dtype)

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> BackboneOutput:
        bsz, seq_len = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)

        hidden_states = self.embeddings(input_ids)
        cos, sin = self._compute_rope(position_ids, hidden_states.dtype)

        attn_mask_4d: Optional[torch.Tensor] = None
        if attention_mask is not None:
            if attention_mask.dim() == 2:
                attn_mask_4d = attention_mask[:, None, None, :].bool()
            elif attention_mask.dim() == 3:
                attn_mask_4d = attention_mask[:, None, :, :].bool()
            elif attention_mask.dim() == 4:
                attn_mask_4d = attention_mask.bool()

        local_mask_4d: Optional[torch.Tensor] = None
        every_n = max(1, int(self.config.global_attn_every_n_layers))
        if any(i % every_n != 0 for i in range(len(self.layers))) and int(self.config.local_attention) > 0:
            half = int(self.config.local_attention) // 2
            near = (position_ids[:, :, None] - position_ids[:, None, :]).abs() <= half  # [B, L, L]
            near = near[:, None, :, :]
            local_mask_4d = near if attn_mask_4d is None else (attn_mask_4d & near)

        for i, layer in enumerate(self.layers):
            is_global = (i % every_n == 0) or local_mask_4d is None
            hidden_states = layer(hidden_states, cos, sin, attn_mask_4d if is_global else local_mask_4d)

        hidden_states = self.final_norm(hidden_states)
        return BackboneOutput(last_hidden_state=hidden_states)


class EquivariantSetInteractionHead(nn.Module):
    """
    Permutation-equivariant candidate-set readout head (exact 922,633 parameters in FP32).
    Guarantees exact equivariance: permuting the K options identically permutes the K output logits.
    """

    def __init__(self, config: JEVSetConfig) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.d_set = config.set_hidden_size
        self.num_regimes = config.num_decision_regimes

        self.type_emb = nn.Embedding(self.num_regimes, self.d_set)
        self.ctx_proj = nn.Linear(self.hidden_size, self.d_set, bias=True)
        self.opt_proj = nn.Linear(self.hidden_size, self.d_set, bias=True)
        self.ctx_norm = nn.LayerNorm(self.d_set, eps=1e-5, bias=True)
        self.opt_norm = nn.LayerNorm(self.d_set, eps=1e-5, bias=True)

        # Permutation-equivariant inter-option self-attention (no positional encoding across K)
        self.set_attn = nn.MultiheadAttention(
            embed_dim=self.d_set,
            num_heads=config.set_num_heads,
            dropout=config.dropout,
            bias=True,
            batch_first=True,
        )
        self.set_norm = nn.LayerNorm(self.d_set, eps=1e-5, bias=True)

        # Residual direct linear path initialized from cross-encoder/ettin-reranker-150m-v1 head
        self.base_scorer = nn.Linear(self.hidden_size, 1, bias=True)

        # Comparative feature vector per option k:
        # [u_k; c; u_k * c; u_k - mean(u)] -> 4 * d_set (1024) -> mlp_width (256) -> 1
        self.scorer_mlp = nn.Sequential(
            nn.Linear(4 * self.d_set, config.mlp_width, bias=True),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_width, 1, bias=True),
        )

        # Zero-initialized gate so at step 0 the head matches the pretrained reranker linear path
        self.set_gate = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.log_temperature = nn.Parameter(torch.zeros(self.num_regimes, dtype=torch.float32))
        self.cardinality_slope = nn.Parameter(torch.zeros(self.num_regimes, dtype=torch.float32))
        self.register_buffer("multi_label_bias", torch.zeros(1, dtype=torch.float32))

    def pool_option_tokens(
        self,
        option_hidden: torch.Tensor,  # [K, L_opt, 768]
        option_mask: torch.Tensor,    # [K, L_opt] bool
    ) -> torch.Tensor:
        opt_f32 = option_hidden.float()
        attn_logits = self.base_scorer(opt_f32).squeeze(-1)
        attn_logits = attn_logits.masked_fill(~option_mask, -1e9)
        attn_weights = torch.softmax(attn_logits, dim=-1).unsqueeze(-1)
        return (opt_f32 * attn_weights).sum(dim=1)

    def forward(
        self,
        h_ctx: torch.Tensor,   # [768] or [1, 768]
        h_opts: torch.Tensor,  # [K, 768]
        qtype_id: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        h_ctx_f32 = h_ctx.float().view(1, -1)
        h_opts_f32 = h_opts.float()
        K = h_opts_f32.size(0)
        if K == 0:
            empty = h_opts_f32.new_zeros((0,))
            return empty, empty

        regime_id = resolve_regime_id(qtype_id)
        t_emb = self.type_emb.weight[regime_id].float().unsqueeze(0)  # [1, d_set]

        c = self.ctx_norm(self.ctx_proj(h_ctx_f32) + t_emb).expand(K, self.d_set)
        u0 = self.opt_norm(self.opt_proj(h_opts_f32) + t_emb)

        # Set-level attention over the K candidates: (1, K, d_set) -> (1, K, d_set)
        u_attn, _ = self.set_attn(u0.unsqueeze(0), u0.unsqueeze(0), u0.unsqueeze(0), need_weights=False)
        u = self.set_norm(u0 + u_attn.squeeze(0))

        # Mean-centered relative competition feature
        u_mean = u.mean(dim=0, keepdim=True).expand(K, self.d_set)
        comp_features = torch.cat([u, c, u * c, u - u_mean], dim=-1)  # [K, 1024]
        z_set = self.scorer_mlp(comp_features).squeeze(-1)            # [K]
        z_base = self.base_scorer(h_opts_f32).squeeze(-1)             # [K]

        card_scale = 1.0 / math.sqrt(max(1.0, math.log(K + 1.0)))
        raw_logits = z_base + torch.tanh(self.set_gate) * card_scale * z_set

        # Cardinality-aware temperature scaling: T(K, r) = exp(log_temp[r]) * (1 + softplus(card_slope[r]) * log(K/2))
        base_temp = torch.exp(self.log_temperature[regime_id]).clamp(min=0.05, max=10.0)
        slope = F.softplus(self.cardinality_slope[regime_id])
        eff_temp = base_temp * (1.0 + slope * math.log(max(K, 2) / 2.0))

        if regime_id == 1:  # multi_label
            calibrated_logits = (raw_logits - self.multi_label_bias) / eff_temp
        else:
            calibrated_logits = raw_logits / eff_temp

        return raw_logits, calibrated_logits


class JEVSetForDecisionIndex(nn.Module):
    """
    Full JEV-Set-150M model combining the 149,014,272-parameter Ettin/ModernBERT backbone
    with the 922,633-parameter EquivariantSetInteractionHead (total = 149,936,905 parameters).
    """

    def __init__(self, config: JEVSetConfig, load_pretrained: bool = True) -> None:
        super().__init__()
        self.config = config
        self.backbone = EttinBackbone(config)
        self.head = EquivariantSetInteractionHead(config).float()

        if load_pretrained and config.backbone_name_or_path:
            self.load_pretrained_weights(config.backbone_name_or_path)

        if torch.cuda.is_available():
            self.backbone = self.backbone.to(dtype=torch.bfloat16)
        self.head = self.head.float()
        self.verify_parameter_budget()

    def _load_dense4_into_base_scorer(self, dense4_state: dict[str, torch.Tensor]) -> bool:
        w = dense4_state.get("linear.weight", dense4_state.get("weight"))
        b = dense4_state.get("linear.bias", dense4_state.get("bias"))
        loaded = False
        with torch.no_grad():
            if w is not None and w.shape == self.head.base_scorer.weight.shape:
                self.head.base_scorer.weight.copy_(w.float())
                loaded = True
            if b is not None and b.shape == self.head.base_scorer.bias.shape:
                self.head.base_scorer.bias.copy_(b.float())
                loaded = True
        return loaded

    def load_pretrained_weights(self, name_or_path: str | Path) -> dict[str, int]:
        from safetensors.torch import load_file

        ckpt_path = Path(name_or_path)
        safetensors_file: Optional[Path] = None
        ckpt_dir: Optional[Path] = None
        if ckpt_path.is_file() and ckpt_path.suffix == ".safetensors":
            safetensors_file = ckpt_path
            ckpt_dir = ckpt_path.parent
        elif ckpt_path.is_dir() and (ckpt_path / "model.safetensors").exists():
            safetensors_file = ckpt_path / "model.safetensors"
            ckpt_dir = ckpt_path

        if safetensors_file is not None:
            state_dict = load_file(str(safetensors_file))
            has_full_prefix = any(k.startswith("backbone.") or k.startswith("head.") for k in state_dict)
            if has_full_prefix:
                cleaned_full = {
                    (k[len("_orig_mod.") :] if k.startswith("_orig_mod.") else k): v
                    for k, v in state_dict.items()
                }
                incompatible = self.load_state_dict(cleaned_full, strict=False)
                return {
                    "missing_keys": len(incompatible.missing_keys),
                    "unexpected_keys": len(incompatible.unexpected_keys),
                }

            cleaned_backbone = {}
            for k, v in state_dict.items():
                clean_k = k
                for prefix in ("_orig_mod.", "model.", "encoder."):
                    if clean_k.startswith(prefix):
                        clean_k = clean_k[len(prefix) :]
                cleaned_backbone[clean_k] = v
            incompatible = self.backbone.load_state_dict(cleaned_backbone, strict=False)

            if ckpt_dir is not None and (ckpt_dir / "4_Dense" / "model.safetensors").exists():
                self._load_dense4_into_base_scorer(load_file(str(ckpt_dir / "4_Dense" / "model.safetensors")))

            return {
                "missing_keys": len(incompatible.missing_keys),
                "unexpected_keys": len(incompatible.unexpected_keys),
            }

        hf_model = AutoModel.from_pretrained(
            str(name_or_path),
            trust_remote_code=True,
        ).float()  # checkpoint is fp32; never round-trip through bf16 (cast happens once, later, on CUDA)
        cleaned_backbone = {}
        for k, v in hf_model.state_dict().items():
            clean_k = k
            for prefix in ("_orig_mod.", "model.", "encoder."):
                if clean_k.startswith(prefix):
                    clean_k = clean_k[len(prefix) :]
            cleaned_backbone[clean_k] = v
        incompatible = self.backbone.load_state_dict(cleaned_backbone, strict=False)

        try:
            from huggingface_hub import hf_hub_download

            dense4_path = hf_hub_download(repo_id=str(name_or_path), filename="4_Dense/model.safetensors")
            if not self._load_dense4_into_base_scorer(load_file(dense4_path)):
                warnings.warn("4_Dense/model.safetensors found but shapes did not match base_scorer.", stacklevel=2)
        except Exception as exc:  # repo may legitimately have no 4_Dense (e.g. plain encoder checkpoints)
            warnings.warn(f"4_Dense scorer not loaded ({type(exc).__name__}); base_scorer stays at init.", stacklevel=2)

        return {
            "missing_keys": len(incompatible.missing_keys),
            "unexpected_keys": len(incompatible.unexpected_keys),
        }

    def verify_parameter_budget(self) -> dict[str, int]:
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = sum(p.numel() for p in self.head.parameters())
        total_params = backbone_params + head_params
        if total_params > self.config.hard_param_budget:
            raise ValueError(
                f"Parameter budget exceeded: {total_params:,} > {self.config.hard_param_budget:,}"
            )
        return {
            "backbone_params": backbone_params,
            "head_params": head_params,
            "total_params": total_params,
            "margin_below_150m": self.config.hard_param_budget - total_params,
        }


def forward_single_request(
    model: JEVSetForDecisionIndex,
    tokenizer: Any,
    state: str,
    instructions: str,
    options: list[str] | tuple[str, ...],
    question_type: str,
    device: torch.device | str,
) -> tuple[torch.Tensor, torch.Tensor]:
    dev = torch.device(device) if isinstance(device, str) else device
    if len(options) == 0:
        empty = torch.zeros((0,), dtype=torch.float32, device=dev)
        return empty, empty

    window_packs = build_tree_masked_packs(
        tokenizer=tokenizer,
        state=state,
        instructions=instructions,
        options=options,
        max_window_tokens=model.config.max_window_tokens,
        max_state_tokens=model.config.max_state_tokens,
        option_chunk_size=model.config.option_chunk_size,
        device=dev,
    )
    qtype_id = resolve_question_type_id(question_type)

    win_ctx_list: list[torch.Tensor] = []
    win_opts_list: list[torch.Tensor] = []
    use_bf16_autocast = dev.type == "cuda" and torch.cuda.is_bf16_supported()

    for chunk_list in window_packs:
        chunk_opt_vecs: list[torch.Tensor] = []
        ctx_vec_win: Optional[torch.Tensor] = None
        for pack in chunk_list:
            with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=use_bf16_autocast):
                outputs = model.backbone(
                    input_ids=pack.input_ids,
                    position_ids=pack.position_ids,
                    attention_mask=pack.attention_mask_2d,
                )
                hidden = outputs.last_hidden_state[0]

            if ctx_vec_win is None:
                c_start, c_end = pack.ctx_span
                ctx_mean = hidden[c_start:c_end].float().mean(dim=0)
                cls_vec = hidden[0].float()
                ctx_vec_win = 0.5 * (ctx_mean + cls_vec)

            for o_start, o_end in pack.option_spans:
                span_h = hidden[o_start:o_end].unsqueeze(0)
                span_m = torch.ones((1, o_end - o_start), dtype=torch.bool, device=dev)
                opt_vec = model.head.pool_option_tokens(span_h, span_m).squeeze(0)
                chunk_opt_vecs.append(opt_vec)

        assert ctx_vec_win is not None
        win_ctx_list.append(ctx_vec_win)
        win_opts_list.append(torch.stack(chunk_opt_vecs, dim=0))

    h_ctx = torch.stack(win_ctx_list, dim=0).max(dim=0).values
    h_opts = torch.stack(win_opts_list, dim=0).max(dim=0).values

    return model.head(h_ctx, h_opts, qtype_id=qtype_id)
