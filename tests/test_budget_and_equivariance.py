# tests/test_budget_and_equivariance.py
from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from scripts.verify_params import verify_exact_budget_and_equivariance
from src.jev_set.modeling_jev_set import (
    QUESTION_TYPE_TO_ID,
    EquivariantSetInteractionHead,
    JEVSetConfig,
    JEVSetForDecisionIndex,
    forward_single_request,
)
from src.jev_set.packing import DeterministicWhitespaceTokenizer, build_tree_masked_packs


def test_exact_150m_parameter_budget() -> None:
    report = verify_exact_budget_and_equivariance()
    assert report["backbone_params"] == 149_014_272
    assert report["head_params"] == 922_633
    assert report["total_params"] == 149_936_905
    assert report["total_params"] <= 150_000_000
    assert report["margin"] == 63_095


def test_budget_violation_raises_value_error() -> None:
    strict_cfg = JEVSetConfig(hard_param_budget=149_000_000)
    with pytest.raises(ValueError, match="Parameter budget exceeded"):
        JEVSetForDecisionIndex(strict_cfg, load_pretrained=False)


@pytest.mark.parametrize("num_options", [2, 10, 77, 151])
@pytest.mark.parametrize("qtype_name,qtype_id", list(QUESTION_TYPE_TO_ID.items()))
def test_head_permutation_equivariance_all_types_and_cardinalities(
    num_options: int,
    qtype_name: str,
    qtype_id: int,
) -> None:
    torch.manual_seed(100 + num_options + qtype_id)
    cfg = JEVSetConfig()
    head = EquivariantSetInteractionHead(cfg).eval()
    nn.init.constant_(head.set_gate, 0.65)
    nn.init.uniform_(head.log_temperature, -0.3, 0.3)
    nn.init.uniform_(head.cardinality_slope, 0.1, 0.8)

    h_ctx = torch.randn(1, cfg.hidden_size)
    h_opts = torch.randn(num_options, cfg.hidden_size)
    perm = torch.randperm(num_options)

    with torch.no_grad():
        raw_orig, cal_orig = head(h_ctx, h_opts, qtype_id=qtype_id)
        raw_perm, cal_perm = head(h_ctx, h_opts[perm], qtype_id=qtype_id)

    raw_diff = (raw_orig[perm] - raw_perm).abs().max().item()
    cal_diff = (cal_orig[perm] - cal_perm).abs().max().item()
    assert raw_diff < 1e-5, f"Raw equivariance failed for {qtype_name} (K={num_options}): {raw_diff}"
    assert cal_diff < 1e-5, f"Calibrated equivariance failed for {qtype_name} (K={num_options}): {cal_diff}"

    # Verify 0.0% top-1 flip rate under permutation
    orig_winner_in_perm = int((perm == int(torch.argmax(cal_orig).item())).nonzero(as_tuple=True)[0].item())
    perm_winner = int(torch.argmax(cal_perm).item())
    assert orig_winner_in_perm == perm_winner


def test_tree_mask_isolation_and_virtual_pair_positions() -> None:
    tokenizer = DeterministicWhitespaceTokenizer()
    state = "Shared system context with several tokens for testing."
    instructions = "Choose the best candidate action."
    options = ["First option alpha", "Second option beta with more words", "Third option gamma"]

    packs = build_tree_masked_packs(
        tokenizer=tokenizer,
        state=state,
        instructions=instructions,
        options=options,
        device="cpu",
    )
    assert len(packs) == 1
    assert len(packs[0]) == 1
    batch = packs[0][0]

    c_start, c_end = batch.ctx_span
    P = c_end + 1  # [CLS] + state + [SEP]
    mask = batch.attention_mask_2d[0, 0]  # [L, L]
    pos = batch.position_ids[0]

    # Prefix tokens only attend to prefix [0 .. P-1]
    assert torch.all(mask[:P, :P])
    assert not torch.any(mask[:P, P:])

    # Each option segment starts its position_ids at P and cannot attend to other options
    assert len(batch.option_spans) == 3
    for i, (o_start_i, o_end_i) in enumerate(batch.option_spans):
        assert torch.all(mask[o_start_i:o_end_i, :P])
        assert torch.all(mask[o_start_i:o_end_i, o_start_i:o_end_i])
        for j, (o_start_j, o_end_j) in enumerate(batch.option_spans):
            if i != j:
                assert not torch.any(mask[o_start_i:o_end_i, o_start_j:o_end_j])

    # Verify virtual-pair position_ids reset for each option segment
    q_prefix_len = len(tokenizer.encode(f"Question: {instructions}\nOption: ", add_special_tokens=False))
    for o_start, _ in batch.option_spans:
        seg_start = o_start - q_prefix_len
        assert int(pos[seg_start].item()) == P


def test_end_to_end_forward_single_request_equivariance_across_chunks() -> None:
    torch.manual_seed(2026)
    # Use a compact 2-layer config with option_chunk_size=3 to test cross-chunk + backbone + head equivariance
    cfg = JEVSetConfig(num_hidden_layers=2, option_chunk_size=3)
    model = JEVSetForDecisionIndex(cfg, load_pretrained=False).eval()
    nn.init.constant_(model.head.set_gate, 0.5)
    tokenizer = DeterministicWhitespaceTokenizer()

    state = "Customer wants to dispute an unauthorized international credit card transaction."
    instructions = "Classify the banking intent."
    options = [
        "card_not_working",
        "compromised_card_dispute",
        "transfer_not_received_by_recipient",
        "exchange_rate_inquiry",
        "lost_or_stolen_phone",
        "pending_cash_withdrawal",
        "declined_card_payment",
    ]
    perm = [3, 1, 6, 0, 5, 2, 4]
    perm_t = torch.tensor(perm, dtype=torch.long)
    options_perm = [options[i] for i in perm]

    with torch.no_grad():
        raw_1, cal_1 = forward_single_request(
            model=model,
            tokenizer=tokenizer,
            state=state,
            instructions=instructions,
            options=options,
            question_type="choice",
            device="cpu",
        )
        raw_2, cal_2 = forward_single_request(
            model=model,
            tokenizer=tokenizer,
            state=state,
            instructions=instructions,
            options=options_perm,
            question_type="choice",
            device="cpu",
        )

    assert (raw_1[perm_t] - raw_2).abs().max().item() < 1e-5
    assert (cal_1[perm_t] - cal_2).abs().max().item() < 1e-5


def test_modernbert_backbone_exact_numerical_equivalence() -> None:
    from transformers import ModernBertConfig, ModernBertModel

    torch.manual_seed(77)
    cfg = JEVSetConfig(num_hidden_layers=2)
    assert cfg.rope_theta == 160000.0
    jev_model = JEVSetForDecisionIndex(cfg, load_pretrained=False).eval()

    mb_cfg = ModernBertConfig(
        vocab_size=cfg.vocab_size,
        hidden_size=cfg.hidden_size,
        intermediate_size=cfg.intermediate_size,
        num_hidden_layers=cfg.num_hidden_layers,
        num_attention_heads=cfg.num_attention_heads,
        global_attn_every_n_layers=1,
    )
    mb_model = ModernBertModel(mb_cfg).eval()

    assert set(jev_model.backbone.state_dict().keys()) == set(mb_model.state_dict().keys())
    jev_model.backbone.load_state_dict(mb_model.state_dict(), strict=True)

    input_ids = torch.randint(10, 1000, (1, 24))
    with torch.no_grad():
        out_mb = mb_model(input_ids).last_hidden_state
        out_jev = jev_model.backbone(input_ids).last_hidden_state
    assert (out_mb - out_jev).abs().max().item() < 1e-5


def test_checkpoint_save_and_reload_preserves_head_and_dense4_weights(tmp_path) -> None:
    from safetensors.torch import save_file

    # 1. Full JEVSetForDecisionIndex checkpoint round-trip preserves head parameters
    full_dir = tmp_path / "full_ckpt"
    full_dir.mkdir(parents=True, exist_ok=True)
    cfg = JEVSetConfig(backbone_name_or_path=str(full_dir), num_hidden_layers=2)
    m_orig = JEVSetForDecisionIndex(cfg, load_pretrained=False)
    with torch.no_grad():
        m_orig.head.set_gate.fill_(0.85)
        m_orig.head.log_temperature.copy_(torch.tensor([0.15, -0.25, 0.35]))
        m_orig.head.cardinality_slope.copy_(torch.tensor([0.40, 0.50, 0.60]))
    save_file(m_orig.state_dict(), str(full_dir / "model.safetensors"))

    m_reloaded = JEVSetForDecisionIndex(cfg, load_pretrained=True)
    assert torch.isclose(m_reloaded.head.set_gate, torch.tensor([0.85]))
    assert torch.allclose(m_reloaded.head.log_temperature, torch.tensor([0.15, -0.25, 0.35]))
    assert torch.allclose(m_reloaded.head.cardinality_slope, torch.tensor([0.40, 0.50, 0.60]))

    # 2. Unprefixed ModernBertModel + 4_Dense/model.safetensors initializes backbone & base_scorer
    st_dir = tmp_path / "st_reranker_ckpt"
    (st_dir / "4_Dense").mkdir(parents=True, exist_ok=True)
    save_file(m_orig.backbone.state_dict(), str(st_dir / "model.safetensors"))
    dense4_w = torch.full((1, cfg.hidden_size), 0.125)
    dense4_b = torch.tensor([0.42])
    save_file({"linear.weight": dense4_w, "linear.bias": dense4_b}, str(st_dir / "4_Dense" / "model.safetensors"))

    st_cfg = JEVSetConfig(backbone_name_or_path=str(st_dir), num_hidden_layers=2)
    m_from_st = JEVSetForDecisionIndex(st_cfg, load_pretrained=True)
    assert torch.allclose(m_from_st.head.base_scorer.weight, dense4_w)
    assert torch.allclose(m_from_st.head.base_scorer.bias, dense4_b)
