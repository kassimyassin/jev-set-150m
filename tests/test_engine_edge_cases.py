# tests/test_engine_edge_cases.py
from __future__ import annotations

from pathlib import Path

import pytest
import torch

from decision_index.engines.base import validate
from scripts.prepare_data import ThreeTierDecontaminator
from scripts.soup_and_calibrate import greedy_model_soup
from src.jev_set.calibration import (
    compute_brier_score,
    compute_ece,
    fit_cardinality_temperatures_lbfgs,
    select_multi_label_indices,
)
from src.jev_set.engine import EngineRequest, JEVSetEngine
from src.jev_set.losses import JEVCompositeDecisionLoss, lambdarank_ndcg_loss, zlpr_multi_label_loss
from src.jev_set.modeling_jev_set import JEVSetConfig
from src.jev_set.packing import DeterministicWhitespaceTokenizer, build_tree_masked_packs


@pytest.fixture(scope="module")
def fast_engine() -> JEVSetEngine:
    """Engine instance with 2 transformer layers for fast CPU edge-case testing."""
    cfg = JEVSetConfig(num_hidden_layers=2)
    return JEVSetEngine(
        checkpoint_dir="offline-test",
        device="cpu",
        load_pretrained=False,
        config=cfg,
    )


def test_full_engine_info_reports_exact_149936905_params() -> None:
    full_engine = JEVSetEngine(
        checkpoint_dir="offline-full-150m",
        device="cpu",
        load_pretrained=False,
    )
    assert full_engine.info.name == "JEV-Set-150M"
    assert full_engine.info.parameters == 149_936_905
    assert full_engine.info.context_length == 32768


def test_k512_options_packing_and_scoring(fast_engine: JEVSetEngine) -> None:
    options_512 = [f"option_candidate_{i:03d}" for i in range(512)]
    packs = build_tree_masked_packs(
        tokenizer=fast_engine.tokenizer,
        state="Select one option out of 512 candidates.",
        instructions="Pick the matching candidate index.",
        options=options_512,
        option_chunk_size=64,
        device="cpu",
    )
    # 512 options with option_chunk_size=64 -> 8 chunks in 1 state window
    assert len(packs) == 1
    assert len(packs[0]) == 8
    assert sum(len(chunk.option_spans) for chunk in packs[0]) == 512

    req = EngineRequest(
        request_id="req_k512",
        state="Select one option out of 512 candidates.",
        instructions="Pick the matching candidate index.",
        options=options_512,
        question_type="choice",
    )
    resp = fast_engine.run(req)
    assert resp.status == "ok"
    assert len(resp.option_scores) == 512
    assert len(resp.ranked_indices) == 512
    assert set(resp.ranked_indices) == set(range(512))
    assert len(resp.selected_indices) == 1
    assert resp.selected_indices[0] == resp.ranked_indices[0]
    assert abs(sum(resp.option_scores) - 1.0) < 1e-6


def test_32k_ultra_long_state_windowing_zero_truncation(fast_engine: JEVSetEngine) -> None:
    tokenizer = DeterministicWhitespaceTokenizer()
    long_state = " ".join(f"tok_{i}" for i in range(32_768))
    options = ["Clause A is violated", "Clause B is satisfied", "Neutral contract state"]

    packs = build_tree_masked_packs(
        tokenizer=tokenizer,
        state=long_state,
        instructions="Determine the contract NLI relationship.",
        options=options,
        max_window_tokens=8192,
        max_state_tokens=6144,
        option_chunk_size=64,
        device="cpu",
    )
    # 32,768 tokens with 6,144 max_state_tokens (512 anchor + 5,632 window, stride 4,608) -> 7 windows
    assert len(packs) >= 6
    first_window_anchor = packs[0][0].input_ids[0, 1:513]
    for win_chunks in packs:
        pack = win_chunks[0]
        assert pack.input_ids.shape[1] <= 8192
        # Verify the first 512 state anchor tokens are preserved in every window
        assert torch.equal(pack.input_ids[0, 1:513], first_window_anchor)

    req = EngineRequest(
        request_id="req_32k",
        state=long_state,
        instructions="Determine the contract NLI relationship.",
        options=options,
        question_type="choice",
    )
    resp = fast_engine.run(req)
    assert resp.status == "ok"
    assert len(resp.option_scores) == 3
    assert abs(sum(resp.option_scores) - 1.0) < 1e-6


def test_multi_label_selection_and_non_empty_fallback(fast_engine: JEVSetEngine) -> None:
    # 1. Explicit select_k
    req_k2 = EngineRequest(
        request_id="req_ml_k2",
        state="Multi-label tool selection state.",
        instructions="Select all applicable tools.",
        options=["tool_a", "tool_b", "tool_c", "tool_d"],
        question_type="multi_label",
        select_k=2,
    )
    resp_k2 = fast_engine.run(req_k2)
    assert len(resp_k2.selected_indices) == 2
    assert all(0.0 < s < 1.0 for s in resp_k2.option_scores)

    # 2. All-negative logits must still trigger non-empty top-1 fallback
    neg_logits = torch.tensor([-5.0, -3.2, -8.1, -4.5], dtype=torch.float32)
    selected, ranked, scores = select_multi_label_indices(
        neg_logits,
        select_k=None,
        threshold=0.50,
        max_gap=1.35,
    )
    assert selected == (1,)
    assert ranked[0] == 1
    assert len(scores) == 4


def test_empty_options_and_blank_strings_handled_gracefully(fast_engine: JEVSetEngine) -> None:
    empty_req = EngineRequest(
        request_id="req_empty",
        state="",
        instructions="",
        options=[],
        question_type="choice",
    )
    empty_resp = fast_engine.run(empty_req)
    assert empty_resp.status == "ok"
    assert empty_resp.selected_indices == ()
    assert empty_resp.ranked_indices == ()
    assert empty_resp.option_scores == ()

    blank_text_req = EngineRequest(
        request_id="req_blank",
        state="   ",
        instructions="",
        options=["", "   "],
        question_type="Preference",
    )
    blank_resp = fast_engine.run(blank_text_req)
    assert blank_resp.status == "ok"
    assert len(blank_resp.option_scores) == 2
    assert abs(sum(blank_resp.option_scores) - 1.0) < 1e-6


def test_official_decision_index_runner_protocol(fast_engine: JEVSetEngine) -> None:
    questions = {
        "main_choice": {
            "type": "choice",
            "instructions": "Which option best answers the query?",
            "criteria": {"opt_a": "First answer", "opt_b": "Second answer", "opt_c": "Third answer"},
        },
        "abstention_check": {
            "type": "noul",
            "instructions": "Should the model abstain?",
        },
    }
    response, raw = fast_engine("Sample benchmark state.", questions)
    # Official validator from apolinario/decision-index
    validate(questions, response)
    assert response["answers"]["main_choice"]["choice"] in ("opt_a", "opt_b", "opt_c")
    assert 0.0 <= response["answers"]["abstention_check"]["noul"] <= 1.0
    assert "main_choice" in raw


def test_composite_losses_and_calibration_and_decontaminator(tmp_path: Path) -> None:
    loss_fn = JEVCompositeDecisionLoss(kd_alpha=0.5, kd_temperature=2.0)
    raw = torch.tensor([1.5, -0.5, 0.2, -1.0], requires_grad=True)
    cal = raw * 0.9
    teacher = torch.tensor([0.70, 0.10, 0.15, 0.05])

    for qtype, golds in [
        ("choice", [0]),
        ("multi_label", [0, 2]),
        ("ranking", [0, 2, 1, 3]),
        ("Preference", [0]),
        ("Match", [0]),
    ]:
        l_val = loss_fn(raw, cal, qtype, golds, teacher_probs=teacher, difficulty_weight=1.2)
        assert torch.isfinite(l_val)
        assert l_val.item() > 0.0

    # Calibration L-BFGS & metrics
    oof_logits = [torch.tensor([2.5, -1.0, 0.0]), torch.tensor([-0.5, 3.0, 0.2])] * 8
    oof_golds = [0, 1] * 8
    oof_qtypes = [0, 3] * 8
    lt, cs = fit_cardinality_temperatures_lbfgs(oof_logits, oof_golds, oof_qtypes, max_iter=25)
    assert lt.shape == (3,)
    assert cs.shape == (3,)
    probs_list = [torch.softmax(z, dim=-1) for z in oof_logits]
    assert 0.0 <= compute_ece(probs_list, oof_golds) <= 1.0
    assert 0.0 <= compute_brier_score(probs_list, oof_golds) <= 2.0

    # 3-Tier Decontaminator
    decontam = ThreeTierDecontaminator(jaccard_threshold=0.70, cosine_threshold=0.90)
    decontam.add_reference_text(
        "test_0",
        "The patient presented with acute myocardial infarction and elevated troponin.",
        "What is the primary diagnosis?",
    )
    assert decontam.is_contaminated(
        "The patient presented with acute myocardial infarction and elevated troponin.",
        "What is the primary diagnosis?",
    )
    assert not decontam.is_contaminated(
        "Calculate the eigenvalues of a 3x3 orthogonal rotation matrix.",
        "Select the correct characteristic polynomial.",
    )

    # Model Souping
    ckpt1 = tmp_path / "c1.safetensors"
    ckpt2 = tmp_path / "c2.safetensors"
    out_soup = tmp_path / "soup.safetensors"
    from safetensors.torch import load_file, save_file

    save_file({"w": torch.tensor([1.0, 2.0])}, str(ckpt1))
    save_file({"w": torch.tensor([3.0, 4.0])}, str(ckpt2))
    greedy_model_soup([ckpt1, ckpt2], [0.25, 0.75], out_soup)
    loaded = load_file(str(out_soup))
    assert torch.allclose(loaded["w"], torch.tensor([2.5, 3.5]))

    # In-place teacher logit annotation & teacher ensemble blending
    from scripts.generate_teacher_logits import (
        annotate_jsonl_with_mock_teachers,
        combine_teacher_distributions,
    )
    from scripts.prepare_data import build_smoke_dataset

    smoke_file = tmp_path / "stage2_smoke.jsonl"
    assert build_smoke_dataset(smoke_file, num_samples=8) == 8
    assert annotate_jsonl_with_mock_teachers(smoke_file, smoke_file) == 8
    blended = combine_teacher_distributions(
        {
            "knowledgator/gliclass-v4-edge": [0.7, 0.2, 0.1],
            "MLukauskas/amber-0.4b": [0.6, 0.3, 0.1],
            "Qwen/Qwen3-30B-A3B-Instruct-2507": [0.8, 0.1, 0.1],
        },
        question_type="choice",
    )
    assert len(blended) == 3
    assert abs(sum(blended) - 1.0) < 1e-6


def test_forward_single_request_empty_options_and_long_option_window_budget(
    fast_engine: JEVSetEngine,
) -> None:
    from src.jev_set.modeling_jev_set import forward_single_request

    raw_empty, cal_empty = forward_single_request(
        model=fast_engine.model,
        tokenizer=fast_engine.tokenizer,
        state="Non-empty state",
        instructions="Non-empty instructions",
        options=[],
        question_type="choice",
        device="cpu",
    )
    assert raw_empty.numel() == 0
    assert cal_empty.numel() == 0

    tok = DeterministicWhitespaceTokenizer()
    state_6k = " ".join(f"s_{i}" for i in range(6000))
    inst_1500 = " ".join(f"q_{i}" for i in range(1500))
    opt_1000 = [" ".join(f"o_{i}" for i in range(1000))]
    packs = build_tree_masked_packs(
        tokenizer=tok,
        state=state_6k,
        instructions=inst_1500,
        options=opt_1000,
        max_window_tokens=8192,
        max_state_tokens=6144,
        device="cpu",
    )
    assert len(packs) >= 2
    for win_chunks in packs:
        for pack in win_chunks:
            assert pack.input_ids.shape[1] <= 8192


def test_noul_regime_routing_and_calibrated_checkpoint_engine(tmp_path: Path) -> None:
    from safetensors.torch import load_file, save_file
    from scripts.soup_and_calibrate import inject_calibrated_temperatures_into_checkpoint

    ckpt_dir = tmp_path / "calibrated_ckpt"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = JEVSetConfig(backbone_name_or_path=str(ckpt_dir), num_hidden_layers=2)
    eng = JEVSetEngine(
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
        load_pretrained=False,
        config=cfg,
    )
    ckpt_file = ckpt_dir / "model.safetensors"
    save_file(eng.model.state_dict(), str(ckpt_file))

    log_t = torch.tensor([0.0, 1.25, -0.40], dtype=torch.float32)
    slope_t = torch.tensor([0.10, 0.55, 0.20], dtype=torch.float32)
    inject_calibrated_temperatures_into_checkpoint(ckpt_file, log_t, slope_t)
    saved_state = load_file(str(ckpt_file))
    assert torch.allclose(saved_state["head.log_temperature"], log_t)
    assert torch.allclose(saved_state["head.cardinality_slope"], slope_t)

    # Reload via JEVSetEngine with load_pretrained=True on local directory without tokenizer.json
    reloaded_eng = JEVSetEngine(
        checkpoint_dir=str(ckpt_dir),
        device="cpu",
        load_pretrained=True,
        config=JEVSetConfig(backbone_name_or_path=str(ckpt_dir), num_hidden_layers=2),
    )
    assert torch.allclose(reloaded_eng.model.head.log_temperature.cpu(), log_t)
    assert torch.allclose(reloaded_eng.model.head.cardinality_slope.cpu(), slope_t)

    # Verify noul question routes through regime 1 (not regime 0)
    with torch.no_grad():
        reloaded_eng.model.head.base_scorer.weight.fill_(0.05)
        reloaded_eng.model.head.log_temperature.copy_(torch.tensor([0.0, 2.0, 0.0]))
    q_noul = {"q1": {"type": "noul", "instructions": "Verify binary abstention."}}
    resp_high_temp, _ = reloaded_eng("Context state for noul test.", q_noul)
    with torch.no_grad():
        reloaded_eng.model.head.log_temperature.copy_(torch.tensor([0.0, -2.0, 0.0]))
    resp_low_temp, _ = reloaded_eng("Context state for noul test.", q_noul)
    # Changing regime 1 temperature (while keeping regime 0 fixed at 0.0) changes noul probability
    assert resp_high_temp["answers"]["q1"]["noul"] != resp_low_temp["answers"]["q1"]["noul"]

