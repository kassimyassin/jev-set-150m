# tests/test_phase2_data_and_eval.py
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import pytest
import torch

from scripts.evaluate_official import (
    COEFF_O,
    COEFF_S,
    INTERCEPT_FULL,
    compute_linear_full_val,
    compute_local_full_val_and_bootstrap,
    compute_official_full_score,
)
from decision_index.scoring.index import score_run
from scripts.generate_teacher_logits import (
    TEACHER_ENSEMBLE_WEIGHTS,
    TeacherEnsembleExtractor,
    combine_teacher_distributions,
    extract_teacher_logits_resumable,
)
from scripts.prepare_data import (
    CANONICAL_JSONL_FIELDS,
    DATASET_INVENTORY_16,
    UNSEEN_15_OOD_TASKS,
    ThreeTierDecontaminator,
    build_cluster_stratified_5fold_s_val,
    build_o_val_15task_holdout,
    build_smoke_dataset,
    compute_dataset_sampling_weights,
    convert_upstream_row_to_canonical,
    curate_full_pipeline,
    permute_and_curriculum_subsample_options,
    resolve_dataset_inventory,
    verify_zero_fold_leakage,
)
from scripts.train import run_ablation_experiment, run_training_stage


def test_three_tier_decontaminator_all_tiers_and_audit_report(tmp_path: Path) -> None:
    decontam = ThreeTierDecontaminator(
        jaccard_threshold=0.70,
        cosine_threshold=0.90,
        shared_span_chars=64,
    )

    ref_state = (
        "In a double-blind randomized clinical trial of 480 patients with refractory atrial "
        "fibrillation, catheter cryoablation reduced arrhythmia recurrence by 42 percent."
    )
    ref_inst = "Which primary clinical endpoint was statistically significant at p < 0.001?"
    decontam.add_reference_text("bench_0", ref_state, ref_inst)

    # Set a known unit reference embedding for Tier 3 testing
    ref_emb = torch.zeros((1, 16), dtype=torch.float32)
    ref_emb[0, 0] = 1.0
    decontam.set_reference_embeddings(ref_emb)

    # 1. Tier 1: Exact normalized SHA-256 hash match (different casing/punctuation)
    v1 = decontam.inspect_candidate(ref_state.upper() + "!!!", ref_inst)
    assert v1["contaminated"] is True
    assert v1["tier"] == 1
    assert v1["reason"] == "tier1_exact_sha256"

    # 2. Tier 2a: 13-gram MinHash LSH Jaccard >= 0.70 (short text < 64 chars so span64 won't fire first)
    short_ref = "Solve x^2 + 5x + 6 = 0 over real numbers."
    decontam.add_reference_text("bench_short", short_ref, "Pick roots.")
    v2_minhash = decontam.inspect_candidate(
        "Solve x^2 + 5x + 6 = 0 over real numbers.",
        "Pick root.",
    )
    assert v2_minhash["contaminated"] is True
    assert v2_minhash["tier"] == 2
    assert v2_minhash["reason"] == "tier2_minhash_jaccard"

    # 3. Tier 2b: Exact shared 13-gram contiguous span >= 64 chars embedded in a huge unrelated document
    # Overall Jaccard is well below 0.70 (< 0.15), so only the 64-char shared span check catches it
    unrelated_padding = " ".join(
        f"completely_distinct_Filler_token_{i}_xyz" for i in range(180)
    )
    embedded_state = (
        f"{unrelated_padding} "
        "catheter cryoablation reduced arrhythmia recurrence by 42 percent "
        f"{unrelated_padding}"
    )
    v2_span = decontam.inspect_candidate(embedded_state, "Unrelated question prompt?")
    assert v2_span["contaminated"] is True
    assert v2_span["tier"] == 2
    assert v2_span["reason"] == "tier2_shared_span_64chars"

    # 4. Tier 3: Dense embedding cosine similarity >= 0.90 (lexically distinct text)
    near_emb = torch.zeros((1, 16), dtype=torch.float32)
    near_emb[0, 0] = 0.95
    near_emb[0, 1] = 0.3122
    v3 = decontam.inspect_candidate(
        "Completely novel wording about pulmonary vein isolation outcomes.",
        "Determine the hazard ratio.",
        candidate_embedding=near_emb,
    )
    assert v3["contaminated"] is True
    assert v3["tier"] == 3
    assert v3["reason"] == "tier3_dense_cosine"

    # 5. Clean candidate passes all 3 tiers
    orth_emb = torch.zeros((1, 16), dtype=torch.float32)
    orth_emb[0, 2] = 1.0
    v_clean = decontam.inspect_candidate(
        "Configure a Kubernetes HorizontalPodAutoscaler for stateless gRPC microservices.",
        "Select the valid YAML manifest.",
        candidate_embedding=orth_emb,
    )
    assert v_clean["contaminated"] is False
    assert v_clean["tier"] == 0

    # 6. Batched encoder embeddings fallback works offline
    batched_embs = decontam.encode_texts_batched(
        ["First text sample", "Second text sample"],
        batch_size=1,
        use_pretrained_encoder=False,
    )
    assert batched_embs.shape == (2, 256)
    assert torch.allclose(batched_embs.norm(p=2, dim=-1), torch.ones(2), atol=1e-5)

    # 7. Cryptographic SHA-256 decontamination audit report
    audit_path = tmp_path / "decontamination_audit.json"
    audit = decontam.emit_audit_report(output_path=audit_path)
    assert audit_path.exists()
    loaded_audit = json.loads(audit_path.read_text(encoding="utf-8"))
    assert loaded_audit["statistics"]["total_checked"] == 5
    assert loaded_audit["statistics"]["total_contaminated"] == 4
    assert loaded_audit["statistics"]["clean_retained"] == 1
    assert loaded_audit["statistics"]["tier1_exact_hits"] == 1
    assert loaded_audit["statistics"]["tier2_minhash_hits"] == 1
    assert loaded_audit["statistics"]["tier2_span64_hits"] == 1
    assert loaded_audit["statistics"]["tier3_cosine_hits"] == 1
    assert len(loaded_audit["audit_signature_sha256"]) == 64


def test_16_dataset_inventory_and_commercial_clean_anli_substitution() -> None:
    default_inv = resolve_dataset_inventory(commercial_clean=False)
    clean_inv = resolve_dataset_inventory(commercial_clean=True)

    assert len(default_inv) == 16
    assert len(clean_inv) == 16

    # Group 12 uses facebook/anli by default, and substitutes nyu-mll/multi_nli + stanfordnlp/snli when commercial_clean=True
    g12_default = default_inv[11]
    g12_clean = clean_inv[11]
    assert g12_default.group_index == 12
    assert "facebook/anli" in g12_default.hf_identifiers
    assert "facebook/anli" not in g12_clean.hf_identifiers
    assert "nyu-mll/multi_nli" in g12_clean.hf_identifiers
    assert "stanfordnlp/snli" in g12_clean.hf_identifiers
    assert "Commercial-Clean" in g12_clean.license

    # Verify public 37 benchmark filtering in broad OOD Group 1 (tasksource)
    g1_spec = default_inv[0]
    contaminated_task_row = {
        "task": "mmlu_pro",
        "state": "Some MMLU-Pro test question",
        "options": ["A", "B", "C", "D"],
        "label": 1,
    }
    assert convert_upstream_row_to_canonical(contaminated_task_row, g1_spec, row_idx=0) is None

    clean_task_row = {
        "task": "custom_discourse_coherence",
        "premise": "The satellite entered low Earth orbit.",
        "hypothesis": "The launch vehicle reached orbital velocity.",
        "label": "A",
    }
    canonical = convert_upstream_row_to_canonical(clean_task_row, g1_spec, row_idx=1)
    assert canonical is not None
    for field_name in CANONICAL_JSONL_FIELDS:
        assert field_name in canonical


def test_temperature_scaled_sqrt_sampling_law_and_option_permutation() -> None:
    counts = {
        "ood_large": 100_000,  # Capped at 50,000
        "ood_small": 5_000,
        "same_gold": 45_000,
        "same_std": 45_000,
    }
    is_gold = {"ood_large": False, "ood_small": False, "same_gold": True, "same_std": False}
    is_ood = {"ood_large": True, "ood_small": True, "same_gold": False, "same_std": False}

    # Stage 2: 55% Broad OOD / 45% Same-Skill, alpha=0.55, cap=50,000
    w_s2 = compute_dataset_sampling_weights(
        counts, is_gold, is_ood, alpha=0.55, cap=50_000, gold_multiplier=1.2, stage="stage2"
    )
    ood_share_s2 = w_s2["ood_large"] + w_s2["ood_small"]
    same_share_s2 = w_s2["same_gold"] + w_s2["same_std"]
    assert math.isclose(ood_share_s2, 0.55, rel_tol=1e-6)
    assert math.isclose(same_share_s2, 0.45, rel_tol=1e-6)

    # Verify exact ratio (50000 / 5000) ** 0.55 = 10 ** 0.55 within OOD pool
    expected_ood_ratio = (50_000 / 5_000) ** 0.55
    assert math.isclose(w_s2["ood_large"] / w_s2["ood_small"], expected_ood_ratio, rel_tol=1e-6)

    # Stage 3: 35% Broad OOD / 65% Gold-Weighted Same-Skill (1.2x on Gold benchmarks)
    w_s3 = compute_dataset_sampling_weights(
        counts, is_gold, is_ood, alpha=0.55, cap=50_000, gold_multiplier=1.2, stage="stage3"
    )
    ood_share_s3 = w_s3["ood_large"] + w_s3["ood_small"]
    same_share_s3 = w_s3["same_gold"] + w_s3["same_std"]
    assert math.isclose(ood_share_s3, 0.35, rel_tol=1e-6)
    assert math.isclose(same_share_s3, 0.65, rel_tol=1e-6)
    assert math.isclose(w_s3["same_gold"] / w_s3["same_std"], 1.2, rel_tol=1e-6)

    # Verify dynamic option shuffling + cardinality curriculum subsampling preserves gold alignment
    rng = random.Random(123)
    rec = {
        "id": "rec_20",
        "options": [f"opt_{i}" for i in range(20)],
        "question_type": "choice",
        "gold_indices": [15],
        "teacher_probs": [0.80 if i == 15 else 0.20 / 19 for i in range(20)],
    }
    # Force curriculum subsampling (full_k_prob=0.0) into K' in [8, 12]
    shuf = permute_and_curriculum_subsample_options(
        rec, rng=rng, full_k_prob=0.0, hard_neg_k_range=(8, 12)
    )
    assert 8 <= len(shuf["options"]) <= 12
    gold_new_idx = shuf["gold_indices"][0]
    assert shuf["options"][gold_new_idx] == "opt_15"
    assert shuf["teacher_probs"] is not None
    assert math.isclose(sum(shuf["teacher_probs"]), 1.0, rel_tol=1e-6)
    assert shuf["teacher_probs"][gold_new_idx] == max(shuf["teacher_probs"])


def test_5fold_cluster_stratified_split_zero_group_and_doc_leakage() -> None:
    # Construct records with transitive links:
    # Record 0 and Record 1 share group_id="grp_chain"
    # Record 1 and Record 2 have DIFFERENT group_ids ("grp_chain" vs "grp_other")
    # but share the exact same normalized `state` document text!
    # Union-Find must place Records 0, 1, and 2 into the exact same fold.
    raw_records: list[dict[str, Any]] = [
        {
            "id": "r0",
            "area": "knowledge",
            "group_id": "grp_chain",
            "state": "Alpha document unique text.",
            "instructions": "Q0",
            "options": ["a", "b"],
            "gold_indices": [0],
        },
        {
            "id": "r1",
            "area": "knowledge",
            "group_id": "grp_chain",
            "state": "Shared transitive document body across different group IDs!",
            "instructions": "Q1",
            "options": ["a", "b"],
            "gold_indices": [1],
        },
        {
            "id": "r2",
            "area": "knowledge",
            "group_id": "grp_other_linked_by_doc",
            "state": "  shared TRANSITIVE document body across different group ids!!! ",
            "instructions": "Q2",
            "options": ["a", "b"],
            "gold_indices": [0],
        },
    ]
    for i in range(3, 50):
        raw_records.append(
            {
                "id": f"r{i}",
                "area": ["knowledge", "language", "retrieval", "tools", "taste"][i % 5],
                "group_id": f"grp_{i // 2:03d}",
                "state": f"Distinct state document for cluster {i // 2}.",
                "instructions": f"Instruction {i}",
                "options": ["x", "y", "z"],
                "gold_indices": [i % 3],
            }
        )

    folded = build_cluster_stratified_5fold_s_val(raw_records, num_folds=5, seed=42)
    assert len(folded) == len(raw_records)

    # Verify transitive cluster (r0, r1, r2) landed in the exact same fold
    assert folded[0]["fold"] == folded[1]["fold"] == folded[2]["fold"]

    leakage_report = verify_zero_fold_leakage(folded)
    assert leakage_report["num_folds"] == 5
    assert leakage_report["group_id_leakage_count"] == 0
    assert leakage_report["doc_hash_leakage_count"] == 0
    assert leakage_report["zero_leakage_verified"] is True


def test_15_task_unseen_o_val_holdout_coverage() -> None:
    o_val = build_o_val_15task_holdout(samples_per_task=4, seed=2026)
    assert len(o_val) == 15 * 4
    tasks_seen = {r["dataset"] for r in o_val}
    areas_seen = {r["area"] for r in o_val}
    qtypes_seen = {r["question_type"] for r in o_val}

    assert len(tasks_seen) == 15
    assert areas_seen == {"knowledge", "language", "retrieval", "tools", "taste"}
    assert qtypes_seen == {"choice", "multi_label", "ranking", "Preference", "Match"}


def test_10000_iteration_paired_bootstrap_evaluator_and_promotion_gates() -> None:
    assert callable(score_run)

    # 1. Verify linear surrogate formula: Full_val = 0.6069 * S_val + 0.4940 * O_val + 0.4269
    s_val, o_val = 31.20, 9.40
    expected_linear = 0.6069 * s_val + 0.4940 * o_val + 0.4269
    assert math.isclose(compute_linear_full_val(s_val, o_val), expected_linear, rel_tol=1e-7)
    assert abs(compute_official_full_score(s_val, o_val) - expected_linear) < 0.005

    # 2. Linked-case (group_id) aggregation + positive significant improvement -> promote_checkpoint=True
    s_cases = [
        {"group_id": "g0", "is_gold_benchmark": True, "candidate_score": 100.0, "baseline_score": 100.0},
        {"group_id": "g0", "is_gold_benchmark": True, "candidate_score": 100.0, "baseline_score": 0.0},
    ] + [
        {
            "group_id": f"g_{i}",
            "is_gold_benchmark": (i % 2 == 0),
            "candidate_score": 32.0 + (i % 3),
            "baseline_score": 26.0 + (i % 3),
        }
        for i in range(1, 30)
    ]
    o_cases = [
        {
            "group_id": f"og_{j}",
            "is_gold_benchmark": False,
            "candidate_score": 9.5 + (j % 2),
            "baseline_score": 7.2 + (j % 2),
        }
        for j in range(20)
    ]

    res_promote = compute_local_full_val_and_bootstrap(
        s_val_cases=s_cases,
        o_val_cases=o_cases,
        n_bootstrap=10_000,
        seed=42,
    )
    assert res_promote["n_bootstrap"] == 10_000
    assert res_promote["num_s_cases"] == 30
    assert res_promote["num_o_cases"] == 20
    assert res_promote["delta_full_val"] > 0.0
    assert res_promote["delta_o_val"] >= -0.15
    assert res_promote["p_value"] < 0.05
    assert res_promote["promote_checkpoint"] is True

    # 3. O_val drift gate: if S_val improves strongly (+5.0) so Delta Full_val > 0,
    #    but O_val drops by -0.50 (< -0.15 limit), promote_checkpoint MUST be False!
    res_o_drift = compute_local_full_val_and_bootstrap(
        s_val_cases=[35.0] * 25,
        o_val_cases=[6.70] * 25,
        s_val_baseline=[30.0] * 25,
        o_val_baseline=[7.20] * 25,
        n_bootstrap=10_000,
        seed=42,
    )
    assert res_o_drift["delta_full_val"] > 0.0
    assert res_o_drift["delta_o_val"] < -0.15
    assert res_o_drift["promote_checkpoint"] is False


def test_resumable_multi_teacher_logit_extraction_and_ablation_runner(tmp_path: Path) -> None:
    input_file = tmp_path / "stage2_input.jsonl"
    output_file = tmp_path / "stage2_output.jsonl"
    assert build_smoke_dataset(input_file, num_samples=10, seed=99) == 10

    call_counts = {m: 0 for m in TEACHER_ENSEMBLE_WEIGHTS}

    def make_counting_scorer(model_name: str):
        def _scorer(batch_rows: list[dict[str, Any]]) -> list[list[float]]:
            call_counts[model_name] += len(batch_rows)
            out: list[list[float]] = []
            for r in batch_rows:
                K = len(r["options"])
                if r.get("question_type") == "multi_label":
                    out.append([0.85 if k == 0 else 0.10 for k in range(K)])
                else:
                    out.append([0.70 if k == 0 else 0.30 / max(1, K - 1) for k in range(K)])
            return out

        return _scorer

    custom_scorers = {m: make_counting_scorer(m) for m in TEACHER_ENSEMBLE_WEIGHTS}
    extractor = TeacherEnsembleExtractor(custom_scorers=custom_scorers)

    # Pass 1: Simulate Colab Pro interruption after processing 4 rows
    part1 = extract_teacher_logits_resumable(
        input_path=input_file,
        output_path=output_file,
        extractor=extractor,
        batch_size=2,
        resume=True,
        max_new_rows=4,
    )
    assert part1["resumed_rows"] == 0
    assert part1["newly_processed_rows"] == 4
    assert part1["completed_rows"] == 4
    assert part1["complete"] is False
    for m in TEACHER_ENSEMBLE_WEIGHTS:
        assert call_counts[m] == 4

    # Pass 2: Resume from checkpoint -> must skip the first 4 rows and only score the remaining 6 rows
    part2 = extract_teacher_logits_resumable(
        input_path=input_file,
        output_path=output_file,
        extractor=extractor,
        batch_size=3,
        resume=True,
    )
    assert part2["resumed_rows"] == 4
    assert part2["newly_processed_rows"] == 6
    assert part2["completed_rows"] == 10
    assert part2["complete"] is True
    for m in TEACHER_ENSEMBLE_WEIGHTS:
        assert call_counts[m] == 10  # Zero duplicate evaluations on the first 4 rows!

    # Verify resumable training stage and Experiment A-H ablation runner
    ckpt_dir = tmp_path / "stage1_resume_test"
    t_part1 = run_training_stage(
        config_path=Path("configs/stage1_ood.yaml"),
        max_steps=2,
        load_pretrained=False,
        num_layers=2,
        resume=True,
        output_dir_override=ckpt_dir,
    )
    assert t_part1["steps"] == 2
    assert t_part1["resumed_from_checkpoint"] is False

    t_part2 = run_training_stage(
        config_path=Path("configs/stage1_ood.yaml"),
        max_steps=4,
        load_pretrained=False,
        num_layers=2,
        resume=True,
        output_dir_override=ckpt_dir,
    )
    assert t_part2["steps"] == 4
    assert t_part2["resumed_from_checkpoint"] is True

    ablations = run_ablation_experiment("all")
    assert set(ablations.keys()) == {
        "exp_a",
        "exp_b",
        "exp_c",
        "exp_d",
        "exp_e",
        "exp_f",
        "exp_g",
        "exp_h",
    }
    assert all(v["gate_passed"] is True for v in ablations.values())


def test_curate_full_pipeline_end_to_end(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets as hf_datasets
    from scripts.prepare_data import stream_hf_dataset_group

    # Verify live HF datasets.load_dataset streaming adapter path via deterministic mock stream
    def fake_load_dataset(hf_id: str, split: str = "train", streaming: bool = True):
        return [
            {
                "id": f"{hf_id}_0",
                "question": f"Sample question from {hf_id}",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "answer_index": 2,
            }
            for _ in range(4)
        ]

    monkeypatch.setattr(hf_datasets, "load_dataset", fake_load_dataset)
    inv = resolve_dataset_inventory(commercial_clean=True)
    streamed = stream_hf_dataset_group(inv[6], max_rows=3, offline=False)
    assert len(streamed) == 3
    assert streamed[0]["gold_indices"] == [2]

    summary = curate_full_pipeline(
        output_dir=tmp_path / "processed",
        commercial_clean=True,
        rows_per_group=4,
        allow_synthetic_fallback=True,
        offline=False,
        seed=42,
    )
    assert summary["commercial_clean"] is True
    assert summary["dataset_groups_curated"] == 16
    assert summary["s_val_zero_leakage"] is True
    assert summary["o_val_unseen_tasks"] == 15
    assert (tmp_path / "processed" / "decontamination_audit.json").exists()
    assert (tmp_path / "processed" / "s_val_5fold.jsonl").exists()
    assert (tmp_path / "processed" / "o_val_holdout15.jsonl").exists()


def test_decontaminator_auto_tier3_and_strided_span_boundary() -> None:
    # 1. Verify strided anchor indexing uses ~16x fewer hashes than 1-char sliding windows
    #    and enforces exact 64-char boundary (63-char shared span -> False, 64-char shared span -> True)
    decontam = ThreeTierDecontaminator(jaccard_threshold=0.95, cosine_threshold=0.99, shared_span_chars=64)
    shared_64 = "a1b2c3d4e5f6g7h8" * 4  # Exactly 64 alphanumeric chars
    assert len(shared_64) == 64
    ref_pad = " ".join(f"refword_{i}" for i in range(40))
    cand_pad = " ".join(f"candword_{i}" for i in range(40))
    decontam.add_reference_text("ref_span", f"{ref_pad} p{shared_64}q {ref_pad}", "Question?")

    norm_ref_len = len(decontam._normalized_ref_texts[0])
    assert len(decontam._span64_hashes) <= (norm_ref_len // 16) + 2

    # Strictly 63-char shared substring (bounded by r...s vs p...q) should NOT trigger 64-char shared span
    v_63 = decontam.inspect_candidate(f"{cand_pad} r{shared_64[:63]}s {cand_pad}", "Other?")
    assert v_63["contaminated"] is False

    # Full 64-char shared substring MUST trigger tier2_shared_span_64chars regardless of offset % 16
    for prefix_offset in (0, 3, 7, 13, 15):
        pad_prefix = "x" * prefix_offset + " " + cand_pad
        v_64 = decontam.inspect_candidate(f"{pad_prefix} r{shared_64}s {cand_pad}", "Other?")
        assert v_64["contaminated"] is True
        assert v_64["reason"] == "tier2_shared_span_64chars"

    # 2. Verify Tier 3 fires automatically via is_contaminated(state, instructions) WITHOUT passing candidate_embedding
    #    when 13-gram Jaccard < 0.70 and shared span < 64 chars, but dense embedding cosine >= threshold
    decontam_t3 = ThreeTierDecontaminator(jaccard_threshold=0.85, cosine_threshold=0.75, shared_span_chars=64)
    # Reordered 4-word clauses break 13-gram Jaccard (< 0.35) and have < 64 char spans,
    # while sharing identical word unigrams and char 4-grams so dense cosine > 0.90!
    ref_clauses = [
        "quantum chromodynamics lattice gauge",
        "asymptotic freedom quark confinement",
        "renormalization group beta function",
        "chiral symmetry breaking instantons",
    ]
    cand_clauses = list(reversed(ref_clauses))
    decontam_t3.add_reference_text("qcd_0", " ".join(ref_clauses), "compute coupling")
    verdict_t3 = decontam_t3.inspect_candidate(" ".join(cand_clauses), "compute coupling")
    assert verdict_t3["contaminated"] is True
    assert verdict_t3["tier"] == 3
    assert verdict_t3["reason"] == "tier3_dense_cosine"


def test_in_place_and_truncated_jsonl_teacher_resumption_and_native_architectures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # 1. Test in_place=True (input_path == output_path) interrupted mid-file and resumed
    inplace_file = tmp_path / "inplace_stage2.jsonl"
    assert build_smoke_dataset(inplace_file, num_samples=6, seed=77) == 6
    sidecar_ckpt = inplace_file.with_suffix(".teacher_ckpt.jsonl")

    extractor = TeacherEnsembleExtractor(mock_teachers=True)
    p1 = extract_teacher_logits_resumable(
        input_path=inplace_file,
        output_path=inplace_file,
        extractor=extractor,
        batch_size=2,
        resume=True,
        max_new_rows=2,
    )
    assert p1["completed_rows"] == 2
    assert p1["complete"] is False
    assert sidecar_ckpt.exists()

    p2 = extract_teacher_logits_resumable(
        input_path=inplace_file,
        output_path=inplace_file,
        extractor=extractor,
        batch_size=2,
        resume=True,
    )
    assert p2["resumed_rows"] == 2
    assert p2["newly_processed_rows"] == 4
    assert p2["completed_rows"] == 6
    assert p2["complete"] is True
    assert not sidecar_ckpt.exists()

    # 2. Test crash recovery when output_path has a truncated/corrupted trailing JSONL line
    in_f = tmp_path / "crash_in.jsonl"
    out_f = tmp_path / "crash_out.jsonl"
    assert build_smoke_dataset(in_f, num_samples=5, seed=88) == 5
    extract_teacher_logits_resumable(
        input_path=in_f,
        output_path=out_f,
        extractor=extractor,
        batch_size=2,
        resume=True,
        max_new_rows=2,
    )
    # Simulate mid-write Colab disconnect leaving a broken tail line without newline
    with out_f.open("a", encoding="utf-8") as f:
        f.write('{"id": "smoke_00002", "state": "truncated_crash_line')

    recovered = extract_teacher_logits_resumable(
        input_path=in_f,
        output_path=out_f,
        extractor=extractor,
        batch_size=2,
        resume=True,
    )
    assert recovered["resumed_rows"] == 2
    assert recovered["newly_processed_rows"] == 3
    assert recovered["completed_rows"] == 5
    assert recovered["complete"] is True
    lines = [json.loads(line) for line in out_f.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert [r["id"] for r in lines] == [f"smoke_{i:05d}" for i in range(5)]

    # 3. Test native CausalLM (Qwen3) and 2-logit SequenceClassification (Amber) scoring paths
    import transformers

    class DummyTokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
            return [10, 11] if "Paris" in text else [20, 21]

        def __call__(self, texts_a, texts_b=None, **kwargs):
            class _Batch(dict):
                def to(self, _dev):
                    return self

            # Encode whether each candidate option is "Paris"
            flags = [1.0 if "Paris" in b else 0.0 for b in (texts_b or texts_a)]
            return _Batch({"is_paris": torch.tensor(flags)})

    class DummyCausalLM:
        def to(self, _dev):
            return self

        def eval(self):
            return self

        def __call__(self, input_ids: torch.Tensor):
            # Favor token 10 & 11 ("Paris") over 20 & 21
            B, L = input_ids.shape
            logits = torch.zeros((B, L, 32), dtype=torch.float32)
            logits[:, :, 10] = 5.0
            logits[:, :, 11] = 5.0
            logits[:, :, 20] = -2.0
            logits[:, :, 21] = -2.0
            return type("Out", (), {"logits": logits})()

    class DummySeqCls2Logit:
        def to(self, _dev):
            return self

        def eval(self):
            return self

        def __call__(self, **enc):
            flags = enc["is_paris"]
            # 2-column binary logits: col 0 is negative/contradiction, col 1 is positive/entailment
            neg_col = torch.where(flags > 0.5, torch.tensor(-3.0), torch.tensor(3.0))
            pos_col = torch.where(flags > 0.5, torch.tensor(3.0), torch.tensor(-3.0))
            logits = torch.stack([neg_col, pos_col], dim=-1)
            return type("Out", (), {"logits": logits})()

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: DummyTokenizer())
    monkeypatch.setattr(transformers.AutoModelForCausalLM, "from_pretrained", lambda *a, **kw: DummyCausalLM())
    monkeypatch.setattr(
        transformers.AutoModelForSequenceClassification, "from_pretrained", lambda *a, **kw: DummySeqCls2Logit()
    )

    real_ext = TeacherEnsembleExtractor(
        weights={
            "MLukauskas/amber-0.4b": 0.50,
            "Qwen/Qwen3-30B-A3B-Instruct-2507": 0.50,
        },
        device="cpu",
        mock_teachers=False,
    )
    scored = real_ext.score_batch(
        [
            {
                "id": "cap_0",
                "state": "France is in Western Europe.",
                "instructions": "Select the capital of France.",
                "options": ["Berlin", "Paris", "Madrid"],
                "question_type": "choice",
                "gold_indices": [1],
            }
        ]
    )
    probs = scored[0]["teacher_probs"]
    assert len(probs) == 3
    # Both Qwen CausalLM and Amber 2-logit classifier must assign highest probability to index 1 ("Paris")!
    assert probs[1] > probs[0] and probs[1] > probs[2]


def test_edge_cases_fold_splitter_none_group_id_and_chance_adjusted_eval(tmp_path: Path) -> None:
    from scripts.evaluate_official import evaluate_engine_on_jsonl
    from scripts.prepare_data import stream_hf_dataset_group
    from src.jev_set.engine import JEVSetEngine
    from src.jev_set.modeling_jev_set import JEVSetConfig

    # 1. Verify stream_hf_dataset_group synthetic fallback assigns consecutive pairs to shared group_ids
    spec = resolve_dataset_inventory(commercial_clean=True)[7]
    synth_rows = stream_hf_dataset_group(spec, max_rows=4, offline=True, seed=42)
    assert len(synth_rows) == 4
    assert synth_rows[0]["group_id"] == synth_rows[1]["group_id"]
    assert synth_rows[2]["group_id"] == synth_rows[3]["group_id"]
    assert synth_rows[0]["group_id"] != synth_rows[2]["group_id"]

    # 2. Verify records with group_id=None or "" and distinct states are NOT merged into a single cluster
    ungrouped = [
        {"id": f"u{i}", "area": "knowledge", "group_id": None if i % 2 == 0 else "", "state": f"Unique doc {i}"}
        for i in range(10)
    ]
    folded_u = build_cluster_stratified_5fold_s_val(ungrouped, num_folds=5, seed=42)
    leak_u = verify_zero_fold_leakage(folded_u)
    assert leak_u["num_folds"] == 5
    assert leak_u["zero_leakage_verified"] is True

    # 3. Verify len(records) < num_folds and single connected component edge cases
    single_comp = [
        {"id": f"sc{i}", "area": "tools", "group_id": "shared_grp", "state": "Same doc"}
        for i in range(3)
    ]
    folded_sc = build_cluster_stratified_5fold_s_val(single_comp, num_folds=5, seed=42)
    assert len({r["fold"] for r in folded_sc}) == 1
    assert verify_zero_fold_leakage(folded_sc)["zero_leakage_verified"] is True

    # 4. Verify evaluate_engine_on_jsonl computes unbiased chance-adjusted scores and ranking nDCG@10
    eval_jsonl = tmp_path / "eval_ranking_and_choice.jsonl"
    eval_rows = [
        {
            "id": "rank_0",
            "group_id": None,
            "area": "retrieval",
            "is_gold_benchmark": True,
            "state": "Query document for retrieval ranking.",
            "instructions": "Rank the passages.",
            "options": ["Passage A", "Passage B", "Passage C", "Passage D"],
            "question_type": "ranking",
            "select_k": 1,
            "gold_indices": [1],
        },
        {
            "id": "choice_0",
            "group_id": "",
            "area": "knowledge",
            "is_gold_benchmark": False,
            "state": "Binary question state.",
            "instructions": "Pick one.",
            "options": ["Opt 0", "Opt 1"],
            "question_type": "choice",
            "select_k": 1,
            "gold_indices": [0],
        },
    ]
    eval_jsonl.write_text("\n".join(json.dumps(r) for r in eval_rows) + "\n", encoding="utf-8")
    eng = JEVSetEngine(load_pretrained=False, device="cpu", config=JEVSetConfig(num_hidden_layers=2))
    scored_cases = evaluate_engine_on_jsonl(eng, eval_jsonl)
    assert len(scored_cases) == 2
    assert scored_cases[0]["group_id"] == "rank_0"
    assert scored_cases[1]["group_id"] == "choice_0"
    # Verify a 50/50 random guesser on binary tasks (+100 on hit, -100 on miss) averages 0.0 after chance correction
    boot_random = compute_local_full_val_and_bootstrap(
        s_val_cases=[
            {"id": "c1", "group_id": None, "candidate_score": 100.0, "baseline_score": 0.0},
            {"id": "c2", "group_id": "", "candidate_score": -100.0, "baseline_score": 0.0},
        ],
        o_val_cases=[0.0, 0.0],
        n_bootstrap=500,
    )
    assert boot_random["s_val"] == 0.0
    assert boot_random["num_s_cases"] == 2

