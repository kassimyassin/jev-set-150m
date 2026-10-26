# scripts/prepare_data.py
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Optional

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.jev_set.data_curation import (
    CANONICAL_JSONL_FIELDS,
    DATASET_INVENTORY_16,
    PUBLIC_37_BENCHMARK_NAMES,
    UNSEEN_15_OOD_TASKS,
    DatasetGroupSpec,
    ThreeTierDecontaminator,
    build_cluster_stratified_5fold_s_val,
    build_o_val_15task_holdout,
    char_ngrams,
    compute_dataset_sampling_weights,
    compute_deterministic_text_embeddings,
    convert_upstream_row_to_canonical,
    extract_char_spans,
    make_minhash,
    normalize_text,
    permute_and_curriculum_subsample_options,
    resolve_dataset_inventory,
    stream_hf_dataset_group,
    verify_zero_fold_leakage,
)


def write_jsonl(records: list[dict[str, Any]], output_path: Path) -> int:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for row in records:
            f.write(json.dumps(row) + "\n")
    return len(records)


def sample_records_by_stage_law(
    records_by_dataset: dict[str, list[dict[str, Any]]],
    is_gold: dict[str, bool],
    is_ood: dict[str, bool],
    stage: str,
    total_samples: int,
    alpha: float = 0.55,
    cap: int = 50_000,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Samples `total_samples` records across the 16 dataset groups according to the
    temperature-scaled square-root law (alpha=0.55) and stage-specific OOD / Same-Skill mixture ratio.
    """
    counts = {d: len(rows) for d, rows in records_by_dataset.items() if len(rows) > 0}
    if not counts or total_samples <= 0:
        return []

    weights = compute_dataset_sampling_weights(
        dataset_counts=counts,
        is_gold=is_gold,
        is_ood=is_ood,
        alpha=alpha,
        cap=cap,
        stage=stage,
    )
    rng = random.Random(seed)
    dataset_names = list(weights.keys())
    dataset_probs = [weights[d] for d in dataset_names]

    sampled: list[dict[str, Any]] = []
    cursors = {d: 0 for d in dataset_names}
    for _ in range(total_samples):
        chosen_d = rng.choices(dataset_names, weights=dataset_probs, k=1)[0]
        pool = records_by_dataset[chosen_d]
        raw_cursor = cursors[chosen_d]
        idx = raw_cursor % len(pool)
        epoch_pass = raw_cursor // len(pool)
        cursors[chosen_d] += 1
        rec = permute_and_curriculum_subsample_options(pool[idx], rng=rng)
        if epoch_pass > 0:
            base_id = str(rec.get("id", f"{chosen_d}_{idx:05d}"))
            rec["id"] = f"{base_id}__pass{epoch_pass}"
        sampled.append(rec)

    if stage == "stage3":
        # Sort by group_id so linked cases are adjacent in Stage 3 mini-batches
        sampled.sort(key=lambda r: (str(r.get("group_id", "")), str(r.get("id", ""))))

    return sampled


def build_smoke_dataset(output_path: Path, num_samples: int = 64, seed: int = 42) -> int:
    """
    Generates a deterministic canonical 15-field JSONL dataset for local pipeline verification
    and Colab Pro smoke tests. When `output_path.name == 's_val_5fold.jsonl'`, assigns 5-fold
    cluster-stratified fold labels with zero `group_id` or `doc_hash` leakage.
    """
    rng = random.Random(seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    question_types = ["choice", "multi_label", "ranking", "Preference", "Match"]
    areas = ["knowledge", "language", "retrieval", "tools", "taste"]

    if output_path.name == "o_val_holdout15.jsonl":
        per_task = max(1, (num_samples + len(UNSEEN_15_OOD_TASKS) - 1) // len(UNSEEN_15_OOD_TASKS))
        o_rows = build_o_val_15task_holdout(samples_per_task=per_task, seed=seed)[:num_samples]
        return write_jsonl(o_rows, output_path)

    rows: list[dict[str, Any]] = []
    for i in range(num_samples):
        qtype = question_types[i % len(question_types)]
        area = areas[i % len(areas)]
        K = rng.choice([2, 4, 8, 10])
        options = [f"Candidate option {k} for task {i}" for k in range(K)]
        if qtype == "multi_label":
            gold_indices = sorted(rng.sample(range(K), k=min(2, K)))
        else:
            gold_indices = [rng.randrange(K)]
        gold_set = set(gold_indices)
        gold_ranking = list(gold_indices) + [k for k in range(K) if k not in gold_set]
        grp_idx = i // 2
        rows.append(
            {
                "id": f"smoke_{i:05d}",
                "dataset": f"synthetic_{area}",
                "area": area,
                "is_gold_benchmark": bool(i % 2 == 0),
                "is_ood_general": bool(i % 3 != 0),
                "group_id": f"grp_{grp_idx:04d}",
                "state": f"System context state for decision problem group #{grp_idx} in domain {area}.",
                "instructions": f"Select the optimal {qtype} outcome among the {K} options.",
                "options": options,
                "question_type": qtype,
                "select_k": len(gold_indices) if qtype == "multi_label" else 1,
                "gold_indices": gold_indices,
                "gold_ranking": gold_ranking,
                "teacher_probs": None,
                "difficulty_weight": 1.2 if (i % 2 == 0) else 1.0,
            }
        )

    if output_path.name == "s_val_5fold.jsonl":
        rows = build_cluster_stratified_5fold_s_val(rows, num_folds=5, seed=seed)

    return write_jsonl(rows, output_path)


def curate_full_pipeline(
    output_dir: Path,
    suite_jsonl: Optional[Path] = None,
    commercial_clean: bool = False,
    rows_per_group: int = 16,
    allow_synthetic_fallback: bool = True,
    offline: bool = False,
    seed: int = 42,
    min_real_groups: int = 0,
    min_real_fraction: float = 0.0,
) -> dict[str, Any]:
    """
    Executes the complete Phase 2 data curation, 3-tier decontamination, alpha=0.55 mixture
    sampling, 5-fold cluster-stratified S_val splitting, 15-task O_val holdout construction,
    and cryptographic SHA-256 decontamination audit log generation.

    Real-data mode (`allow_synthetic_fallback=False`): groups that cannot be loaded from Hugging Face
    are skipped (error recorded, never replaced by synthetic rows). The run then FAILS LOUDLY
    (`RuntimeError`) if fewer than `min_real_groups` groups produced real rows or the real-row
    fraction is below `min_real_fraction`. The summary always includes per-group real/synthetic counts.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    decontam = ThreeTierDecontaminator()
    if suite_jsonl is not None and suite_jsonl.exists():
        decontam.index_benchmark_requests(suite_jsonl, compute_embeddings=True)
    else:
        # Index representative benchmark anchor prompts so audit log records all 3 tiers
        decontam.add_reference_text(
            "anchor_gpqa_0",
            "Two quantum states with energies E1 and E2 have lifetimes of 10^-9 sec and 10^-8 sec.",
            "Which option best describes the energy uncertainty difference?",
        )

    inventory = resolve_dataset_inventory(commercial_clean=commercial_clean)
    records_by_dataset: dict[str, list[dict[str, Any]]] = {}
    is_gold_map: dict[str, bool] = {}
    is_ood_map: dict[str, bool] = {}
    same_skill_pool: list[dict[str, Any]] = []
    per_group_stats: dict[str, dict[str, Any]] = {}

    for spec in inventory:
        gstats: dict[str, Any] = {}
        try:
            group_rows = stream_hf_dataset_group(
                group_spec=spec,
                max_rows=rows_per_group,
                decontaminator=decontam,
                allow_synthetic_fallback=allow_synthetic_fallback,
                offline=offline,
                seed=seed,
                stats=gstats,
            )
        except Exception:
            # Only reachable with allow_synthetic_fallback=False: record and continue (no substitution).
            group_rows = []
        gstats.setdefault("real_rows", len(group_rows))
        gstats.setdefault("synthetic_rows", 0)
        gstats.setdefault("real_error", None)
        gstats.setdefault("contaminated_dropped", 0)
        per_group_stats[spec.group_name] = gstats
        print(
            f"[data] {spec.group_name:<32} real={gstats['real_rows']:>5} synthetic={gstats['synthetic_rows']:>5}"
            + (f"  ERROR={gstats['real_error']}" if gstats["real_error"] else ""),
            flush=True,
        )
        records_by_dataset[spec.group_name] = group_rows
        is_gold_map[spec.group_name] = spec.is_gold_benchmark
        is_ood_map[spec.group_name] = spec.is_ood_general
        if not spec.is_ood_general:
            same_skill_pool.extend(group_rows)

    total_real = sum(s["real_rows"] for s in per_group_stats.values())
    total_synth = sum(s["synthetic_rows"] for s in per_group_stats.values())
    total_raw = total_real + total_synth
    groups_with_real = sorted(g for g, s in per_group_stats.items() if s["real_rows"] > 0)
    groups_without_real = sorted(g for g, s in per_group_stats.items() if s["real_rows"] == 0)
    real_fraction = (total_real / total_raw) if total_raw else 0.0

    if not allow_synthetic_fallback and total_synth != 0:  # pragma: no cover - invariant
        raise RuntimeError(f"Synthetic rows present ({total_synth}) although synthetic fallback is disabled.")
    if len(groups_with_real) < min_real_groups or real_fraction < min_real_fraction or (
        not allow_synthetic_fallback and total_real == 0
    ):
        raise RuntimeError(
            "REAL-DATA ASSERTION FAILED: "
            f"{len(groups_with_real)}/{len(inventory)} groups have real rows (need >= {min_real_groups}); "
            f"real fraction {real_fraction:.2%} (need >= {min_real_fraction:.2%}); "
            f"groups without real data: {groups_without_real}. "
            "Check network / HF token / gated-dataset access (huggingface-cli login)."
        )

    if total_raw == 0:
        raise RuntimeError("No rows curated at all (all groups empty).")
    stage1_rows = sample_records_by_stage_law(
        records_by_dataset, is_gold_map, is_ood_map, stage="stage1", total_samples=max(32, total_raw), seed=seed
    )
    stage2_rows = sample_records_by_stage_law(
        records_by_dataset, is_gold_map, is_ood_map, stage="stage2", total_samples=max(32, total_raw), seed=seed + 1
    )
    stage3_rows = sample_records_by_stage_law(
        records_by_dataset, is_gold_map, is_ood_map, stage="stage3", total_samples=max(32, total_raw), seed=seed + 2
    )
    s_val_rows = build_cluster_stratified_5fold_s_val(same_skill_pool, num_folds=5, seed=seed)
    leakage_check = verify_zero_fold_leakage(s_val_rows)
    o_val_rows = build_o_val_15task_holdout(samples_per_task=max(2, rows_per_group // 4), seed=seed)

    write_jsonl(stage1_rows, output_dir / "stage1_ood_train.jsonl")
    write_jsonl(stage2_rows, output_dir / "stage2_distill_train.jsonl")
    write_jsonl(stage3_rows, output_dir / "stage3_equilibrium_train.jsonl")
    write_jsonl(s_val_rows, output_dir / "s_val_5fold.jsonl")
    write_jsonl(o_val_rows, output_dir / "o_val_holdout15.jsonl")

    audit_path = output_dir / "decontamination_audit.json"
    audit_report = decontam.emit_audit_report(output_path=audit_path)

    summary = {
        "commercial_clean": commercial_clean,
        "dataset_groups_curated": len(inventory),
        "real_rows_total": total_real,
        "synthetic_rows_total": total_synth,
        "real_fraction": round(real_fraction, 4),
        "groups_with_real_data": len(groups_with_real),
        "groups_without_real_data": groups_without_real,
        "per_group": per_group_stats,
        "stage1_count": len(stage1_rows),
        "stage2_count": len(stage2_rows),
        "stage3_count": len(stage3_rows),
        "s_val_count": len(s_val_rows),
        "s_val_zero_leakage": leakage_check["zero_leakage_verified"],
        "o_val_count": len(o_val_rows),
        "o_val_unseen_tasks": len(UNSEEN_15_OOD_TASKS),
        # O_val is ALWAYS template-generated placeholder rows (no real unseen-task corpus is wired in).
        "o_val_is_synthetic_placeholder": True,
        "audit_report_path": str(audit_path),
        "audit_signature_sha256": audit_report["audit_signature_sha256"],
    }
    (output_dir / "data_build_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary



def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare and decontaminate JEV-Set-150M training data.")
    parser.add_argument("--suite-jsonl", type=Path, help="Path to public suite rows JSONL for 3-tier decontamination.")
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--smoke-test", action="store_true", help="Generate synthetic canonical JSONL smoke datasets.")
    parser.add_argument("--curate-all", action="store_true", help="Curate all 16 dataset groups + S_val + O_val + audit.")
    parser.add_argument("--commercial-clean", action="store_true", help="Substitute CC-BY-NC ANLI with MultiNLI + SNLI.")
    parser.add_argument("--rows-per-group", type=int, default=16)
    parser.add_argument("--num-smoke", type=int, default=64)
    parser.add_argument("--no-synthetic-fallback", action="store_true",
                        help="Real-data mode: never substitute synthetic rows for unloadable groups.")
    parser.add_argument("--offline", action="store_true", help="Skip Hugging Face entirely (synthetic only).")
    parser.add_argument("--min-real-groups", type=int, default=0,
                        help="Fail loudly if fewer groups than this yield real rows.")
    parser.add_argument("--min-real-fraction", type=float, default=0.0,
                        help="Fail loudly if real/(real+synthetic) rows is below this fraction.")
    args = parser.parse_args()

    if args.curate_all:
        summary = curate_full_pipeline(
            output_dir=args.output_dir,
            suite_jsonl=args.suite_jsonl,
            commercial_clean=args.commercial_clean,
            rows_per_group=args.rows_per_group,
            allow_synthetic_fallback=not args.no_synthetic_fallback,
            offline=args.offline,
            min_real_groups=args.min_real_groups,
            min_real_fraction=args.min_real_fraction,
        )
        print(json.dumps(summary, indent=2))
        return

    if args.smoke_test:
        decontam = ThreeTierDecontaminator()
        if args.suite_jsonl and args.suite_jsonl.exists():
            decontam.index_benchmark_requests(args.suite_jsonl)
        for name in (
            "stage1_ood_train.jsonl",
            "stage2_distill_train.jsonl",
            "stage3_equilibrium_train.jsonl",
            "s_val_5fold.jsonl",
            "o_val_holdout15.jsonl",
        ):
            out_file = args.output_dir / name
            n = build_smoke_dataset(out_file, num_samples=args.num_smoke)
            print(f"Wrote {n} canonical records to {out_file}")
        decontam.emit_audit_report(output_path=args.output_dir / "decontamination_audit.json")
        return

    if args.suite_jsonl and args.suite_jsonl.exists():
        decontam = ThreeTierDecontaminator()
        indexed = decontam.index_benchmark_requests(args.suite_jsonl, compute_embeddings=True)
        audit = decontam.emit_audit_report(output_path=args.output_dir / "decontamination_audit.json")
        print(
            f"Indexed {indexed:,} benchmark requests into 3-tier decontaminator "
            f"(audit SHA-256: {audit['audit_signature_sha256']})."
        )
    else:
        print("Pass --smoke-test, --curate-all, or --suite-jsonl for full corpus decontamination.")


if __name__ == "__main__":
    main()
