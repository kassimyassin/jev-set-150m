"""JEV-Set-150M: Tree-Masked Shared-State Encoder with Permutation-Equivariant Set-Interaction Head."""

from .calibration import (
    TemperatureCalibrationResult,
    apply_cardinality_calibration,
    compute_brier_score,
    compute_ece,
    fit_cardinality_temperatures_lbfgs,
    select_multi_label_indices,
)
from .data_curation import (
    CANONICAL_JSONL_FIELDS,
    DATASET_INVENTORY_16,
    UNSEEN_15_OOD_TASKS,
    DatasetGroupSpec,
    ThreeTierDecontaminator,
    build_cluster_stratified_5fold_s_val,
    build_o_val_15task_holdout,
    compute_dataset_sampling_weights,
    convert_upstream_row_to_canonical,
    permute_and_curriculum_subsample_options,
    resolve_dataset_inventory,
    stream_hf_dataset_group,
    verify_zero_fold_leakage,
)
from .losses import JEVCompositeDecisionLoss, lambdarank_ndcg_loss, zlpr_multi_label_loss
from .modeling_jev_set import (
    QUESTION_ID_TO_REGIME,
    QUESTION_TYPE_TO_ID,
    EquivariantSetInteractionHead,
    EttinBackbone,
    JEVSetConfig,
    JEVSetForDecisionIndex,
    forward_single_request,
)
from .packing import PackedOptionTreeBatch, build_tree_masked_packs

__all__ = [
    "CANONICAL_JSONL_FIELDS",
    "DATASET_INVENTORY_16",
    "UNSEEN_15_OOD_TASKS",
    "DatasetGroupSpec",
    "QUESTION_ID_TO_REGIME",
    "QUESTION_TYPE_TO_ID",
    "EquivariantSetInteractionHead",
    "EttinBackbone",
    "JEVSetConfig",
    "JEVSetForDecisionIndex",
    "PackedOptionTreeBatch",
    "TemperatureCalibrationResult",
    "ThreeTierDecontaminator",
    "apply_cardinality_calibration",
    "build_cluster_stratified_5fold_s_val",
    "build_o_val_15task_holdout",
    "build_tree_masked_packs",
    "compute_brier_score",
    "compute_dataset_sampling_weights",
    "compute_ece",
    "convert_upstream_row_to_canonical",
    "fit_cardinality_temperatures_lbfgs",
    "forward_single_request",
    "JEVCompositeDecisionLoss",
    "lambdarank_ndcg_loss",
    "permute_and_curriculum_subsample_options",
    "resolve_dataset_inventory",
    "select_multi_label_indices",
    "stream_hf_dataset_group",
    "verify_zero_fold_leakage",
    "zlpr_multi_label_loss",
]

