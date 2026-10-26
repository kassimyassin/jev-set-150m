# src/jev_set/data_curation.py
from __future__ import annotations

import gzip
import hashlib
import json
import math
import random
import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
import torch.nn.functional as F

try:
    from datasketch import MinHash, MinHashLSH
except ImportError:  # pragma: no cover
    MinHash = None
    MinHashLSH = None


CANONICAL_JSONL_FIELDS: tuple[str, ...] = (
    "id",
    "dataset",
    "area",
    "is_gold_benchmark",
    "is_ood_general",
    "group_id",
    "state",
    "instructions",
    "options",
    "question_type",
    "select_k",
    "gold_indices",
    "gold_ranking",
    "teacher_probs",
    "difficulty_weight",
)

# 37 active public benchmark names in decision-index-0.3 to exclude from broad OOD corpora
PUBLIC_37_BENCHMARK_NAMES: frozenset[str] = frozenset(
    {
        "gpqa",
        "gpqa_diamond",
        "mmlu_pro",
        "hle",
        "bbh",
        "big_bench_hard",
        "winogrande",
        "gsm8k",
        "chessbench",
        "musr",
        "sata_bench",
        "cruxeval",
        "cladder",
        "anli",
        "hellaswag",
        "contract_nli",
        "contractnli",
        "finentity",
        "acos",
        "isarcasmeval",
        "vast",
        "nli4ct",
        "ragtruth",
        "banking77",
        "clinc150",
        "clinc_oos",
        "bright",
        "esci",
        "amazon_esci",
        "phishnchips",
        "hover",
        "bfcl",
        "api_bank",
        "apibank",
        "toolret",
        "home_appliances",
        "when2call",
        "bpomp",
        "humicroedit",
        "pop909",
        "cfcolor",
        "habermas",
        "new_yorker",
    }
)


@dataclass(frozen=True)
class DatasetGroupSpec:
    group_index: int
    group_name: str
    hf_identifiers: tuple[str, ...]
    area: str
    split: str
    raw_rows_estimate: int
    selected_rows: int
    license: str
    is_gold_benchmark: bool
    is_ood_general: bool
    default_question_type: str = "choice"
    commercial_clean_replacements: tuple[str, ...] = ()


DATASET_INVENTORY_16: tuple[DatasetGroupSpec, ...] = (
    # Pool A: Broad Multi-Task OOD Corpora (is_ood_general = True)
    DatasetGroupSpec(
        group_index=1,
        group_name="tasksource_instruct_v0",
        hf_identifiers=("tasksource/tasksource-instruct-v0",),
        area="knowledge",
        split="train",
        raw_rows_estimate=2_400_000,
        selected_rows=420_000,
        license="Apache-2.0 / MIT",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=2,
        group_name="gliclass_v4_data",
        hf_identifiers=("knowledgator/gliclass-v4-data",),
        area="retrieval",
        split="train",
        raw_rows_estimate=650_000,
        selected_rows=180_000,
        license="Apache-2.0",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="multi_label",
    ),
    DatasetGroupSpec(
        group_index=3,
        group_name="flan_v2_zero_shot",
        hf_identifiers=("allenai/flan_v2", "Muennighoff/flan"),
        area="knowledge",
        split="train",
        raw_rows_estimate=1_200_000,
        selected_rows=150_000,
        license="Apache-2.0",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=4,
        group_name="ms_marco_beir_ranking",
        hf_identifiers=("microsoft/ms_marco", "nthakur/beir-corpus"),
        area="retrieval",
        split="train",
        raw_rows_estimate=800_000,
        selected_rows=60_000,
        license="MIT / CC-BY-4.0",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="ranking",
    ),
    DatasetGroupSpec(
        group_index=5,
        group_name="glaive_xlam_function_calling",
        hf_identifiers=("glaiveai/glaive-function-calling-v2", "Salesforce/xlam-function-calling-60k"),
        area="tools",
        split="train",
        raw_rows_estimate=173_000,
        selected_rows=65_000,
        license="Apache-2.0 / CC-BY-4.0",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=6,
        group_name="shp2_helpsteer2_arena_preference",
        hf_identifiers=("stanfordnlp/SHP-2", "nvidia/HelpSteer2", "lmarena-ai/arena-human-preference-55k"),
        area="taste",
        split="train",
        raw_rows_estimate=410_000,
        selected_rows=55_000,
        license="CC-BY-4.0 / Apache-2.0",
        is_gold_benchmark=False,
        is_ood_general=True,
        default_question_type="Preference",
    ),
    # Pool B: Permitted Upstream Train Splits of Public Benchmark Families (is_ood_general = False)
    DatasetGroupSpec(
        group_index=7,
        group_name="mmlu_pro_aux_train",
        hf_identifiers=("TIGER-Lab/MMLU-Pro",),
        area="knowledge",
        split="validation",
        raw_rows_estimate=99_842,
        selected_rows=45_000,
        license="MIT",
        is_gold_benchmark=True,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=8,
        group_name="arc_sciq_openbookqa_medmcqa",
        hf_identifiers=("allenai/ai2_arc", "allenai/sciq", "allenai/openbookqa", "openlifescienceai/medmcqa"),
        area="knowledge",
        split="train",
        raw_rows_estimate=198_000,
        selected_rows=45_000,
        license="CC-BY-SA / Apache-2.0",
        is_gold_benchmark=True,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=9,
        group_name="gsm8k_metamath_dual_track",
        hf_identifiers=("openai/gsm8k", "meta-math/MetaMathQA"),
        area="knowledge",
        split="train",
        raw_rows_estimate=402_473,
        selected_rows=35_000,
        license="MIT",
        is_gold_benchmark=False,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=10,
        group_name="searchless_chess_train",
        hf_identifiers=("deepmind/searchless_chess",),
        area="knowledge",
        split="train",
        raw_rows_estimate=10_000_000,
        selected_rows=25_000,
        license="Apache-2.0",
        is_gold_benchmark=False,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=11,
        group_name="cladder_mbpp_musr_reasoning",
        hf_identifiers=("causal-nlp/cladder", "google-research-datasets/mbpp", "TAUR-Lab/MuSR"),
        area="knowledge",
        split="train",
        raw_rows_estimate=45_000,
        selected_rows=25_000,
        license="MIT / Apache-2.0",
        is_gold_benchmark=False,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=12,
        group_name="anli_hellaswag_winogrande_nlu",
        hf_identifiers=("facebook/anli", "Rowan/hellaswag", "allenai/winogrande"),
        area="language",
        split="train",
        raw_rows_estimate=243_000,
        selected_rows=65_000,
        license="MIT / Apache-2.0",
        is_gold_benchmark=True,
        is_ood_general=False,
        default_question_type="choice",
        commercial_clean_replacements=("nyu-mll/multi_nli", "stanfordnlp/snli", "Rowan/hellaswag", "allenai/winogrande"),
    ),
    DatasetGroupSpec(
        group_index=13,
        group_name="contractnli_finentity_acos_vast_nli4ct_ragtruth",
        hf_identifiers=(
            "kiddothe2b/contract-nli",
            "financial_phrasebank",
            "acos_aspect_sentiment",
            "vast_stance",
            "nli4ct_clinical",
            "ragtruth_hallucination",
        ),
        area="language",
        split="train",
        raw_rows_estimate=68_500,
        selected_rows=45_000,
        license="MIT / CC-BY-4.0",
        is_gold_benchmark=False,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=14,
        group_name="banking77_clinc150_esci_hover",
        hf_identifiers=("PolyAI/banking77", "clinc/clinc_oos", "tasksource/esci", "hover_verification"),
        area="retrieval",
        split="train",
        raw_rows_estimate=210_000,
        selected_rows=55_000,
        license="CC-BY-4.0 / Apache-2.0",
        is_gold_benchmark=True,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=15,
        group_name="bfcl_apibank_when2call_tools",
        hf_identifiers=("gorilla-llm/Berkeley-Function-Calling-Leaderboard", "liminghao1630/API-Bank", "nvidia/When2Call"),
        area="tools",
        split="train",
        raw_rows_estimate=85_000,
        selected_rows=35_000,
        license="Apache-2.0",
        is_gold_benchmark=True,
        is_ood_general=False,
        default_question_type="choice",
    ),
    DatasetGroupSpec(
        group_index=16,
        group_name="humicroedit_bpomp_habermas_pop909_cfcolor",
        hf_identifiers=("humicroedit", "bpomp_preference", "habermas_consensus", "pop909_match", "cfcolor_preference"),
        area="taste",
        split="train",
        raw_rows_estimate=48_000,
        selected_rows=25_000,
        license="MIT / CC-BY-4.0",
        is_gold_benchmark=False,
        is_ood_general=False,
        default_question_type="Preference",
    ),
)

# 15 Unseen New-Domain Holdout Tasks (O_val) across all 5 capability areas (3 per area)
UNSEEN_15_OOD_TASKS: tuple[dict[str, Any], ...] = (
    {"task_id": "ood_01_symbolic_modal_logic", "area": "knowledge", "question_type": "choice", "k": 6},
    {"task_id": "ood_02_pharmacokinetics_dosing", "area": "knowledge", "question_type": "choice", "k": 8},
    {"task_id": "ood_03_quantum_stabilizer_codes", "area": "knowledge", "question_type": "multi_label", "k": 6},
    {"task_id": "ood_04_maritime_charterparty_nli", "area": "language", "question_type": "choice", "k": 3},
    {"task_id": "ood_05_diplomatic_cable_stance", "area": "language", "question_type": "choice", "k": 4},
    {"task_id": "ood_06_radiology_impression_audit", "area": "language", "question_type": "multi_label", "k": 5},
    {"task_id": "ood_07_patent_prior_art_rerank", "area": "retrieval", "question_type": "ranking", "k": 16},
    {"task_id": "ood_08_hs_tariff_code_80way", "area": "retrieval", "question_type": "choice", "k": 80},
    {"task_id": "ood_09_genomic_pathway_lookup", "area": "retrieval", "question_type": "ranking", "k": 12},
    {"task_id": "ood_10_terraform_provider_routing", "area": "tools", "question_type": "choice", "k": 12},
    {"task_id": "ood_11_sql_dialect_migration_tool", "area": "tools", "question_type": "multi_label", "k": 8},
    {"task_id": "ood_12_avionics_telemetry_abstain", "area": "tools", "question_type": "choice", "k": 4},
    {"task_id": "ood_13_industrial_ergonomics_pref", "area": "taste", "question_type": "Preference", "k": 2},
    {"task_id": "ood_14_orchestral_voicing_match", "area": "taste", "question_type": "Match", "k": 16},
    {"task_id": "ood_15_cartographic_palette_pref", "area": "taste", "question_type": "Preference", "k": 4},
)


def resolve_dataset_inventory(commercial_clean: bool = False) -> list[DatasetGroupSpec]:
    """
    Returns the 16 dataset groups from Section 7 of PLAN.md.
    When `commercial_clean=True`, substitutes `facebook/anli` (CC-BY-NC 4.0) in Group 12
    with `nyu-mll/multi_nli` + `stanfordnlp/snli` so all 16 groups are strictly commercial-clean.
    """
    resolved: list[DatasetGroupSpec] = []
    for spec in DATASET_INVENTORY_16:
        if commercial_clean and spec.commercial_clean_replacements:
            resolved.append(
                DatasetGroupSpec(
                    group_index=spec.group_index,
                    group_name=spec.group_name,
                    hf_identifiers=spec.commercial_clean_replacements,
                    area=spec.area,
                    split=spec.split,
                    raw_rows_estimate=spec.raw_rows_estimate,
                    selected_rows=spec.selected_rows,
                    license="MIT / Apache-2.0 (Commercial-Clean)",
                    is_gold_benchmark=spec.is_gold_benchmark,
                    is_ood_general=spec.is_ood_general,
                    default_question_type=spec.default_question_type,
                    commercial_clean_replacements=spec.commercial_clean_replacements,
                )
            )
        else:
            resolved.append(spec)
    return resolved


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", str(text).lower())).strip()


def char_ngrams(text: str, n: int = 13) -> set[str]:
    norm = normalize_text(text)
    if not norm:
        return {""}
    if len(norm) < n:
        return {norm}
    return {norm[i : i + n] for i in range(len(norm) - n + 1)}


def extract_char_spans(norm_text: str, span_chars: int = 64) -> set[str]:
    """
    Extracts contiguous normalized spans of length `span_chars` (64 chars = 52 consecutive 13-grams).
    Two texts share a contiguous normalized span of >= 64 characters iff they share a 64-char window.
    """
    if len(norm_text) < span_chars:
        return set()
    return {norm_text[i : i + span_chars] for i in range(len(norm_text) - span_chars + 1)}


def make_minhash(shingles: Iterable[str], num_perm: int = 128) -> Any:
    if MinHash is None:
        return set(shingles)
    m = MinHash(num_perm=num_perm)
    for s in shingles:
        m.update(s.encode("utf-8"))
    return m


def compute_deterministic_text_embeddings(texts: Sequence[str], dim: int = 256) -> torch.Tensor:
    """
    Fast, deterministic character-4gram + word-unigram hashed dense embedding fallback
    used when offline or when a neural encoder checkpoint is not loaded in memory.
    """
    if not texts:
        return torch.zeros((0, dim), dtype=torch.float32)
    mat = torch.zeros((len(texts), dim), dtype=torch.float32)
    for row_idx, t in enumerate(texts):
        norm = normalize_text(t)
        if not norm:
            mat[row_idx, 0] = 1.0
            continue
        tokens = norm.split()
        grams = list(tokens) + [norm[i : i + 4] for i in range(max(1, len(norm) - 3))]
        for g in grams:
            digest = hashlib.blake2b(g.encode("utf-8"), digest_size=8).digest()
            val = int.from_bytes(digest, "little")
            bucket = val % dim
            sign = 1.0 if ((val >> 32) & 1) == 0 else -1.0
            mat[row_idx, bucket] += sign
    return F.normalize(mat, p=2, dim=-1)


class ThreeTierDecontaminator:
    """
    3-Tier Decontamination Pipeline against the 140,178 scoreable requests in decision-index-0.3:
      - Tier 1: Exact normalized SHA-256 hash match
      - Tier 2: 13-gram MinHash LSH (Jaccard >= 0.70) AND/OR exact shared 13-gram span >= 64 characters
      - Tier 3: Dense embedding cosine similarity >= 0.90 using batched encoder embeddings
                (with clean offline fallback) + cryptographic SHA-256 audit report emission.
    """

    def __init__(
        self,
        jaccard_threshold: float = 0.70,
        cosine_threshold: float = 0.90,
        shared_span_chars: int = 64,
        num_perm: int = 128,
        encoder_model_id: str = "jhu-clsp/ettin-encoder-150m",
    ) -> None:
        self.exact_hashes: set[str] = set()
        self.jaccard_threshold = jaccard_threshold
        self.cosine_threshold = cosine_threshold
        self.shared_span_chars = shared_span_chars
        self.num_perm = num_perm
        self.encoder_model_id = encoder_model_id

        if MinHashLSH is not None:
            self.lsh: Any = MinHashLSH(threshold=jaccard_threshold, num_perm=num_perm)
        else:
            self.lsh = None

        self._reference_shingles: dict[str, set[str]] = {}
        self._span64_hashes: set[bytes] = set()
        self._anchor_to_ref_positions: dict[bytes, list[tuple[int, int]]] = defaultdict(list)
        self._normalized_ref_texts: list[str] = []
        self._test_embeddings: Optional[torch.Tensor] = None
        self._custom_embeddings_set: bool = False
        self._reference_texts: list[str] = []

        # Audit counters
        self.total_checked: int = 0
        self.total_contaminated: int = 0
        self.tier1_exact_hits: int = 0
        self.tier2_minhash_hits: int = 0
        self.tier2_span64_hits: int = 0
        self.tier3_cosine_hits: int = 0
        self._clean_record_hashes: list[str] = []

    @staticmethod
    def _hash_span(span: str) -> bytes:
        return hashlib.blake2b(span.encode("utf-8"), digest_size=12).digest()

    def _span_stride_and_anchor_len(self) -> tuple[int, int]:
        S = max(1, int(self.shared_span_chars))
        stride = max(1, S // 4)
        anchor_len = S - stride + 1
        return stride, anchor_len

    def index_benchmark_requests(
        self,
        suite_jsonl_path: Path,
        compute_embeddings: bool = True,
        batch_size: int = 64,
    ) -> int:
        count = 0
        open_fn = gzip.open if suite_jsonl_path.suffix == ".gz" else open
        with open_fn(suite_jsonl_path, "rt", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                state = str(row.get("state", ""))
                instructions = str(row.get("instructions", ""))
                if not instructions and isinstance(row.get("questions"), dict):
                    instructions = " ".join(
                        str(q.get("instructions", "")) for q in row["questions"].values()
                    )
                self.add_reference_text(f"test_{count}", state, instructions)
                count += 1

        if compute_embeddings and self._reference_texts:
            ref_embs = self.encode_texts_batched(self._reference_texts, batch_size=batch_size)
            self._test_embeddings = F.normalize(ref_embs.float(), p=2, dim=-1)
        return count

    def add_reference_text(self, key: str, state: str, instructions: str) -> None:
        combined = f"{state} {instructions}"
        norm = normalize_text(combined)
        ref_idx = len(self._reference_texts)
        self._reference_texts.append(combined)
        self._normalized_ref_texts.append(norm)
        self.exact_hashes.add(hashlib.sha256(norm.encode("utf-8")).hexdigest())

        shingles = char_ngrams(combined, n=13)
        self._reference_shingles[key] = shingles
        if self.lsh is not None:
            mh = make_minhash(shingles, num_perm=self.num_perm)
            try:
                self.lsh.insert(key, mh)
            except ValueError:
                # Duplicate key already inserted
                pass

        if self.shared_span_chars > 0 and len(norm) >= self.shared_span_chars:
            stride, anchor_len = self._span_stride_and_anchor_len()
            for pos in range(0, len(norm) - anchor_len + 1, stride):
                h = self._hash_span(norm[pos : pos + anchor_len])
                self._span64_hashes.add(h)
                self._anchor_to_ref_positions[h].append((ref_idx, pos))

        if not self._custom_embeddings_set:
            self._test_embeddings = None

    def set_reference_embeddings(self, embeddings: torch.Tensor) -> None:
        self._test_embeddings = F.normalize(embeddings.float(), p=2, dim=-1)
        self._custom_embeddings_set = True

    def encode_texts_batched(
        self,
        texts: Sequence[str],
        batch_size: int = 32,
        use_pretrained_encoder: bool = False,
        device: str = "cpu",
    ) -> torch.Tensor:
        """
        Computes L2-normalized dense embeddings in batches for Tier 3 cosine similarity filtering.
        Uses `self.encoder_model_id` via Hugging Face `transformers` when `use_pretrained_encoder=True`,
        with automatic fallback to deterministic dense embeddings when offline.
        """
        if not texts:
            return torch.zeros((0, 256), dtype=torch.float32)

        if use_pretrained_encoder:
            try:
                from transformers import AutoModel, AutoTokenizer

                tok = AutoTokenizer.from_pretrained(self.encoder_model_id, trust_remote_code=True)
                enc = AutoModel.from_pretrained(self.encoder_model_id, trust_remote_code=True).to(device)
                enc.eval()
                batches: list[torch.Tensor] = []
                with torch.no_grad():
                    for start in range(0, len(texts), batch_size):
                        batch_texts = list(texts[start : start + batch_size])
                        inputs = tok(
                            batch_texts,
                            padding=True,
                            truncation=True,
                            max_length=512,
                            return_tensors="pt",
                        ).to(device)
                        out = enc(**inputs).last_hidden_state
                        mask = inputs["attention_mask"].unsqueeze(-1).float()
                        pooled = (out.float() * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
                        batches.append(F.normalize(pooled.cpu(), p=2, dim=-1))
                return torch.cat(batches, dim=0)
            except Exception:
                pass

        batches_fallback: list[torch.Tensor] = []
        for start in range(0, len(texts), max(1, batch_size)):
            chunk = texts[start : start + max(1, batch_size)]
            batches_fallback.append(compute_deterministic_text_embeddings(chunk, dim=256))
        return torch.cat(batches_fallback, dim=0)

    def has_shared_span(self, norm_text: str) -> bool:
        """
        Checks whether `norm_text` shares any exact contiguous span >= `shared_span_chars` (64 chars)
        using strided pigeonhole anchors (16x memory reduction) followed by exact substring verification.
        """
        S = self.shared_span_chars
        if S <= 0 or len(norm_text) < S or not self._anchor_to_ref_positions:
            return False
        _, anchor_len = self._span_stride_and_anchor_len()
        for j in range(len(norm_text) - anchor_len + 1):
            h = self._hash_span(norm_text[j : j + anchor_len])
            hits = self._anchor_to_ref_positions.get(h)
            if not hits:
                continue
            for ref_idx, ref_pos in hits:
                ref_norm = self._normalized_ref_texts[ref_idx]
                offset = ref_pos - j
                min_c = max(0, j + anchor_len - S, -offset)
                max_c = min(j, len(norm_text) - S, len(ref_norm) - S - offset)
                for start_c in range(min_c, max_c + 1):
                    start_r = start_c + offset
                    if norm_text[start_c : start_c + S] == ref_norm[start_r : start_r + S]:
                        return True
        return False

    def inspect_candidate(
        self,
        state: str,
        instructions: str,
        candidate_embedding: Optional[torch.Tensor] = None,
    ) -> dict[str, Any]:
        """
        Evaluates a candidate `(state, instructions)` across all 3 tiers and returns a detailed verdict.
        """
        self.total_checked += 1
        combined = f"{state} {instructions}"
        norm = normalize_text(combined)
        cand_sha = hashlib.sha256(norm.encode("utf-8")).hexdigest()

        # Tier 1: Exact normalized SHA-256 hash match
        if cand_sha in self.exact_hashes:
            self.total_contaminated += 1
            self.tier1_exact_hits += 1
            return {"contaminated": True, "tier": 1, "reason": "tier1_exact_sha256", "sha256": cand_sha}

        # Tier 2a: 13-gram MinHash LSH (with exact Jaccard verification >= 0.70)
        shingles = char_ngrams(combined, n=13)
        minhash_hit = False
        if self.lsh is not None:
            mh = make_minhash(shingles, num_perm=self.num_perm)
            candidate_keys = self.lsh.query(mh)
            for k in candidate_keys:
                ref_shingles = self._reference_shingles.get(k)
                if ref_shingles is not None:
                    union_len = len(shingles | ref_shingles)
                    if union_len > 0 and (len(shingles & ref_shingles) / union_len) >= self.jaccard_threshold:
                        minhash_hit = True
                        break
                else:
                    minhash_hit = True
                    break
        else:
            for ref in self._reference_shingles.values():
                union_len = len(shingles | ref)
                if union_len > 0 and (len(shingles & ref) / union_len) >= self.jaccard_threshold:
                    minhash_hit = True
                    break

        if minhash_hit:
            self.total_contaminated += 1
            self.tier2_minhash_hits += 1
            return {"contaminated": True, "tier": 2, "reason": "tier2_minhash_jaccard", "sha256": cand_sha}

        # Tier 2b: Exact shared 13-gram contiguous span >= 64 characters
        if self.has_shared_span(norm):
            self.total_contaminated += 1
            self.tier2_span64_hits += 1
            return {"contaminated": True, "tier": 2, "reason": "tier2_shared_span_64chars", "sha256": cand_sha}

        # Tier 3: Dense embedding cosine similarity >= 0.90
        if self._test_embeddings is None and self._reference_texts:
            ref_embs = self.encode_texts_batched(self._reference_texts, batch_size=64)
            self._test_embeddings = F.normalize(ref_embs.float(), p=2, dim=-1)

        if self._test_embeddings is not None and self._test_embeddings.numel() > 0:
            if candidate_embedding is None and self._test_embeddings.size(1) == 256:
                candidate_embedding = self.encode_texts_batched([combined], batch_size=1)[0]
            if candidate_embedding is not None and candidate_embedding.numel() == self._test_embeddings.size(1):
                cand_norm = F.normalize(candidate_embedding.float().view(1, -1), p=2, dim=-1)
                max_cos = float(torch.matmul(cand_norm, self._test_embeddings.T).max().item())
                if max_cos >= self.cosine_threshold:
                    self.total_contaminated += 1
                    self.tier3_cosine_hits += 1
                    return {
                        "contaminated": True,
                        "tier": 3,
                        "reason": "tier3_dense_cosine",
                        "max_cosine": round(max_cos, 6),
                        "sha256": cand_sha,
                    }

        self._clean_record_hashes.append(cand_sha)
        return {"contaminated": False, "tier": 0, "reason": "clean", "sha256": cand_sha}

    def is_contaminated(
        self,
        state: str,
        instructions: str,
        candidate_embedding: Optional[torch.Tensor] = None,
    ) -> bool:
        verdict = self.inspect_candidate(state, instructions, candidate_embedding=candidate_embedding)
        return bool(verdict["contaminated"])

    def emit_audit_report(
        self,
        output_path: Optional[Path] = None,
        suite_edition: str = "decision-index-0.3",
    ) -> dict[str, Any]:
        """
        Emits a cryptographic SHA-256 decontamination audit report to `data/processed/decontamination_audit.json`.
        """
        sorted_ref_hashes = sorted(self.exact_hashes)
        ref_index_sha256 = hashlib.sha256("\n".join(sorted_ref_hashes).encode("utf-8")).hexdigest()
        clean_corpus_sha256 = hashlib.sha256(
            "\n".join(self._clean_record_hashes).encode("utf-8")
        ).hexdigest()

        payload: dict[str, Any] = {
            "suite_edition": suite_edition,
            "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "indexed_reference_requests": len(self.exact_hashes),
            "reference_index_sha256": ref_index_sha256,
            "thresholds": {
                "tier1_exact_sha256": True,
                "tier2_minhash_jaccard": self.jaccard_threshold,
                "tier2_shared_span_chars": self.shared_span_chars,
                "tier3_dense_cosine": self.cosine_threshold,
                "minhash_num_perm": self.num_perm,
            },
            "statistics": {
                "total_checked": self.total_checked,
                "total_contaminated": self.total_contaminated,
                "clean_retained": len(self._clean_record_hashes),
                "tier1_exact_hits": self.tier1_exact_hits,
                "tier2_minhash_hits": self.tier2_minhash_hits,
                "tier2_span64_hits": self.tier2_span64_hits,
                "tier3_cosine_hits": self.tier3_cosine_hits,
            },
            "clean_corpus_sha256": clean_corpus_sha256,
        }
        canonical_bytes = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        payload["audit_signature_sha256"] = hashlib.sha256(canonical_bytes).hexdigest()

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

        return payload


def convert_upstream_row_to_canonical(
    raw_row: dict[str, Any],
    group_spec: DatasetGroupSpec,
    row_idx: int = 0,
    rng: Optional[random.Random] = None,
) -> Optional[dict[str, Any]]:
    """
    Converts a raw upstream Hugging Face dataset row from any of the 16 dataset groups
    into the 15-field canonical JSONL format required by JEV-Set-150M.
    Returns `None` if the row comes from one of the 37 excluded public benchmarks in `tasksource`.
    """
    rng = rng or random.Random(42 + row_idx)
    task_name = str(raw_row.get("task", raw_row.get("dataset", group_spec.group_name))).lower()
    if group_spec.is_ood_general:
        norm_task = re.sub(r"[^a-z0-9]+", "_", task_name).strip("_")
        if norm_task in PUBLIC_37_BENCHMARK_NAMES:
            return None

    qtype = str(raw_row.get("question_type", group_spec.default_question_type))
    state = str(
        raw_row.get(
            "state",
            raw_row.get(
                "context",
                raw_row.get(
                    "premise",
                    raw_row.get("ctx", raw_row.get("sentence", raw_row.get("question", ""))),
                ),
            ),
        )
    ).strip()
    instructions = str(
        raw_row.get(
            "instructions",
            raw_row.get(
                "question",
                raw_row.get("hypothesis", raw_row.get("query", f"Solve {group_spec.group_name} task.")),
            ),
        )
    ).strip()
    if not state:
        state = instructions or f"Context for {group_spec.group_name} sample {row_idx}."

    options: list[str] = []
    gold_indices: list[int] = [0]

    if "options" in raw_row and isinstance(raw_row["options"], (list, tuple)) and len(raw_row["options"]) >= 2:
        options = [str(o) for o in raw_row["options"]]
    elif "choices" in raw_row:
        ch = raw_row["choices"]
        if isinstance(ch, dict) and "text" in ch:
            options = [str(o) for o in ch["text"]]
        elif isinstance(ch, (list, tuple)):
            options = [str(o) for o in ch]
    elif "endings" in raw_row and isinstance(raw_row["endings"], (list, tuple)):
        options = [str(o) for o in raw_row["endings"]]
    elif "all_labels" in raw_row and isinstance(raw_row["all_labels"], (list, tuple)):
        options = [str(o) for o in raw_row["all_labels"]]
        qtype = "multi_label"
    elif "option1" in raw_row and "option2" in raw_row:
        options = [str(raw_row["option1"]), str(raw_row["option2"])]
    elif "response_a" in raw_row and "response_b" in raw_row:
        options = [str(raw_row["response_a"]), str(raw_row["response_b"])]
        qtype = "Preference"
    elif "premise" in raw_row and "hypothesis" in raw_row:
        options = ["Entailment", "Neutral", "Contradiction"]
    elif "answer" in raw_row and ("####" in str(raw_row["answer"]) or group_spec.group_index == 9):
        ans_str = str(raw_row["answer"]).split("####")[-1].strip()
        try:
             base_val = float(re.sub(r"[^\d.\-]", "", ans_str) or "42")
        except ValueError:
            base_val = 42.0
        k_choice = 10 if (row_idx % 2 == 1) else 4
        distractors = [str(int(base_val + delta)) for delta in (-12, -5, -2, -1, 1, 2, 5, 10, 15)]
        options = [ans_str] + distractors[: k_choice - 1]
        gold_indices = [0]

    if len(options) < 2:
        options = [f"Candidate A for {group_spec.group_name}", f"Candidate B for {group_spec.group_name}", f"Candidate C for {group_spec.group_name}"]

    # Resolve gold_indices
    if "gold_indices" in raw_row and isinstance(raw_row["gold_indices"], (list, tuple)):
        gold_indices = [int(i) for i in raw_row["gold_indices"] if 0 <= int(i) < len(options)]
    elif "true_labels" in raw_row and isinstance(raw_row["true_labels"], (list, tuple)):
        true_set = {str(x) for x in raw_row["true_labels"]}
        gold_indices = [i for i, opt in enumerate(options) if opt in true_set]
    elif "answer_index" in raw_row and raw_row["answer_index"] is not None:
        gold_indices = [max(0, min(len(options) - 1, int(raw_row["answer_index"])))]
    elif "label" in raw_row and raw_row["label"] is not None:
        lbl = raw_row["label"]
        if isinstance(lbl, int) and 0 <= lbl < len(options):
            gold_indices = [lbl]
        elif isinstance(lbl, str) and lbl.strip().upper() in ("A", "B", "C", "D", "E", "F", "G", "H", "I", "J"):
            idx = ord(lbl.strip().upper()) - ord("A")
            gold_indices = [max(0, min(len(options) - 1, idx))]
        elif isinstance(lbl, str) and lbl.isdigit():
            gold_indices = [max(0, min(len(options) - 1, int(lbl)))]

    if not gold_indices:
        gold_indices = [0]

    K = len(options)
    gold_set = set(gold_indices)
    gold_ranking = list(gold_indices) + [i for i in range(K) if i not in gold_set]
    group_id = str(raw_row.get("group_id", f"{group_spec.group_name}_grp_{row_idx // 2:06d}"))

    return {
        "id": str(raw_row.get("id", f"{group_spec.group_name}_{row_idx:07d}")),
        "dataset": group_spec.group_name,
        "area": group_spec.area,
        "is_gold_benchmark": bool(group_spec.is_gold_benchmark),
        "is_ood_general": bool(group_spec.is_ood_general),
        "group_id": group_id,
        "state": state,
        "instructions": instructions,
        "options": options,
        "question_type": qtype,
        "select_k": len(gold_indices) if qtype == "multi_label" else 1,
        "gold_indices": gold_indices,
        "gold_ranking": gold_ranking,
        "teacher_probs": raw_row.get("teacher_probs"),
        "difficulty_weight": 1.2 if group_spec.is_gold_benchmark else 1.0,
    }


def stream_hf_dataset_group(
    group_spec: DatasetGroupSpec,
    max_rows: int = 32,
    decontaminator: Optional[ThreeTierDecontaminator] = None,
    allow_synthetic_fallback: bool = True,
    offline: bool = False,
    seed: int = 42,
    stats: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """
    Streams or downloads up to `max_rows` clean canonical records from Hugging Face `datasets.load_dataset`
    for `group_spec`, filtering through `decontaminator`. Falls back to deterministic synthetic records
    when `offline=True` or when network/gated dataset errors occur if `allow_synthetic_fallback=True`.

    If `stats` is given it is filled with `real_rows`, `synthetic_rows`, `real_error` (first real-load
    exception string or None) and `contaminated_dropped`.
    """
    rng = random.Random(seed + group_spec.group_index)
    records: list[dict[str, Any]] = []
    row_counter = 0
    real_error: Optional[str] = None
    contaminated_dropped = 0

    if not offline:
        try:
            import datasets as hf_datasets

            for hf_id in group_spec.hf_identifiers:
                if len(records) >= max_rows:
                    break
                ds = hf_datasets.load_dataset(hf_id, split=group_spec.split, streaming=True)
                for raw_row in ds:
                    if len(records) >= max_rows:
                        break
                    idx = row_counter
                    row_counter += 1
                    canonical = convert_upstream_row_to_canonical(dict(raw_row), group_spec, row_idx=idx, rng=rng)
                    if canonical is None:
                        continue
                    if decontaminator is not None and decontaminator.is_contaminated(
                        canonical["state"], canonical["instructions"]
                    ):
                        contaminated_dropped += 1
                        continue
                    canonical["source"] = "real"
                    records.append(canonical)
        except Exception as exc:
            real_error = f"{type(exc).__name__}: {str(exc)[:200]}"
            if not allow_synthetic_fallback:
                if stats is not None:
                    stats.update(
                        real_rows=len(records), synthetic_rows=0, real_error=real_error,
                        contaminated_dropped=contaminated_dropped,
                    )
                raise

    real_rows = len(records)
    if len(records) < max_rows and allow_synthetic_fallback:
        max_attempts = max(max_rows * 4, 64)
        attempts = 0
        while len(records) < max_rows and attempts < max_attempts:
            idx = row_counter
            row_counter += 1
            attempts += 1
            K = rng.choice([3, 4, 8, 10])
            opts = [f"{group_spec.group_name} candidate option {k} (item {idx})" for k in range(K)]
            qtype = group_spec.default_question_type
            golds = sorted(rng.sample(range(K), k=2)) if qtype == "multi_label" else [rng.randrange(K)]
            synth_raw = {
                "id": f"{group_spec.group_name}_{idx:07d}",
                "group_id": f"{group_spec.group_name}_grp_{idx // 2:05d}",
                "state": f"Domain context [{group_spec.area}] from {group_spec.hf_identifiers[0]} case #{idx // 2}.",
                "instructions": f"Evaluate {group_spec.group_name} decision criteria for item #{idx}.",
                "options": opts,
                "question_type": qtype,
                "gold_indices": golds,
            }
            canonical = convert_upstream_row_to_canonical(synth_raw, group_spec, row_idx=idx, rng=rng)
            if canonical is None:
                continue
            if decontaminator is not None and decontaminator.is_contaminated(
                canonical["state"], canonical["instructions"]
            ):
                continue
            canonical["source"] = "synthetic"
            records.append(canonical)

    if stats is not None:
        stats.update(
            real_rows=real_rows,
            synthetic_rows=len(records) - real_rows,
            real_error=real_error,
            contaminated_dropped=contaminated_dropped,
        )
    return records


def compute_dataset_sampling_weights(
    dataset_counts: dict[str, int],
    is_gold: dict[str, bool],
    is_ood: dict[str, bool],
    target_ood_share: float = 0.55,
    alpha: float = 0.55,
    cap: int = 50_000,
    gold_multiplier: float = 1.2,
    stage: Optional[str] = None,
) -> dict[str, float]:
    """
    Temperature-Scaled Square-Root Sampling Law (Section 8.2 of PLAN.md):
      P(d) propto gamma_split(d) * w_gold(d) * min(N_d, 50000) ** 0.55
    Stage presets:
      - stage="stage1": 70% Broad OOD / 30% Same-Skill (gold_multiplier = 1.0)
      - stage="stage2": 55% Broad OOD / 45% Same-Skill (gold_multiplier = 1.0 or as specified)
      - stage="stage3": 35% Broad OOD / 65% Gold-Weighted Same-Skill (gold_multiplier = 1.2)
    """
    if stage is not None:
        stage_norm = stage.strip().lower()
        if "stage1" in stage_norm or stage_norm == "1":
            target_ood_share = 0.70
        elif "stage2" in stage_norm or stage_norm == "2":
            target_ood_share = 0.55
        elif "stage3" in stage_norm or stage_norm == "3":
            target_ood_share = 0.35
            gold_multiplier = 1.2

    ood_raw: dict[str, float] = {}
    same_raw: dict[str, float] = {}

    for d_name, n_rows in dataset_counts.items():
        if n_rows <= 0:
            continue
        g_mult = gold_multiplier if is_gold.get(d_name, False) else 1.0
        w = g_mult * (min(n_rows, cap) ** alpha)
        if is_ood.get(d_name, True):
            ood_raw[d_name] = w
        else:
            same_raw[d_name] = w

    weights: dict[str, float] = {}
    if ood_raw and same_raw:
        ood_sum = sum(ood_raw.values())
        same_sum = sum(same_raw.values())
        for d_name, w in ood_raw.items():
            weights[d_name] = target_ood_share * (w / ood_sum)
        for d_name, w in same_raw.items():
            weights[d_name] = (1.0 - target_ood_share) * (w / same_sum)
    elif ood_raw:
        ood_sum = sum(ood_raw.values())
        for d_name, w in ood_raw.items():
            weights[d_name] = w / ood_sum
    elif same_raw:
        same_sum = sum(same_raw.values())
        for d_name, w in same_raw.items():
            weights[d_name] = w / same_sum
    return weights


def permute_and_curriculum_subsample_options(
    record: dict[str, Any],
    rng: random.Random,
    full_k_prob: float = 0.70,
    hard_neg_k_range: tuple[int, int] = (8, 64),
) -> dict[str, Any]:
    """
    Implements Section 8.3 Dynamic Option-Order Shuffling & Cardinality Curriculum:
    - Permutes option order deterministically while keeping `gold_indices`, `gold_ranking`,
      and `teacher_probs` aligned.
    - Passes full K on 70% of steps and subsamples K' in [8, 64] on 30% of steps when K > 8.
    """
    options = list(record.get("options", []))
    K = len(options)
    if K <= 1:
        return dict(record)

    gold_indices = [int(i) for i in record.get("gold_indices", [0]) if 0 <= int(i) < K]
    if not gold_indices:
        gold_indices = [0]
    teacher_probs = record.get("teacher_probs")

    selected_old_indices = list(range(K))
    min_k, max_k = hard_neg_k_range
    if K > min_k and rng.random() > full_k_prob:
        target_k = rng.randint(min_k, min(K, max_k))
        gold_set = set(gold_indices)
        non_golds = [i for i in range(K) if i not in gold_set]
        if teacher_probs is not None and len(teacher_probs) == K:
            # Prefer highest-probability hard negatives
            non_golds.sort(key=lambda idx: float(teacher_probs[idx]), reverse=True)
        else:
            rng.shuffle(non_golds)
        keep_neg = max(0, target_k - len(gold_indices))
        selected_old_indices = list(gold_indices) + non_golds[:keep_neg]

    # Shuffle the chosen option indices
    rng.shuffle(selected_old_indices)
    old_to_new = {old_idx: new_idx for new_idx, old_idx in enumerate(selected_old_indices)}

    new_options = [options[old_idx] for old_idx in selected_old_indices]
    new_golds = sorted(old_to_new[g] for g in gold_indices if g in old_to_new)
    new_gold_set = set(new_golds)
    new_ranking = list(new_golds) + [i for i in range(len(new_options)) if i not in new_gold_set]

    new_teacher_probs: Optional[list[float]] = None
    if teacher_probs is not None and len(teacher_probs) == K:
        raw_tp = [float(teacher_probs[old_idx]) for old_idx in selected_old_indices]
        if record.get("question_type") != "multi_label":
            s = sum(raw_tp) or 1.0
            new_teacher_probs = [p / s for p in raw_tp]
        else:
            new_teacher_probs = raw_tp

    updated = dict(record)
    updated["options"] = new_options
    updated["gold_indices"] = new_golds
    updated["gold_ranking"] = new_ranking
    updated["teacher_probs"] = new_teacher_probs
    if updated.get("question_type") == "multi_label":
        updated["select_k"] = len(new_golds)
    return updated


def _resolve_non_empty_group_id(rec: dict[str, Any], idx: int) -> str:
    raw_gid = rec.get("group_id")
    if raw_gid is not None and str(raw_gid).strip():
        return str(raw_gid).strip()
    return f"row_{idx}"


def build_cluster_stratified_5fold_s_val(
    records: Sequence[dict[str, Any]],
    num_folds: int = 5,
    seed: int = 42,
) -> list[dict[str, Any]]:
    """
    Constructs the 5-Fold Cluster-Stratified Validation Split (`S_val`).
    Groups records via Union-Find across BOTH `group_id` AND normalized `state` document SHA-256
    so zero `group_id` leakage and zero source-document leakage can ever occur across folds.
    """
    n = len(records)
    if n == 0:
        return []
    num_folds = max(1, int(num_folds))

    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            if ra < rb:
                parent[rb] = ra
            else:
                parent[ra] = rb

    by_group_id: dict[str, int] = {}
    by_doc_hash: dict[str, int] = {}
    doc_hashes: list[str] = []

    for idx, rec in enumerate(records):
        gid = _resolve_non_empty_group_id(rec, idx)
        norm_state = normalize_text(str(rec.get("state", "")))
        d_hash = hashlib.sha256(norm_state.encode("utf-8")).hexdigest()[:16] if norm_state else f"empty_{idx}"
        doc_hashes.append(d_hash)

        if gid in by_group_id:
            union(idx, by_group_id[gid])
        else:
            by_group_id[gid] = idx

        if d_hash in by_doc_hash:
            union(idx, by_doc_hash[d_hash])
        else:
            by_doc_hash[d_hash] = idx

    clusters: dict[int, list[int]] = defaultdict(list)
    for idx in range(n):
        clusters[find(idx)].append(idx)

    # Group clusters by capability area for stratified fold assignment
    area_clusters: dict[str, list[tuple[int, list[int]]]] = defaultdict(list)
    for root_idx, member_indices in clusters.items():
        area = str(records[member_indices[0]].get("area", "general"))
        area_clusters[area].append((root_idx, member_indices))

    rng = random.Random(seed)
    fold_total_counts = [0] * num_folds
    fold_area_counts: dict[str, list[int]] = {area: [0] * num_folds for area in area_clusters}
    cluster_to_fold: dict[int, int] = {}

    for area in sorted(area_clusters.keys()):
        c_list = area_clusters[area]
        rng.shuffle(c_list)
        c_list.sort(key=lambda item: len(item[1]), reverse=True)
        for root_idx, members in c_list:
            # Choose fold with fewest items in this area, breaking ties by fewest total items
            best_fold = min(
                range(num_folds),
                key=lambda f: (fold_area_counts[area][f], fold_total_counts[f], f),
            )
            cluster_to_fold[root_idx] = best_fold
            fold_area_counts[area][best_fold] += len(members)
            fold_total_counts[best_fold] += len(members)

    annotated: list[dict[str, Any]] = []
    for idx, rec in enumerate(records):
        root_idx = find(idx)
        out_rec = dict(rec)
        out_rec["fold"] = int(cluster_to_fold[root_idx])
        out_rec["cluster_id"] = f"cluster_{root_idx:06d}"
        out_rec["doc_hash"] = doc_hashes[idx]
        annotated.append(out_rec)

    return annotated


def verify_zero_fold_leakage(fold_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    Verifies that no `group_id` and no normalized `state` `doc_hash` appears in more than one fold.
    """
    group_to_folds: dict[str, set[int]] = defaultdict(set)
    doc_to_folds: dict[str, set[int]] = defaultdict(set)
    fold_counts: dict[int, int] = defaultdict(int)

    for idx, rec in enumerate(fold_records):
        fold = int(rec["fold"])
        fold_counts[fold] += 1
        gid = _resolve_non_empty_group_id(rec, idx)
        norm_state = normalize_text(str(rec.get("state", "")))
        d_hash = rec.get(
            "doc_hash",
            hashlib.sha256(norm_state.encode("utf-8")).hexdigest()[:16] if norm_state else f"empty_{idx}",
        )
        group_to_folds[gid].add(fold)
        doc_to_folds[str(d_hash)].add(fold)

    leaked_groups = [gid for gid, fset in group_to_folds.items() if len(fset) > 1]
    leaked_docs = [dh for dh, fset in doc_to_folds.items() if len(fset) > 1]

    return {
        "num_records": len(fold_records),
        "num_folds": len(fold_counts),
        "fold_counts": dict(sorted(fold_counts.items())),
        "group_id_leakage_count": len(leaked_groups),
        "doc_hash_leakage_count": len(leaked_docs),
        "zero_leakage_verified": len(leaked_groups) == 0 and len(leaked_docs) == 0,
    }


def build_o_val_15task_holdout(
    samples_per_task: int = 8,
    seed: int = 2026,
) -> list[dict[str, Any]]:
    """
    Constructs the 15-Task Unseen New-Domain Holdout (`O_val`) spanning all 5 capability areas
    and all 5 question types (`choice`, `multi_label`, `ranking`, `Preference`, `Match`).
    """
    rng = random.Random(seed)
    holdout_rows: list[dict[str, Any]] = []

    for task_spec in UNSEEN_15_OOD_TASKS:
        task_id = str(task_spec["task_id"])
        area = str(task_spec["area"])
        qtype = str(task_spec["question_type"])
        K = int(task_spec["k"])

        for i in range(samples_per_task):
            options = [f"[{task_id}] candidate option {k + 1} for case {i}" for k in range(K)]
            if qtype == "multi_label":
                gold_indices = sorted(rng.sample(range(K), k=min(2, K)))
            else:
                gold_indices = [rng.randrange(K)]
            gold_set = set(gold_indices)
            gold_ranking = list(gold_indices) + [k for k in range(K) if k not in gold_set]

            holdout_rows.append(
                {
                    "id": f"{task_id}_{i:04d}",
                    "dataset": task_id,
                    "area": area,
                    "is_gold_benchmark": False,
                    "is_ood_general": True,
                    "group_id": f"{task_id}_grp_{i // 2:04d}",
                    "state": f"Unseen new-domain holdout state for {task_id} ({area}) scenario #{i // 2}.",
                    "instructions": f"Execute unseen {qtype} decision protocol for {task_id} instance #{i}.",
                    "options": options,
                    "question_type": qtype,
                    "select_k": len(gold_indices) if qtype == "multi_label" else 1,
                    "gold_indices": gold_indices,
                    "gold_ranking": gold_ranking,
                    "teacher_probs": None,
                    "difficulty_weight": 1.0,
                }
            )

    return holdout_rows
