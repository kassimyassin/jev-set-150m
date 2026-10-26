# src/jev_set/engine.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import torch
from transformers import AutoTokenizer

# Ensure external/decision-index is importable if present in repository root
_REPO_ROOT = Path(__file__).resolve().parents[2]
_EXT_DECISION_INDEX = _REPO_ROOT / "external" / "decision-index"
if _EXT_DECISION_INDEX.exists() and str(_EXT_DECISION_INDEX) not in sys.path:
    sys.path.insert(0, str(_EXT_DECISION_INDEX))

try:
    import decision_index.engines.base as _di_base
    import decision_index.scoring.index as _di_scoring_index
    from decision_index.engines.base import Engine, Unsupported, text
    from decision_index.pipeline import score_run as _di_score_run

    if not hasattr(_di_scoring_index, "score_run"):
        setattr(_di_scoring_index, "score_run", _di_score_run)
except ImportError:  # pragma: no cover
    _di_base = None

    class Unsupported(ValueError):  # type: ignore[no-redef]
        pass

    def text(x: Any) -> str:
        return x if isinstance(x, str) else str(x)

    class Engine:  # type: ignore[no-redef]
        name = "engine"
        provenance: dict[str, Any] = {}
        latency = "In-process request wall time including prompt construction; excludes model loading."

        def __init__(self, **options: Any) -> None:
            self.options = options


from .calibration import select_multi_label_indices
from .modeling_jev_set import JEVSetConfig, JEVSetForDecisionIndex, forward_single_request
from .packing import DeterministicWhitespaceTokenizer


@dataclass(frozen=True)
class EngineInfo:
    name: str
    model_id: str
    parameters: int
    context_length: int


@dataclass
class EngineRequest:
    request_id: str
    state: str
    instructions: str
    options: tuple[str, ...] | list[str]
    question_type: str = "choice"
    select_k: Optional[int] = None


@dataclass
class EngineResponse:
    request_id: str
    selected_indices: tuple[int, ...]
    ranked_indices: tuple[int, ...]
    option_scores: tuple[float, ...]
    status: str = "ok"


if _di_base is not None:
    setattr(_di_base, "EngineInfo", EngineInfo)
    setattr(_di_base, "EngineRequest", EngineRequest)
    setattr(_di_base, "EngineResponse", EngineResponse)


class JEVSetEngine(Engine):
    """
    Official Jev Decision Index v0.4.1 Engine implementation for JEV-Set-150M (149,936,905 params).
    Supports both:
      1. Structured `engine.run(EngineRequest)` (choice, multi_label, ranking, Preference, Match)
      2. Official `apolinario/decision-index` runner protocol `engine(state, questions)`
    """

    name = "jev-set-150m"

    def __init__(
        self,
        checkpoint_dir: str = "cross-encoder/ettin-reranker-150m-v1",
        device: str = "cuda",
        multi_label_threshold: float = 0.50,
        multi_label_max_gap: float = 1.35,
        load_pretrained: bool = True,
        tokenizer: Any = None,
        config: Optional[JEVSetConfig] = None,
        **options: Any,
    ) -> None:
        super().__init__(**options)
        if "model" in options and checkpoint_dir == "cross-encoder/ettin-reranker-150m-v1":
            checkpoint_dir = str(options["model"])

        use_cuda = device.startswith("cuda") and torch.cuda.is_available()
        self.device = torch.device(device if use_cuda else "cpu")
        if config is not None:
            if (
                checkpoint_dir != "cross-encoder/ettin-reranker-150m-v1"
                and config.backbone_name_or_path == "cross-encoder/ettin-reranker-150m-v1"
            ):
                config.backbone_name_or_path = checkpoint_dir
            self.cfg = config
        else:
            self.cfg = JEVSetConfig(backbone_name_or_path=checkpoint_dir)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        elif load_pretrained:
            ckpt_p = Path(checkpoint_dir)
            has_local_tok = ckpt_p.is_dir() and (
                (ckpt_p / "tokenizer.json").exists() or (ckpt_p / "tokenizer_config.json").exists()
            )
            tok_source = (
                checkpoint_dir
                if (has_local_tok or not ckpt_p.exists())
                else "cross-encoder/ettin-reranker-150m-v1"
            )
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tok_source, trust_remote_code=True)
            except Exception:
                self.tokenizer = DeterministicWhitespaceTokenizer()
        else:
            self.tokenizer = DeterministicWhitespaceTokenizer()

        self.model = JEVSetForDecisionIndex(self.cfg, load_pretrained=load_pretrained).to(self.device)
        self.model.eval()
        self.multi_label_threshold = multi_label_threshold
        self.multi_label_max_gap = multi_label_max_gap
        self.model_id = checkpoint_dir

        budget = self.model.verify_parameter_budget()
        self._info = EngineInfo(
            name="JEV-Set-150M",
            model_id=checkpoint_dir,
            parameters=budget["total_params"],
            context_length=32768,
        )
        self.provenance = {
            "kind": "jev-set-150m",
            "repo": checkpoint_dir,
            "parameters": budget["total_params"],
            "backbone_parameters": budget["backbone_params"],
            "head_parameters": budget["head_params"],
            "device": str(self.device),
            "context_limit_tokens": 32768,
            "policy": (
                "Tree-masked shared-state virtual-pair encoder with 8-head permutation-equivariant "
                "Set-Attention readout and 32k overlapping context windowing. Zero truncation, "
                "zero label filtering, exact option-order permutation equivariance."
            ),
        }

    @property
    def info(self) -> EngineInfo:
        return self._info

    def runtime(self) -> dict[str, Any]:
        info: dict[str, Any] = {"torch": torch.__version__, "device": str(self.device)}
        if self.device.type == "cuda":
            info.update(cuda=torch.version.cuda, gpu=torch.cuda.get_device_name())
        return info

    def synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize()

    @torch.inference_mode()
    def run(self, request: EngineRequest) -> EngineResponse:
        options_list = list(request.options)
        K = len(options_list)
        if K == 0:
            return EngineResponse(
                request_id=request.request_id,
                selected_indices=(),
                ranked_indices=(),
                option_scores=(),
                status="ok",
            )

        _, cal_logits = forward_single_request(
            model=self.model,
            tokenizer=self.tokenizer,
            state=request.state,
            instructions=request.instructions,
            options=options_list,
            question_type=request.question_type,
            device=self.device,
        )

        cal_f64 = cal_logits.to(torch.float64).cpu()
        ranked_tensor = torch.argsort(-cal_f64, stable=True)
        ranked_indices = tuple(int(i) for i in ranked_tensor.tolist())

        if request.question_type == "multi_label":
            selected_indices, ranked_indices, option_scores = select_multi_label_indices(
                calibrated_logits=cal_f64,
                select_k=request.select_k,
                threshold=self.multi_label_threshold,
                max_gap=self.multi_label_max_gap,
            )
        else:
            probs = torch.softmax(cal_f64, dim=-1).clamp(1e-7, 1.0 - 1e-7)
            probs = probs / probs.sum()
            selected_indices = (ranked_indices[0],)
            option_scores = tuple(float(p) for p in probs.tolist())

        return EngineResponse(
            request_id=request.request_id,
            selected_indices=selected_indices,
            ranked_indices=ranked_indices,
            option_scores=option_scores,
            status="ok",
        )

    @torch.inference_mode()
    def __call__(
        self,
        state: Any,
        questions: dict[str, dict[str, Any]],
        **inputs: Any,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        state_str = text(state) if state not in ("", None, {}, []) else " "
        answers: dict[str, Any] = {}
        raw: dict[str, Any] = {}

        for q_key, q in questions.items():
            q_type = q.get("type", "choice")
            instructions = text(q.get("instructions", ""))
            if q_type == "choice":
                criteria = q["criteria"]
                keys = list(criteria.keys())
                if not keys:
                    raise Unsupported("Empty criteria in choice question")
                opt_texts = [
                    text(criteria[k]) if criteria[k] is not None else str(k) for k in keys
                ]
                req = EngineRequest(
                    request_id=str(q_key),
                    state=state_str,
                    instructions=instructions,
                    options=opt_texts,
                    question_type="choice",
                )
                resp = self.run(req)
                probs_dict = {k: float(p) for k, p in zip(keys, resp.option_scores)}
                best_key = keys[resp.selected_indices[0]]
                answers[q_key] = {
                    "type": "choice",
                    "choice": best_key,
                    "probabilities": probs_dict,
                }
                raw[q_key] = {"ranked_keys": [keys[i] for i in resp.ranked_indices]}
            elif q_type == "noul":
                _, cal_logits = forward_single_request(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    state=state_str,
                    instructions=instructions,
                    options=["False", "True"],
                    question_type="noul",
                    device=self.device,
                )
                probs = torch.softmax(cal_logits.to(torch.float64).cpu(), dim=-1).clamp(
                    1e-7, 1.0 - 1e-7
                )
                probs = probs / probs.sum()
                noul_prob = float(probs[1].item())
                answers[q_key] = {"type": "noul", "noul": noul_prob}
                raw[q_key] = {"noul": noul_prob}
            else:
                raise Unsupported("Unsupported question type " + str(q_type))

        return {"model": self.model_id, "answers": answers}, raw
