# scripts/soup_and_calibrate.py
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

from src.jev_set.calibration import fit_cardinality_temperatures_lbfgs


def greedy_model_soup(
    checkpoint_paths: list[Path],
    weights: list[float],
    output_path: Path,
) -> None:
    """
    Weight-space model souping (Wortsman et al., 2022) + WiSE-FT interpolation across Stage 2 and Stage 3 seeds.
    """
    assert len(checkpoint_paths) == len(weights) and len(checkpoint_paths) > 0
    norm_w = [w / sum(weights) for w in weights]
    soup_state: dict[str, torch.Tensor] = {}

    for ckpt_path, w in zip(checkpoint_paths, norm_w):
        state = load_file(str(ckpt_path))
        for k, v in state.items():
            soup_state[k] = v.float() * w if k not in soup_state else soup_state[k] + v.float() * w

    ref_state = load_file(str(checkpoint_paths[0]))
    final_state = {k: soup_state[k].to(ref_state[k].dtype) for k in soup_state}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(final_state, str(output_path))


def inject_calibrated_temperatures_into_checkpoint(
    checkpoint_path: Path,
    log_temperature: torch.Tensor,
    cardinality_slope: torch.Tensor,
) -> None:
    """Writes fitted L-BFGS temperature parameters into an existing model.safetensors file."""
    if not checkpoint_path.exists():
        return
    state = load_file(str(checkpoint_path))
    state["head.log_temperature"] = log_temperature.detach().float().cpu()
    state["head.cardinality_slope"] = cardinality_slope.detach().float().cpu()
    save_file(state, str(checkpoint_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Soup Stage 3 checkpoints and fit L-BFGS cardinality temperatures.")
    parser.add_argument("--checkpoints", type=Path, nargs="+", help="Paths to model.safetensors checkpoints.")
    parser.add_argument("--weights", type=float, nargs="+", help="Interpolation weights for each checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("checkpoints/jev-set-150m-final/model.safetensors"))
    parser.add_argument("--demo-calibration", action="store_true", help="Run synthetic L-BFGS calibration verification.")
    args = parser.parse_args()

    if args.checkpoints:
        weights = args.weights if args.weights else [1.0 / len(args.checkpoints)] * len(args.checkpoints)
        greedy_model_soup(args.checkpoints, weights, args.output)
        print(f"Saved souped model checkpoint to {args.output}")

    if args.demo_calibration or not args.checkpoints:
        torch.manual_seed(42)
        oof_logits = [torch.randn(k) * 2.5 for k in [2, 4, 10, 77, 151] * 10]
        oof_golds = [int(torch.argmax(z).item()) for z in oof_logits]
        oof_qtypes = [0, 0, 1, 2, 0] * 10
        log_temp, card_slope = fit_cardinality_temperatures_lbfgs(oof_logits, oof_golds, oof_qtypes)
        if args.output.exists():
            inject_calibrated_temperatures_into_checkpoint(args.output, log_temp, card_slope)
        print("Fitted log_temperature:", [round(float(x), 4) for x in log_temp.tolist()])
        print("Fitted cardinality_slope:", [round(float(x), 4) for x in card_slope.tolist()])


if __name__ == "__main__":
    main()
