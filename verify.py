"""验证教学示例的关键结论，不比较容易波动的完整浮点输出。"""

from __future__ import annotations

import argparse
import math
import tempfile
from pathlib import Path

from simple_models import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def verify(metrics: dict[str, object], output_dir: Path) -> None:
    assert metrics["device"] == "cuda:0"
    assert metrics["gpu"]
    assert metrics["versions"]["torch_cuda_runtime"]

    gradient = metrics["gradient_check"]
    assert math.isclose(gradient["manual_dw"], gradient["autograd_dw"], abs_tol=1e-6)
    assert math.isclose(gradient["manual_db"], gradient["autograd_db"], abs_tol=1e-6)

    linear = metrics["linear_regression"]
    assert linear["final_train_mse"] < linear["initial_train_mse"]
    assert abs(linear["learned_weight"] - 0.75) < 0.05
    assert abs(linear["learned_bias"] - 0.10) < 0.03
    assert linear["validation_mae"] < 0.06

    logistic = metrics["logistic_regression"]
    assert logistic["final_train_bce"] < logistic["initial_train_bce"]
    assert logistic["validation_accuracy"] >= 0.97
    assert abs(logistic["estimated_probability_threshold_x"] - 0.5) < 0.04

    nonlinear = metrics["nonlinear_classification"]
    assert nonlinear["mlp_final_train_bce"] < nonlinear["mlp_initial_train_bce"]
    assert nonlinear["mlp_validation_accuracy"] >= 0.90
    assert nonlinear["mlp_validation_accuracy"] >= nonlinear["linear_validation_accuracy"] + 0.25

    expected_files = {
        "01_linear_regression.png",
        "02_logistic_regression.png",
        "03_mlp_nonlinear_boundary.png",
        "04_mlp_forward_backward.png",
        "metrics.json",
    }
    assert expected_files == {path.name for path in output_dir.iterdir() if path.is_file()}


def main() -> None:
    args = parse_args()
    if args.output_dir is not None:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        metrics = run_experiment(args.output_dir, args.seed)
        verify(metrics, args.output_dir)
    else:
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            metrics = run_experiment(output_dir, args.seed)
            verify(metrics, output_dir)
    print("All checks passed.")


if __name__ == "__main__":
    main()
