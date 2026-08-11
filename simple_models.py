"""用三个教学模型演示 PyTorch 的完整训练流程。

数据均为固定随机种子生成的教学模拟数据，不代表真实遥感机理或精度。
"""

from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn


@dataclass
class SplitData:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "没有检测到可用的 NVIDIA GPU。请先按上一篇文章检查驱动、"
            "虚拟环境和 PyTorch CUDA wheel。"
        )
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    return device


def move_to_device(data: SplitData, device: torch.device) -> SplitData:
    return SplitData(
        data.x_train.to(device),
        data.y_train.to(device),
        data.x_val.to(device),
        data.y_val.to(device),
    )


def split_tensors(
    x: torch.Tensor,
    y: torch.Tensor,
    seed: int,
    train_ratio: float = 0.8,
) -> SplitData:
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(x), generator=generator)
    split_at = int(len(x) * train_ratio)
    train_idx = indices[:split_at]
    val_idx = indices[split_at:]
    return SplitData(x[train_idx], y[train_idx], x[val_idx], y[val_idx])


def make_linear_data(seed: int, n_samples: int = 500) -> SplitData:
    """构造单波段像素值到连续参数代理值的线性教学数据。"""
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((n_samples, 1), generator=generator)
    noise = torch.randn((n_samples, 1), generator=generator) * 0.04
    y_continuous = 0.75 * x + 0.10 + noise
    return split_tensors(x, y_continuous, seed + 1)


def make_logistic_data(seed: int, n_samples: int = 500) -> SplitData:
    generator = torch.Generator().manual_seed(seed)
    x = torch.rand((n_samples, 1), generator=generator)
    y = (0.75 * x + 0.10 >= 0.475).float()
    return split_tensors(x, y, seed + 1)


def make_nonlinear_data(seed: int, n_samples: int = 1000) -> SplitData:
    """构造两个模拟光谱特征的 XOR 非线性分类数据。"""
    if n_samples % 4 != 0:
        raise ValueError("n_samples 必须是 4 的倍数，以保持四个象限等量。")
    generator = torch.Generator().manual_seed(seed)
    quarter = n_samples // 4

    def quadrant(x_high: bool, y_high: bool) -> torch.Tensor:
        values = torch.rand((quarter, 2), generator=generator) * 0.48
        if x_high:
            values[:, 0] += 0.52
        if y_high:
            values[:, 1] += 0.52
        return values

    x = torch.cat(
        (
            quadrant(False, False),
            quadrant(False, True),
            quadrant(True, False),
            quadrant(True, True),
        ),
        dim=0,
    )
    left_or_right = x[:, 0] >= 0.5
    low_or_high = x[:, 1] >= 0.5
    y = torch.logical_xor(left_or_right, low_or_high).float().unsqueeze(1)
    return split_tensors(x, y, seed + 1)


def train_model(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    loss_fn: nn.Module,
    optimizer: torch.optim.Optimizer,
    epochs: int,
) -> list[float]:
    """统一训练循环：前向、损失、清零、反向、更新。"""
    losses: list[float] = []
    model.train()
    for _ in range(epochs):
        predictions = model(x)
        loss = loss_fn(predictions, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach().item())
    return losses


def manual_gradient_demo(device: torch.device) -> dict[str, float]:
    """比较一个样本的手算梯度与 autograd 梯度。"""
    x = torch.tensor([[0.5]], device=device)
    target = torch.tensor([[1.5]], device=device)
    weight = torch.tensor([[1.0]], device=device, requires_grad=True)
    bias = torch.tensor([0.0], device=device, requires_grad=True)

    prediction = x @ weight + bias
    loss = torch.mean((prediction - target) ** 2)
    loss.backward()

    manual_dw = 2 * (0.5 * 1.0 + 0.0 - 1.5) * 0.5
    manual_db = 2 * (0.5 * 1.0 + 0.0 - 1.5)
    return {
        "prediction": prediction.detach().item(),
        "loss": loss.detach().item(),
        "manual_dw": manual_dw,
        "autograd_dw": float(weight.grad.item()),
        "manual_db": manual_db,
        "autograd_db": float(bias.grad.item()),
    }


def binary_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    prediction = (torch.sigmoid(logits) >= 0.5).float()
    return (prediction == target).float().mean().item()


def plot_linear(
    data: SplitData,
    model: nn.Module,
    losses: list[float],
    output_path: Path,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    x_line = torch.linspace(0, 1, 200, device=device).unsqueeze(1)
    with torch.no_grad():
        y_line = model(x_line)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(
        data.x_train.cpu().numpy(), data.y_train.cpu().numpy(), s=12, alpha=0.35, label="train"
    )
    axes[0].scatter(
        data.x_val.cpu().numpy(), data.y_val.cpu().numpy(), s=16, alpha=0.65, label="validation"
    )
    axes[0].plot(x_line.cpu().numpy(), y_line.cpu().numpy(), color="#C43D4D", linewidth=2.5, label="fitted line")
    axes[0].set(xlabel="normalized single-band value", ylabel="continuous target", title="Linear regression")
    axes[0].legend(frameon=False)

    axes[1].plot(losses, color="#315A87")
    axes[1].set(xlabel="epoch", ylabel="MSE loss", title="Training loss")
    axes[1].set_yscale("log")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_logistic(
    data: SplitData,
    model: nn.Module,
    losses: list[float],
    output_path: Path,
) -> None:
    model.eval()
    device = next(model.parameters()).device
    x_line = torch.linspace(0, 1, 300, device=device).unsqueeze(1)
    with torch.no_grad():
        probability = torch.sigmoid(model(x_line))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    axes[0].scatter(
        data.x_train.cpu().numpy(), data.y_train.cpu().numpy(), s=12, alpha=0.25, label="train label"
    )
    axes[0].scatter(
        data.x_val.cpu().numpy(), data.y_val.cpu().numpy(), s=16, alpha=0.6, label="validation label"
    )
    axes[0].plot(x_line.cpu().numpy(), probability.cpu().numpy(), color="#C43D4D", linewidth=2.5, label="sigmoid probability")
    axes[0].axhline(0.5, color="#777777", linestyle="--", linewidth=1)
    axes[0].set(xlabel="normalized single-band value", ylabel="class probability", title="Logistic regression")
    axes[0].legend(frameon=False)

    axes[1].plot(losses, color="#315A87")
    axes[1].set(xlabel="epoch", ylabel="BCE loss", title="Training loss")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def decision_surface(model: nn.Module, resolution: int = 160) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    device = next(model.parameters()).device
    axis = torch.linspace(0, 1, resolution, device=device)
    grid_x, grid_y = torch.meshgrid(axis, axis, indexing="xy")
    points = torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=1)
    model.eval()
    with torch.no_grad():
        probability = torch.sigmoid(model(points)).reshape(resolution, resolution)
    return grid_x.cpu().numpy(), grid_y.cpu().numpy(), probability.cpu().numpy()


def plot_nonlinear(
    data: SplitData,
    baseline: nn.Module,
    mlp: nn.Module,
    baseline_losses: list[float],
    mlp_losses: list[float],
    output_path: Path,
) -> None:
    grid_x, grid_y, baseline_probability = decision_surface(baseline)
    _, _, mlp_probability = decision_surface(mlp)
    x_val = data.x_val.cpu().numpy()
    y_val = data.y_val.squeeze(1).cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, probability, title in (
        (axes[0], baseline_probability, "Linear classifier"),
        (axes[1], mlp_probability, "MLP: 2-8-1 with ReLU"),
    ):
        ax.contourf(grid_x, grid_y, probability, levels=np.linspace(0, 1, 11), cmap="RdBu_r", alpha=0.75)
        ax.contour(grid_x, grid_y, probability, levels=[0.5], colors="black", linewidths=1.4)
        ax.scatter(x_val[:, 0], x_val[:, 1], c=y_val, cmap="bwr", s=10, edgecolors="none")
        ax.set(xlabel="feature 1", ylabel="feature 2", title=title, xlim=(0, 1), ylim=(0, 1))

    axes[2].plot(baseline_losses, label="linear classifier", color="#777777")
    axes[2].plot(mlp_losses, label="MLP", color="#315A87")
    axes[2].set(xlabel="epoch", ylabel="BCE loss", title="Training loss")
    axes[2].legend(frameon=False)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def trace_mlp(model: nn.Sequential, x: torch.Tensor, target: torch.Tensor) -> dict[str, object]:
    sample = x.clone().requires_grad_(True)
    model.zero_grad()
    z1 = model[0](sample)
    z1.retain_grad()
    a1 = model[1](z1)
    a1.retain_grad()
    logit = model[2](a1)
    logit.retain_grad()
    loss = nn.BCEWithLogitsLoss()(logit, target)
    loss.backward()

    return {
        "input": sample.detach().squeeze(0).cpu().tolist(),
        "target": int(target.item()),
        "hidden_pre_activation": z1.detach().squeeze(0).cpu().tolist(),
        "hidden_activation": a1.detach().squeeze(0).cpu().tolist(),
        "logit": logit.detach().item(),
        "probability": torch.sigmoid(logit.detach()).item(),
        "loss": loss.detach().item(),
        "input_gradient": sample.grad.detach().squeeze(0).cpu().tolist(),
        "hidden_pre_activation_gradient_norm": z1.grad.norm().item(),
        "hidden_activation_gradient_norm": a1.grad.norm().item(),
        "logit_gradient": logit.grad.item(),
        "first_layer_weight_gradient_norm": model[0].weight.grad.norm().item(),
        "output_layer_weight_gradient_norm": model[2].weight.grad.norm().item(),
    }


def short_vector(values: list[float], count: int = 4) -> str:
    shown = ", ".join(f"{value:.3f}" for value in values[:count])
    suffix = ", ..." if len(values) > count else ""
    return f"[{shown}{suffix}]"


def plot_forward_backward(trace: dict[str, object], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(14, 5), constrained_layout=True)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    hidden_pre_activation = trace["hidden_pre_activation"]
    hidden_activation = trace["hidden_activation"]
    negative_count = sum(value < 0 for value in hidden_pre_activation)
    boxes = [
        (0.3, "Input x", short_vector(trace["input"])),
        (
            3.0,
            "Linear 2→8",
            f"z={short_vector(hidden_pre_activation, 3)}\nlast={hidden_pre_activation[-1]:.3f}",
        ),
        (
            5.8,
            "ReLU",
            f"a={short_vector(hidden_activation, 3)}\n{negative_count} negatives → 0",
        ),
        (8.5, "Linear 8→1", f"logit={trace['logit']:.3f}"),
        (11.2, "BCE loss", f"loss={trace['loss']:.3f}\ntarget={trace['target']}"),
    ]
    for x_pos, title, value in boxes:
        patch = FancyBboxPatch(
            (x_pos, 2.7),
            2.1,
            1.2,
            boxstyle="round,pad=0.08",
            facecolor="#E8F0F7",
            edgecolor="#315A87",
            linewidth=1.5,
        )
        ax.add_patch(patch)
        ax.text(x_pos + 1.05, 3.45, title, ha="center", va="center", fontsize=11, weight="bold")
        ax.text(x_pos + 1.05, 3.05, value, ha="center", va="center", fontsize=9)

    for start in (2.4, 5.1, 7.9, 10.6):
        ax.annotate("", xy=(start + 0.55, 3.3), xytext=(start, 3.3), arrowprops={"arrowstyle": "->", "color": "#315A87", "lw": 1.8})

    ax.text(7.0, 4.55, "Forward pass: values flow from input to loss", ha="center", fontsize=13, weight="bold")
    ax.text(7.0, 0.45, "Backward pass: gradients flow from loss to parameters", ha="center", fontsize=13, weight="bold", color="#A33A3A")
    gradient_labels = [
        (1.35, "∂L/∂x=" + short_vector(trace["input_gradient"])),
        (4.05, f"||∂L/∂W₁||={trace['first_layer_weight_gradient_norm']:.3f}"),
        (6.85, f"||∂L/∂a||={trace['hidden_activation_gradient_norm']:.3f}"),
        (9.55, f"∂L/∂logit={trace['logit_gradient']:.3f}"),
        (12.25, f"probability={trace['probability']:.3f}"),
    ]
    for x_pos, label in gradient_labels:
        ax.text(x_pos, 1.25, label, ha="center", va="center", fontsize=9, color="#A33A3A")
    for start in (2.95, 5.65, 8.45, 11.15):
        ax.annotate("", xy=(start - 0.55, 1.75), xytext=(start, 1.75), arrowprops={"arrowstyle": "->", "color": "#A33A3A", "lw": 1.8})

    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def run_experiment(output_dir: Path, seed: int) -> dict[str, object]:
    device = require_cuda()
    set_seed(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    gradient_check = manual_gradient_demo(device)

    linear_data = move_to_device(make_linear_data(seed), device)
    linear_model = nn.Linear(1, 1).to(device)
    linear_loss = nn.MSELoss()
    linear_losses = train_model(
        linear_model,
        linear_data.x_train,
        linear_data.y_train,
        linear_loss,
        torch.optim.SGD(linear_model.parameters(), lr=0.2),
        epochs=300,
    )
    with torch.no_grad():
        linear_val_prediction = linear_model(linear_data.x_val)
        linear_val_mse = linear_loss(linear_val_prediction, linear_data.y_val).item()
        linear_val_mae = torch.mean(torch.abs(linear_val_prediction - linear_data.y_val)).item()
    plot_linear(linear_data, linear_model, linear_losses, output_dir / "01_linear_regression.png")

    logistic_data = move_to_device(make_logistic_data(seed), device)
    logistic_model = nn.Linear(1, 1).to(device)
    logistic_loss = nn.BCEWithLogitsLoss()
    logistic_losses = train_model(
        logistic_model,
        logistic_data.x_train,
        logistic_data.y_train,
        logistic_loss,
        torch.optim.SGD(logistic_model.parameters(), lr=0.5),
        epochs=800,
    )
    with torch.no_grad():
        logistic_val_logits = logistic_model(logistic_data.x_val)
        logistic_val_accuracy = binary_accuracy(logistic_val_logits, logistic_data.y_val)
        threshold_x = float(-logistic_model.bias.item() / logistic_model.weight.item())
    plot_logistic(logistic_data, logistic_model, logistic_losses, output_dir / "02_logistic_regression.png")

    nonlinear_data = move_to_device(make_nonlinear_data(seed), device)
    nonlinear_baseline = nn.Linear(2, 1).to(device)
    baseline_losses = train_model(
        nonlinear_baseline,
        nonlinear_data.x_train,
        nonlinear_data.y_train,
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD(nonlinear_baseline.parameters(), lr=0.3),
        epochs=1000,
    )
    mlp = nn.Sequential(nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 1)).to(device)
    mlp_losses = train_model(
        mlp,
        nonlinear_data.x_train,
        nonlinear_data.y_train,
        nn.BCEWithLogitsLoss(),
        torch.optim.SGD(mlp.parameters(), lr=0.3),
        epochs=3000,
    )
    with torch.no_grad():
        baseline_accuracy = binary_accuracy(nonlinear_baseline(nonlinear_data.x_val), nonlinear_data.y_val)
        mlp_accuracy = binary_accuracy(mlp(nonlinear_data.x_val), nonlinear_data.y_val)
    plot_nonlinear(
        nonlinear_data,
        nonlinear_baseline,
        mlp,
        baseline_losses,
        mlp_losses,
        output_dir / "03_mlp_nonlinear_boundary.png",
    )

    with torch.no_grad():
        validation_probability = torch.sigmoid(mlp(nonlinear_data.x_val)).squeeze(1)
        trace_index = int(torch.argmin(torch.abs(validation_probability - 0.5)))
    trace = trace_mlp(
        mlp,
        nonlinear_data.x_val[trace_index : trace_index + 1],
        nonlinear_data.y_val[trace_index : trace_index + 1],
    )
    plot_forward_backward(trace, output_dir / "04_mlp_forward_backward.png")

    metrics: dict[str, object] = {
        "seed": seed,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device),
        "versions": {
            "python": ".".join(map(str, __import__("sys").version_info[:3])),
            "torch": torch.__version__,
            "torch_cuda_runtime": torch.version.cuda,
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "gradient_check": gradient_check,
        "linear_regression": {
            "initial_train_mse": linear_losses[0],
            "final_train_mse": linear_losses[-1],
            "validation_mse": linear_val_mse,
            "validation_mae": linear_val_mae,
            "learned_weight": float(linear_model.weight.item()),
            "learned_bias": float(linear_model.bias.item()),
        },
        "logistic_regression": {
            "initial_train_bce": logistic_losses[0],
            "final_train_bce": logistic_losses[-1],
            "validation_accuracy": logistic_val_accuracy,
            "estimated_probability_threshold_x": threshold_x,
        },
        "nonlinear_classification": {
            "linear_validation_accuracy": baseline_accuracy,
            "mlp_initial_train_bce": mlp_losses[0],
            "mlp_final_train_bce": mlp_losses[-1],
            "mlp_validation_accuracy": mlp_accuracy,
        },
        "mlp_trace": trace,
    }
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = run_experiment(args.output_dir, args.seed)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
