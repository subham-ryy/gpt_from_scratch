"""Generates publication-quality metric plots from an experiment's metrics.csv."""
import argparse
import csv
from pathlib import Path
import matplotlib.pyplot as plt


def plot_experiment_metrics(run_dir: Path) -> list[Path]:
    """Read metrics.csv and generate loss, learning rate, gradient norm, memory, and throughput plots."""
    run_dir = Path(run_dir)
    csv_path = run_dir / "metrics.csv"
    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not csv_path.exists():
        print(f"[Warning] No metrics.csv found at {csv_path}")
        return []

    # Read records
    train_steps, train_losses = [], []
    val_steps, val_losses = [], []
    lr_steps, lrs = [], []
    grad_steps, grad_norms = [], []
    tok_steps, tok_rates = [], []
    step_rates = []
    gpu_alloc, gpu_res = [], []

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            step = int(r["step"])
            # Train loss
            if r.get("train_loss") and r["train_loss"] != "":
                train_steps.append(step)
                train_losses.append(float(r["train_loss"]))
            # Val loss
            if r.get("val_loss") and r["val_loss"] != "":
                val_steps.append(step)
                val_losses.append(float(r["val_loss"]))
            # LR
            if r.get("learning_rate") and r["learning_rate"] != "":
                lr_steps.append(step)
                lrs.append(float(r["learning_rate"]))
            # Grad norm
            if r.get("grad_norm") and r["grad_norm"] != "":
                grad_steps.append(step)
                grad_norms.append(float(r["grad_norm"]))
            # Throughput
            if r.get("tokens_per_sec") and r["tokens_per_sec"] != "":
                tok_steps.append(step)
                tok_rates.append(float(r["tokens_per_sec"]))
                step_rates.append(float(r.get("steps_per_sec", 0.0)))
            # GPU memory
            if r.get("gpu_allocated_mb") and r["gpu_allocated_mb"] != "":
                gpu_alloc.append(float(r["gpu_allocated_mb"]))
                gpu_res.append(float(r.get("gpu_reserved_mb", 0.0)))

    generated_plots = []

    # 1. Loss Curve
    if train_losses or val_losses:
        plt.figure(figsize=(8, 4.5))
        if train_losses:
            plt.plot(train_steps, train_losses, label="Train Loss", color="#2563eb", linewidth=1.8)
        if val_losses:
            plt.plot(val_steps, val_losses, label="Validation Loss", color="#dc2626", linewidth=2.0, marker="o")
        plt.title("Training & Validation Loss", fontsize=12, fontweight="bold")
        plt.xlabel("Step")
        plt.ylabel("Cross-Entropy Loss")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        p = plots_dir / "loss_curve.png"
        plt.savefig(p, dpi=150)
        plt.close()
        generated_plots.append(p)

    # 2. Learning Rate & Gradient Norm
    if lrs or grad_norms:
        fig, ax1 = plt.subplots(figsize=(8, 4.5))
        if lrs:
            color = "#059669"
            ax1.set_xlabel("Step")
            ax1.set_ylabel("Learning Rate", color=color)
            ax1.plot(lr_steps, lrs, color=color, linewidth=1.8, label="LR")
            ax1.tick_params(axis="y", labelcolor=color)

        if grad_norms:
            ax2 = ax1.twinx()
            color = "#d97706"
            ax2.set_ylabel("Gradient Norm", color=color)
            ax2.plot(grad_steps, grad_norms, color=color, linewidth=1.5, linestyle="--", label="Grad Norm")
            ax2.tick_params(axis="y", labelcolor=color)

        plt.title("Optimization: Learning Rate & Gradient Norm", fontsize=12, fontweight="bold")
        fig.tight_layout()
        p = plots_dir / "optimization_curve.png"
        plt.savefig(p, dpi=150)
        plt.close()
        generated_plots.append(p)

    # 3. Throughput
    if tok_rates:
        plt.figure(figsize=(8, 4.5))
        plt.plot(tok_steps, tok_rates, color="#7c3aed", linewidth=1.8, label="Tokens/sec")
        plt.title("Training Throughput (Tokens per Second)", fontsize=12, fontweight="bold")
        plt.xlabel("Step")
        plt.ylabel("Tokens / sec")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        p = plots_dir / "throughput.png"
        plt.savefig(p, dpi=150)
        plt.close()
        generated_plots.append(p)

    # 4. Memory Usage
    if any(gpu_alloc):
        plt.figure(figsize=(8, 4.5))
        plt.plot(train_steps[:len(gpu_alloc)], gpu_alloc, color="#0284c7", linewidth=1.8, label="GPU Allocated (MB)")
        plt.plot(train_steps[:len(gpu_res)], gpu_res, color="#94a3b8", linestyle="--", linewidth=1.5, label="GPU Reserved (MB)")
        plt.title("GPU Memory Utilization", fontsize=12, fontweight="bold")
        plt.xlabel("Step")
        plt.ylabel("Memory (MB)")
        plt.grid(True, linestyle="--", alpha=0.5)
        plt.legend()
        plt.tight_layout()
        p = plots_dir / "memory_usage.png"
        plt.savefig(p, dpi=150)
        plt.close()
        generated_plots.append(p)

    return generated_plots


def main():
    parser = argparse.ArgumentParser(description="Plot metrics for an experiment.")
    parser.add_argument("run_dir", type=str, help="Path to experiment directory.")
    args = parser.parse_args()
    plots = plot_experiment_metrics(Path(args.run_dir))
    print(f"Generated {len(plots)} plots in {Path(args.run_dir) / 'plots'}")


if __name__ == "__main__":
    main()

