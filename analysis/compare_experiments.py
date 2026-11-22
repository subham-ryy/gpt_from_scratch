"""Compares multiple experiments and generates comparative visualizations."""
import argparse
import csv
import json
from pathlib import Path
import matplotlib.pyplot as plt

# Add project root to sys.path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def compare_all_experiments(experiments_dir: Path = config.EXPERIMENTS_DIR) -> list[dict]:
    """Scan experiments directory and compare metrics across runs."""
    experiments_dir = Path(experiments_dir)
    results = []

    for exp_dir in sorted(experiments_dir.iterdir()):
        if not exp_dir.is_dir():
            continue
        csv_path = exp_dir / "metrics.csv"
        cfg_path = exp_dir / "config_snapshot.json"

        if not csv_path.exists():
            continue

        cfg = {}
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                cfg = json.load(f)

        records = []
        with open(csv_path, "r", encoding="utf-8") as f:
            records = list(csv.DictReader(f))

        if not records:
            continue

        val_records = [r for r in records if r.get("val_loss") and r["val_loss"] != ""]
        best_val = min([float(r["val_loss"]) for r in val_records]) if val_records else None
        last_train = float(records[-1]["train_loss"]) if records[-1].get("train_loss") else None
        total_steps = int(records[-1]["step"]) if records[-1].get("step") else len(records)
        total_tokens = 0
        for r in reversed(records):
            if r.get("total_tokens") and r["total_tokens"] != "":
                try:
                    total_tokens = int(float(r["total_tokens"]))
                    break
                except (ValueError, TypeError):
                    pass

        results.append({
            "experiment_id": exp_dir.name,
            "best_val_loss": best_val,
            "final_train_loss": last_train,
            "total_steps": total_steps,
            "total_tokens": total_tokens,
            "lr": cfg.get("LEARNING_RATE", "N/A"),
            "block_size": cfg.get("BLOCK_SIZE", "N/A"),
        })

    if not results:
        print(f"No experiment metrics found in {experiments_dir.resolve()}")
        return []

    # Sort: runs with val_loss first (lowest first), then train loss
    results_sorted = sorted(results, key=lambda r: (r["best_val_loss"] is None, r["best_val_loss"] or 999.0))

    # Print comparison table
    print("\n" + "=" * 95)
    print("EXPERIMENT COMPARISON SUMMARY")
    print("=" * 95)
    header = f"{'EXPERIMENT ID':36} {'BEST VAL':>10} {'FINAL TRAIN':>12} {'STEPS':>8} {'TOKENS':>12} {'LR':>10}"
    print(header)
    print("-" * len(header))
    for r in results_sorted:
        b_val = f"{r['best_val_loss']:.4f}" if r['best_val_loss'] is not None else "N/A"
        f_tr = f"{r['final_train_loss']:.4f}" if r['final_train_loss'] is not None else "N/A"
        print(f"{r['experiment_id'][:36]:36} {b_val:>10} {f_tr:>12} {r['total_steps']:8d} {r['total_tokens']:12,d} {str(r['lr']):>10}")
    print("-" * len(header) + "\n")

    # Generate comparison chart if 2+ experiments exist
    if len(results) >= 2:
        names = [r["experiment_id"][-18:] for r in results_sorted]
        val_losses = [r["best_val_loss"] or 0.0 for r in results_sorted]

        plt.figure(figsize=(10, 5))
        bars = plt.bar(names, val_losses, color="#2563eb", edgecolor="white", width=0.5)
        plt.title("Cross-Experiment Best Validation Loss", fontsize=12, fontweight="bold")
        plt.ylabel("Validation Loss")
        plt.xticks(rotation=25, ha="right")
        for bar in bars:
            yval = bar.get_height()
            if yval > 0:
                plt.text(bar.get_x() + bar.get_width() / 2, yval + 0.02, f"{yval:.3f}", ha="center", va="bottom", fontsize=9)
        plt.tight_layout()
        chart_path = experiments_dir / "experiment_comparison.png"
        plt.savefig(chart_path, dpi=150)
        plt.close()
        print(f"Saved comparison chart: {chart_path.resolve()}\n")

    return results_sorted


def main():
    parser = argparse.ArgumentParser(description="Compare saved experiments.")
    parser.add_argument("--experiments-dir", type=str, default=str(config.EXPERIMENTS_DIR), help="Experiments root directory.")
    args = parser.parse_args()
    compare_all_experiments(Path(args.experiments_dir))


if __name__ == "__main__":
    main()
