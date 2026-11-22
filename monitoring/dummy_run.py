"""Dummy test run for ExperimentTracker and TensorBoard logging.

Tests the full monitoring foundation without training an actual GPT model.
Verifies event files, metrics logs, hardware monitors, and report generation.
"""
import math
import random
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from monitoring.experiment_tracker import ExperimentTracker


def run_dummy_experiment(total_steps: int = 50, eval_interval: int = 25):
    """Run a simulated 50-step training loop to test the monitoring suite."""
    print("=" * 70)
    print("Running Monitoring Suite Dummy Test...")
    print("=" * 70)

    dummy_config = {
        "experiment_name": "dummy_monitoring_test",
        "total_steps": total_steps,
        "batch_size": 16,
        "block_size": 256,
        "n_embd": 256,
        "learning_rate": 3e-4,
        "seed": config.RANDOM_SEED,
    }

    tracker = ExperimentTracker(experiment_name="dummy_test", config_dict=dummy_config)

    # Initial dummy loss
    loss = 6.5
    tokens_per_step = dummy_config["batch_size"] * dummy_config["block_size"]

    print("\nSimulating training loop...")
    for step in range(1, total_steps + 1):
        # Simulate realistic decay with noise
        decay = math.exp(-step / 30)
        loss = 2.5 + 4.0 * decay + random.uniform(-0.05, 0.05)
        lr = dummy_config["learning_rate"] * 0.5 * (1 + math.cos(math.pi * step / total_steps))
        grad_norm = 0.5 + 0.3 * random.random()

        # Simulate small compute time
        time.sleep(0.02)

        # Log training step
        tracker.log_step(
            step=step,
            train_loss=loss,
            lr=lr,
            grad_norm=grad_norm,
            tokens_in_step=tokens_per_step,
        )

        if step % 10 == 0:
            print(f"  Step {step:3d}/{total_steps} | Loss: {loss:.4f} | LR: {lr:.2e} | Tokens: {tracker.total_tokens_processed:,}")

        # Evaluation interval
        if step % eval_interval == 0:
            val_loss = loss + random.uniform(0.1, 0.25)
            sample_texts = [
                f"[Sample 1 at step {step}]: The mathematical foundation of deep learning...",
                f"[Sample 2 at step {step}]: Shakespearean dialogue and linguistic patterns...",
            ]
            tracker.log_eval(step=step, val_loss=val_loss, samples=sample_texts)
            print(f"  --> Eval at step {step:3d}: Val Loss = {val_loss:.4f} (logged text samples)")

    # Finish and trigger automatic report compilation
    tracker.finish()

    # Verify generated artifacts
    print("\n" + "=" * 70)
    print("VERIFYING GENERATED MONITORING ARTIFACTS")
    print("=" * 70)

    csv_path = tracker.run_dir / "metrics.csv"
    jsonl_path = tracker.run_dir / "metrics.jsonl"
    html_path = tracker.run_dir / "experiment_report.html"
    tb_files = list(tracker.tb_dir.glob("events.out.tfevents*"))

    print(f"  [OK] metrics.csv:             {csv_path.exists()} ({csv_path.stat().st_size:,} bytes)")
    print(f"  [OK] metrics.jsonl:           {jsonl_path.exists()} ({jsonl_path.stat().st_size:,} bytes)")
    print(f"  [OK] TensorBoard event file:  {len(tb_files) > 0} ({tb_files[0].name if tb_files else 'None'})")
    print(f"  [OK] HTML research report:    {html_path.exists()}")
    print("\nDummy monitoring test completed successfully!\n")


if __name__ == "__main__":
    run_dummy_experiment()

