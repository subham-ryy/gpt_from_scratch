"""Unified Experiment Tracker for GPT Pretraining.

Manages experiment reproducibility, configuration snapshots, TensorBoard events,
CSV/JSONL metric records, and non-fatal hardware monitoring.
"""
import datetime
import json
import time
from pathlib import Path

# Add project root to path
import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config

from monitoring.system_monitor import SystemMonitor
from monitoring.tensorboard_logger import TensorBoardLogger
from monitoring.metrics_logger import MetricsLogger


class ExperimentTracker:
    """Production-grade experiment manager and metric recorder."""

    def __init__(
        self,
        experiment_name: str = "gpt_pretrain",
        config_dict: dict | None = None,
        base_dir: Path = config.EXPERIMENTS_DIR,
        logs_dir: Path = config.LOGS_DIR,
    ):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.experiment_id = f"{timestamp}_{experiment_name}"
        self.experiment_name = experiment_name

        # Distinct output directories to guarantee runs are never overwritten
        self.run_dir = Path(base_dir) / self.experiment_id
        self.tb_dir = Path(logs_dir) / self.experiment_id

        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.tb_dir.mkdir(parents=True, exist_ok=True)

        # 1. Snapshot static hardware and environment
        self.hardware_info = SystemMonitor.get_hardware_info()
        self.hardware_info["timestamp"] = timestamp
        self.hardware_info["experiment_id"] = self.experiment_id
        self.hardware_info["experiment_name"] = experiment_name
        self.hardware_info["random_seed"] = config.RANDOM_SEED

        with open(self.run_dir / "system_info.json", "w", encoding="utf-8") as f:
            json.dump(self.hardware_info, f, indent=2)

        # 2. Snapshot configuration
        self.config_snapshot = config_dict or {
            k: v for k, v in vars(config).items()
            if not k.startswith("_") and isinstance(v, (str, int, float, bool, list, dict, type(None)))
        }
        self.config_snapshot["experiment_id"] = self.experiment_id

        with open(self.run_dir / "config_snapshot.json", "w", encoding="utf-8") as f:
            json.dump(self.config_snapshot, f, indent=2)

        # 3. Initialize loggers
        self.tb_logger = TensorBoardLogger(self.tb_dir)
        self.metrics_logger = MetricsLogger(self.run_dir)

        # 4. State for throughput computation
        self.start_time = time.time()
        self.last_step_time = time.time()
        self.total_tokens_processed = 0

        print(f"\n[ExperimentTracker] Initialized experiment: {self.experiment_id}")
        print(f"  Artifacts & Metrics: {self.run_dir.resolve()}")
        print(f"  TensorBoard Logs:    {self.tb_dir.resolve()}\n")

    def log_step(
        self,
        step: int,
        train_loss: float,
        lr: float,
        grad_norm: float | None = None,
        tokens_in_step: int | None = None,
        total_tokens: int | None = None,
        **extra_metrics,
    ) -> dict:
        """Log regular training iteration metrics and hardware resource usage."""
        now = time.time()
        step_elapsed = max(now - self.last_step_time, 1e-6)
        self.last_step_time = now

        steps_per_sec = 1.0 / step_elapsed

        if total_tokens is not None:
            self.total_tokens_processed = total_tokens
        elif tokens_in_step:
            self.total_tokens_processed += tokens_in_step

        if tokens_in_step:
            tokens_per_sec = tokens_in_step / step_elapsed
        else:
            tokens_per_sec = 0.0

        # Collect non-fatal hardware metrics
        hw = SystemMonitor.get_current_metrics()

        record = {
            "step": step,
            "train_loss": round(float(train_loss), 4),
            "val_loss": None,
            "learning_rate": lr,
            "grad_norm": round(float(grad_norm), 4) if grad_norm is not None else None,
            "tokens_in_step": tokens_in_step,
            "total_tokens": self.total_tokens_processed,
            "tokens_per_sec": round(tokens_per_sec, 1),
            "steps_per_sec": round(steps_per_sec, 2),
            **hw,
            **extra_metrics,
        }

        # Save to CSV and JSONL
        try:
            self.metrics_logger.log(record)
        except Exception as e:
            print(f"[Warning] Failed to write metric record: {e}")

        # Send to TensorBoard under clean categories
        tb_scalars = {
            "Loss/train": record["train_loss"],
            "Optimization/learning_rate": record["learning_rate"],
            "Throughput/tokens_per_sec": record["tokens_per_sec"],
            "Throughput/steps_per_sec": record["steps_per_sec"],
            "Throughput/total_tokens": record["total_tokens"],
            "System/gpu_allocated_mb": record["gpu_allocated_mb"],
            "System/gpu_reserved_mb": record["gpu_reserved_mb"],
            "System/gpu_peak_mb": record["gpu_peak_mb"],
            "System/cpu_ram_used_mb": record["cpu_ram_used_mb"],
            "System/cpu_ram_percent": record["cpu_ram_percent"],
        }
        if grad_norm is not None:
            tb_scalars["Optimization/grad_norm"] = record["grad_norm"]

        self.tb_logger.log_scalars(tb_scalars, step=step)
        return record

    def log_eval(
        self,
        step: int,
        val_loss: float,
        samples: list[str] | None = None,
        **extra_eval,
    ) -> None:
        """Log evaluation loss, metrics, and generated text samples."""
        record = {
            "step": step,
            "train_loss": None,
            "val_loss": round(float(val_loss), 4),
            "total_tokens": self.total_tokens_processed,
            **extra_eval,
        }
        self.metrics_logger.log(record)
        self.tb_logger.log_scalars({"Loss/val": val_loss}, step=step)

        # Log generated text samples
        if samples:
            sample_file = self.run_dir / "samples.txt"
            with open(sample_file, "a", encoding="utf-8") as f:
                f.write(f"\n--- STEP {step} (Val Loss: {val_loss:.4f}) ---\n")
                for idx, text in enumerate(samples):
                    f.write(f"[Sample {idx+1}]\n{text}\n\n")

            for idx, text in enumerate(samples):
                self.tb_logger.log_text(f"Generation/sample_{idx+1}", text, step=step)

    def finish(self) -> None:
        """Close loggers and generate experiment plots and HTML report."""
        self.tb_logger.flush()
        self.tb_logger.close()
        self.metrics_logger.close()

        total_duration = time.time() - self.start_time
        print(f"\n[ExperimentTracker] Experiment {self.experiment_id} completed in {total_duration:.1f}s.")
        print(f"  Total tokens processed: {self.total_tokens_processed:,}")

        # Automatically trigger HTML report and plot generation
        try:
            from analysis.generate_report import generate_experiment_report
            generate_experiment_report(self.run_dir)
        except Exception as e:
            print(f"[Warning] Automatic report generation failed: {e}")
