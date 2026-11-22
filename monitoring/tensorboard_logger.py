"""TensorBoard logging module.

Wraps PyTorch SummaryWriter to record scalar training metrics,
system memory usage, throughput, and generated text samples.
"""
from pathlib import Path
import warnings

try:
    from torch.utils.tensorboard import SummaryWriter
    _HAS_TENSORBOARD = True
except ImportError:
    _HAS_TENSORBOARD = False


class TensorBoardLogger:
    """Non-fatal TensorBoard event logger for GPT training."""

    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.writer = None
        self.enabled = _HAS_TENSORBOARD

        if self.enabled:
            try:
                self.log_dir.mkdir(parents=True, exist_ok=True)
                self.writer = SummaryWriter(log_dir=str(self.log_dir))
            except Exception as e:
                warnings.warn(f"Failed to initialize TensorBoard SummaryWriter: {e}")
                self.enabled = False
        else:
            warnings.warn("tensorboard package not found. TensorBoard logging is disabled.")

    def log_scalars(self, tag_dict: dict, step: int) -> None:
        """Log a dictionary of scalar values under appropriate namespaces."""
        if not self.enabled or self.writer is None:
            return

        try:
            for tag, value in tag_dict.items():
                if value is not None and isinstance(value, (int, float)):
                    self.writer.add_scalar(tag, value, global_step=step)
        except Exception as e:
            warnings.warn(f"TensorBoard log_scalars error at step {step}: {e}")

    def log_text(self, tag: str, text: str, step: int) -> None:
        """Log sample generated text or prompts."""
        if not self.enabled or self.writer is None:
            return

        try:
            self.writer.add_text(tag, text, global_step=step)
        except Exception as e:
            warnings.warn(f"TensorBoard log_text error at step {step}: {e}")

    def flush(self) -> None:
        if self.enabled and self.writer:
            try:
                self.writer.flush()
            except Exception:
                pass

    def close(self) -> None:
        if self.enabled and self.writer:
            try:
                self.writer.close()
            except Exception:
                pass

