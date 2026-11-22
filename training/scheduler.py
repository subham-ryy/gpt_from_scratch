"""Linear warmup followed by cosine learning-rate decay.

Schedule:
  step  in [0, warmup_steps)         : LR increases linearly from 0 → peak_lr
  step  in [warmup_steps, max_steps) : LR decays via cosine from peak_lr → min_lr
  step >= max_steps                  : LR stays at min_lr

Usage
-----
    scheduler = CosineScheduler(optimizer, warmup_steps=100, max_steps=5000,
                                peak_lr=3e-4, min_lr=3e-5)
    for step in range(max_steps):
        scheduler.step()          # call AFTER optimizer.step()
        current_lr = scheduler.get_lr()
"""
from __future__ import annotations

import math


class CosineScheduler:
    """Manual cosine-decay scheduler (no PyTorch dependency on the schedule itself).

    Manipulates optimizer param-group LR directly, so it works with any PyTorch
    optimizer and does not require a ``torch.optim.lr_scheduler`` wrapper.
    """

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        max_steps: int,
        peak_lr: float,
        min_lr: float,
    ) -> None:
        assert warmup_steps >= 0
        assert max_steps > warmup_steps
        assert peak_lr >= min_lr > 0

        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.max_steps    = max_steps
        self.peak_lr      = peak_lr
        self.min_lr       = min_lr
        self._step        = 0

        # Set initial LR to 0 (warmup starts from zero)
        self._set_lr(0.0 if warmup_steps > 0 else peak_lr)

    # ── public API ────────────────────────────────────────────────────────────

    def step(self) -> None:
        """Advance the schedule by one step and update the optimizer LR."""
        self._step += 1
        self._set_lr(self.get_lr(self._step))

    def get_lr(self, step: int | None = None) -> float:
        """Return the learning rate at the given step (defaults to current step)."""
        if step is None:
            step = self._step

        if step < self.warmup_steps:
            # Linear warmup: 0 → peak_lr
            return self.peak_lr * step / max(self.warmup_steps, 1)

        if step >= self.max_steps:
            # After max_steps: hold at min_lr
            return self.min_lr

        # Cosine decay from warmup_steps to max_steps
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.min_lr + (self.peak_lr - self.min_lr) * cosine_decay

    @property
    def current_step(self) -> int:
        return self._step

    # ── private ───────────────────────────────────────────────────────────────

    def _set_lr(self, lr: float) -> None:
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def state_dict(self) -> dict:
        return {
            "step"        : self._step,
            "warmup_steps": self.warmup_steps,
            "max_steps"   : self.max_steps,
            "peak_lr"     : self.peak_lr,
            "min_lr"      : self.min_lr,
        }

    def load_state_dict(self, state: dict) -> None:
        self._step        = state["step"]
        self.warmup_steps = state["warmup_steps"]
        self.max_steps    = state["max_steps"]
        self.peak_lr      = state["peak_lr"]
        self.min_lr       = state["min_lr"]
        self._set_lr(self.get_lr(self._step))
