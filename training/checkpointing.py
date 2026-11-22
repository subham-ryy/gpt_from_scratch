"""Checkpoint saving and resumption for GPT pretraining.

A checkpoint contains everything needed to continue training with
bit-exact reproducibility:
  - model weights
  - optimizer state (m, v, step counts)
  - scheduler state (LR step counter)
  - optimizer step and micro-step counters
  - configuration snapshot
  - Python / NumPy / PyTorch RNG states
  - tokenizer and dataset metadata references
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def save_checkpoint(
    checkpoint_dir: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    opt_step: int,
    micro_step: int,
    train_loss: float,
    val_loss: float | None,
    config_snapshot: dict,
    tokenizer_config_path: Path,
    dataset_meta_paths: dict[str, Path],
    is_best: bool = False,
) -> Path:
    """Save a full training checkpoint.

    Parameters
    ----------
    checkpoint_dir       : Directory in which to write the checkpoint file.
    model                : The GPT model (state_dict is extracted).
    optimizer            : The AdamW optimizer (state_dict is extracted).
    scheduler            : CosineScheduler instance.
    opt_step             : Number of completed optimizer (outer) steps.
    micro_step           : Number of completed micro (inner) gradient steps.
    train_loss           : Most recent training loss.
    val_loss             : Most recent validation loss (None if not yet computed).
    config_snapshot      : Serialisable dict of hyper-parameters.
    tokenizer_config_path: Path to tokenizer_config.json — stored as a reference.
    dataset_meta_paths   : Dict mapping split name → .json metadata path.
    is_best              : If True, also write a ``best_model.pt`` symlink-free copy.

    Returns
    -------
    Path of the saved checkpoint file.
    """
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        # ── Training counters ────────────────────────────────────────────────
        "opt_step"  : opt_step,
        "micro_step": micro_step,
        "train_loss": train_loss,
        "val_loss"  : val_loss,

        # ── Model and optimiser state ────────────────────────────────────────
        "model_state"    : model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),

        # ── Configuration ────────────────────────────────────────────────────
        "config_snapshot": config_snapshot,

        # ── RNG states (Python, NumPy, PyTorch CPU, PyTorch CUDA) ───────────
        "rng_python": random.getstate(),
        "rng_numpy" : np.random.get_state(),
        "rng_torch" : torch.get_rng_state(),
        "rng_cuda"  : torch.cuda.get_rng_state() if torch.cuda.is_available() else None,

        # ── Artifact references (paths, not full files) ───────────────────────
        "tokenizer_config_path": str(tokenizer_config_path),
        "dataset_meta_paths"   : {k: str(v) for k, v in dataset_meta_paths.items()},
    }

    # Save the latest checkpoint to last.pt (overwriting previous to bound disk usage)
    last_file = checkpoint_dir / "last.pt"
    tmp_last = checkpoint_dir / "last.pt.tmp"
    torch.save(ckpt, str(tmp_last))
    if tmp_last.exists():
        if last_file.exists():
            last_file.unlink()
        tmp_last.rename(last_file)

    if is_best:
        best_file = checkpoint_dir / "best.pt"
        tmp_best = checkpoint_dir / "best.pt.tmp"
        torch.save(ckpt, str(tmp_best))
        if tmp_best.exists():
            if best_file.exists():
                best_file.unlink()
            tmp_best.rename(best_file)
        print(f"[Checkpoint] New best model saved → {best_file.name}")

    size_mb = last_file.stat().st_size / 1e6
    print(f"[Checkpoint] Saved step {opt_step:,} → {last_file.name}  ({size_mb:.1f} MB)")
    return last_file


def load_checkpoint(
    checkpoint_path: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    device: torch.device,
) -> dict[str, Any]:
    """Resume training from a checkpoint.

    Restores model weights, optimizer state, scheduler step counter, and all
    RNG states so that training continues deterministically.

    Accepts path to a file (last.pt, best.pt) or a directory containing them.
    """
    checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.is_dir():
        for candidate in ["last.pt", "best.pt", "best_model.pt"]:
            p = checkpoint_path / candidate
            if p.exists():
                checkpoint_path = p
                break

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"[Checkpoint] Loading from {checkpoint_path} …")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu", weights_only=False)

    # Restore model (strict=True ensures architecture has not changed)
    model.load_state_dict(ckpt["model_state"])
    model.to(device)

    # Restore optimizer
    optimizer.load_state_dict(ckpt["optimizer_state"])
    # Ensure optimizer state tensors reside on target device
    if str(device) != "cpu":
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    # Restore scheduler (also re-applies the correct LR to the optimizer)
    scheduler.load_state_dict(ckpt["scheduler_state"])

    # Restore RNG states for reproducibility
    random.setstate(ckpt["rng_python"])
    np.random.set_state(ckpt["rng_numpy"])
    if ckpt.get("rng_torch") is not None:
        torch.set_rng_state(ckpt["rng_torch"].cpu().to(torch.uint8))
    if ckpt.get("rng_cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state(ckpt["rng_cuda"].cpu().to(torch.uint8))

    meta = {
        "opt_step"  : ckpt["opt_step"],
        "micro_step": ckpt["micro_step"],
        "train_loss": ckpt["train_loss"],
        "val_loss"  : ckpt["val_loss"],
        "config_snapshot"      : ckpt.get("config_snapshot", {}),
        "tokenizer_config_path": ckpt.get("tokenizer_config_path"),
        "dataset_meta_paths"   : ckpt.get("dataset_meta_paths", {}),
    }

    print(f"[Checkpoint] Resumed from opt_step={meta['opt_step']:,}, "
          f"micro_step={meta['micro_step']:,}, "
          f"train_loss={meta['train_loss']:.4f}")
    return meta
