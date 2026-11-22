"""AdamW optimizer with correct weight-decay parameter grouping.

Parameters that should NOT receive weight decay:
  - All bias terms (1-D)
  - LayerNorm scale (weight) and shift (bias)
  - Embeddings (they are regularised by dropout instead)

Parameters that DO receive weight decay:
  - Linear weight matrices (2-D)

This is the standard GPT-2 / GPT-3 recipe.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def configure_adamw(
    model: nn.Module,
    learning_rate: float,
    weight_decay: float,
    betas: tuple[float, float] = (0.9, 0.95),
    device_type: str = "cuda",
) -> torch.optim.AdamW:
    """Return an AdamW optimizer with two parameter groups.

    Group 1 (weight_decay > 0): 2-D weight matrices in Linear layers.
    Group 2 (weight_decay = 0): biases, LayerNorm params, embeddings.
    """
    decay_params    = []
    no_decay_params = []

    seen_ids: set[int] = set()

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if id(param) in seen_ids:
            continue
        seen_ids.add(id(param))

        if param.dim() >= 2:
            decay_params.append(param)
        else:
            no_decay_params.append(param)

    param_groups = [
        {"params": decay_params,    "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    decay_total    = sum(p.numel() for p in decay_params)
    no_decay_total = sum(p.numel() for p in no_decay_params)
    print(f"[Optimizer] Param groups:")
    print(f"  Decayed   : {len(decay_params):3d} tensors, {decay_total:>12,} params")
    print(f"  No-decay  : {len(no_decay_params):3d} tensors, {no_decay_total:>12,} params")
    print(f"  LR={learning_rate}, betas={betas}, wd={weight_decay}")

    try:
        optimizer = torch.optim.AdamW(
            param_groups, lr=learning_rate, betas=betas, fused=(device_type == "cuda"),
        )
        print("  Fused AdamW: enabled")
    except TypeError:
        optimizer = torch.optim.AdamW(param_groups, lr=learning_rate, betas=betas)
        print("  Fused AdamW: not available — using standard AdamW")

    return optimizer
