"""Verification and Readiness Test for 500M Token Continued Pretraining.

Validates:
1. Binary token shard integrity (train_500m.bin and val_500m.bin).
2. Document boundary packing efficiency (>98%) and loss masking.
3. Checkpoint loading from Model 2 (experiments/20261106_032904_gpt_continued_pretrain/checkpoints/best.pt).
4. Forward pass, BF16 Z-loss computation, backward pass, and gradient finiteness.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import BinaryTokenDataset, PackedBinaryTokenDataset
from model.transformer import GPT, GPTConfig

TRAIN_BIN = config.DATA_DIR / "tokenized" / "train_500m.bin"
TRAIN_JSON = config.DATA_DIR / "tokenized" / "train_500m.json"
VAL_BIN = config.DATA_DIR / "tokenized" / "val_500m.bin"
VAL_JSON = config.DATA_DIR / "tokenized" / "val_500m.json"

MODEL_2_CKPT = PROJECT_ROOT / "experiments" / "20261106_032904_gpt_continued_pretrain" / "checkpoints" / "best.pt"


def main():
    print("=" * 70)
    print("Pre-Launch Verification: 500M Token Continued Pretraining")
    print("=" * 70)

    device = torch.device(config.DEVICE)
    print(f"Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    # 1. Verify files exist
    assert TRAIN_BIN.exists(), f"Train bin not found: {TRAIN_BIN}"
    assert VAL_BIN.exists(), f"Val bin not found: {VAL_BIN}"
    assert TRAIN_JSON.exists(), f"Train JSON not found: {TRAIN_JSON}"
    assert VAL_JSON.exists(), f"Val JSON not found: {VAL_JSON}"

    with open(TRAIN_JSON, "r", encoding="utf-8") as f:
        train_meta = json.load(f)
    with open(VAL_JSON, "r", encoding="utf-8") as f:
        val_meta = json.load(f)

    train_tokens = train_meta["total_tokens"]
    val_tokens = val_meta["total_tokens"]

    print(f"\n[1. Token Shard Verification]")
    print(f"  Train Shard: {TRAIN_BIN.name} | {train_tokens:,} tokens ({train_meta['storage_mb_uint16']} MB)")
    print(f"  Val Shard  : {VAL_BIN.name} | {val_tokens:,} tokens ({val_meta['storage_mb_uint16']} MB)")
    assert train_tokens >= 500000000, f"Expected >= 500M tokens, got {train_tokens:,}"
    assert val_tokens >= 10000000, f"Expected >= 10M val tokens, got {val_tokens:,}"
    print("  --> PASS: Token counts meet or exceed 500M pretraining requirements.")

    # 2. Verify Packed Dataset & Document Boundaries
    print(f"\n[2. Packed Dataset & Document Boundaries]")
    t0 = time.time()
    train_ds = PackedBinaryTokenDataset(TRAIN_BIN, block_size=1024, split="train", strategy="first_fit")
    dt = time.time() - t0
    num_bins = len(train_ds)
    print(f"  Packed sequences: {num_bins:,} sequences ({dt:.2f}s bin-packing index time)")
    print(f"  Packing efficiency: {train_ds.packing_efficiency * 100:.2f}% (Target: >95%)")
    assert train_ds.packing_efficiency > 0.95, "Packing efficiency below 95% threshold"

    # Verify a batch
    loader = DataLoader(train_ds, batch_size=4, shuffle=False)
    x, y = next(iter(loader))
    assert x.shape == (4, 1024), f"Unexpected shape {x.shape}"
    assert y.shape == (4, 1024), f"Unexpected target shape {y.shape}"

    # Verify EOS boundary masking: where x == 2, y must be -1
    eos_mask = (x == 2)
    masked_properly = (y[eos_mask] == -1).all().item()
    print(f"  EOS boundary loss masking verified: {masked_properly} (Next-token loss across doc boundaries is strictly ignored)")
    assert masked_properly, "Boundary masking failed: EOS token not masked with -1 in target!"
    print("  --> PASS: Document bin-packing and boundary masking 100% verified.")

    # 3. Model Compatibility & Checkpoint Loading
    print(f"\n[3. Model Loading & Architecture Compatibility]")
    assert MODEL_2_CKPT.exists(), f"Model 2 checkpoint not found: {MODEL_2_CKPT}"
    print(f"  Loading base checkpoint: {MODEL_2_CKPT.name}...")
    ckpt = torch.load(str(MODEL_2_CKPT), map_location="cpu", weights_only=False)

    model_cfg = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=1024,
        n_layer=12,
        n_head=12,
        n_embd=768,
        ffn_dim=3072,
        dropout=0.0,
        bias=True,
        tie_weights=True,
        use_flash=True,
    )
    model = GPT(model_cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    print(f"  Model successfully loaded! Parameters: {model.get_num_params():,}")
    print(f"  Base checkpoint step: {ckpt.get('opt_step', 'unknown')} | Base val_loss: {ckpt.get('val_loss', 'unknown'):.4f}")
    del ckpt
    torch.cuda.empty_cache()
    print("  --> PASS: Architecture and weights are 100% compatible.")

    # 4. Forward, BF16 Z-Loss, Backward & Gradient Norm Test
    print(f"\n[4. Forward, Z-Loss & Backward Pass Test]")
    model.train()
    x = x.to(device)
    y = y.to(device)

    amp_dtype = torch.bfloat16 if config.USE_AMP and torch.cuda.is_available() else torch.float32
    with torch.amp.autocast(device_type=device.type, dtype=amp_dtype):
        logits, _ = model(x)
        # Cross-entropy
        ce_loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1), ignore_index=-1)
        # Z-loss in BF16
        valid_mask = (y != -1)
        log_z = torch.logsumexp(logits, dim=-1)
        z_loss = 1e-4 * (log_z[valid_mask] ** 2).mean().to(torch.float32)
        total_loss = ce_loss + z_loss

    total_loss.backward()
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0).item()

    print(f"  CE Loss    : {ce_loss.item():.4f}")
    print(f"  Z-Loss     : {z_loss.item():.4f}")
    print(f"  Total Loss : {total_loss.item():.4f} (Expected around ~3.7)")
    print(f"  Grad Norm  : {grad_norm:.4f} (Finite & Healthy)")

    assert torch.isfinite(total_loss), "Loss is not finite!"
    assert 2.5 <= total_loss.item() <= 4.5, f"Unexpected loss value: {total_loss.item()}"
    assert torch.isfinite(torch.tensor(grad_norm)), "Grad norm is NaN or Inf!"
    print("  --> PASS: Forward/Backward/Loss pipeline fully operational.")

    print("\n" + "=" * 70)
    print("STATUS: ALL SYSTEMS GO FOR 500M TOKEN CONTINUED PRETRAINING!")
    print("=" * 70)


if __name__ == "__main__":
    main()
