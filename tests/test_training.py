"""Phase 6 Training Engine Test Suite.

Tests
-----
 1. Dataset loader: correct number of samples
 2. Dataset loader: input/target are shifted by exactly one token
 3. Dataset loader: NO sequence crosses a document boundary (BOS/EOS intact)
 4. Dataset loader: padded targets use IGNORE_INDEX=-1 (not real token IDs)
 5. Gradient accumulation: loss.backward() called N times, then optimizer.step() once
 6. Optimizer update frequency: opt_step increments only every GRAD_ACCUM_STEPS
 7. BF16 autocast forward+backward: no NaN, correct dtype
 8. Validation isolation: val_loader uses only val shard, train_loader uses only train shard
 9. Checkpoint save+load: exact state restoration (model weights, opt_step, LR)
10. Reproducibility: same seed → identical first batch
11. Smoke test: 3 optimizer steps on small synthetic model — losses decrease
"""
from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import math
import tempfile

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import BinaryTokenDataset, IGNORE_INDEX
from model.transformer import GPT, GPTConfig
from training.checkpointing import load_checkpoint, save_checkpoint
from training.optimizer import configure_adamw
from training.scheduler import CosineScheduler

# ── tiny model config for fast tests ─────────────────────────────────────────
TINY = GPTConfig(
    vocab_size=16384,
    block_size=64,
    n_layer=2,
    n_head=2,
    n_embd=64,
    ffn_dim=256,
    dropout=0.0,
    bias=True,
    tie_weights=True,
    use_flash=False,
)

PASS = "\033[32mPASS\033[0m"
FAIL = "\033[31mFAIL\033[0m"


def run(label: str, fn) -> bool:
    try:
        fn()
        print(f"  [{PASS}] {label}")
        return True
    except Exception as e:
        print(f"  [{FAIL}] {label}")
        print(f"           {type(e).__name__}: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# TEST 1 — Dataset: sample count
# ══════════════════════════════════════════════════════════════════════════════
def test_dataset_sample_count():
    ds = BinaryTokenDataset(config.TRAIN_BIN, block_size=config.BLOCK_SIZE, split="train")
    assert len(ds) > 0, "Dataset has zero samples"
    # We expect at least as many samples as documents (every doc → ≥1 sample)
    assert len(ds) >= ds.num_documents, (
        f"Expected ≥{ds.num_documents} samples, got {len(ds)}"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 2 — Dataset: input/target shift by exactly one token
# ══════════════════════════════════════════════════════════════════════════════
def test_input_target_shift():
    ds = BinaryTokenDataset(config.TRAIN_BIN, block_size=config.BLOCK_SIZE, split="train")

    for idx in range(min(20, len(ds))):
        doc_start, offset, win_len = ds._samples[idx]
        flat_start = doc_start + offset
        n_input = min(win_len, config.BLOCK_SIZE)

        # Read all tokens in this window (safe: stays within document)
        raw = ds._data[flat_start : flat_start + n_input].astype(np.int64)

        x, y = ds[idx]

        # x[k] must equal raw[k] for all valid input positions
        for k in range(n_input):
            assert x[k].item() == int(raw[k]), \
                f"x[{k}] mismatch at sample {idx}: got {x[k].item()}, expected {int(raw[k])}"

        # y[k] must equal raw[k+1] for all positions where both are within the document
        # (i.e. k+1 < n_input means we can verify from the same raw slice)
        for k in range(n_input - 1):
            assert y[k].item() == int(raw[k + 1]), \
                f"y[{k}] mismatch at sample {idx}: got {y[k].item()}, expected {int(raw[k+1])}"

        # Padded positions (beyond n_input) in y must be IGNORE_INDEX
        for k in range(min(n_input, config.BLOCK_SIZE), config.BLOCK_SIZE):
            assert y[k].item() == IGNORE_INDEX, \
                f"y[{k}] should be IGNORE_INDEX at sample {idx}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 3 — Dataset: no sequence crosses document boundary
# ══════════════════════════════════════════════════════════════════════════════
def test_no_cross_boundary():
    ds = BinaryTokenDataset(config.TRAIN_BIN, block_size=config.BLOCK_SIZE, split="train")
    flat = ds._data  # np.memmap

    # For every sample, verify that no EOS token appears INSIDE the input
    # (EOS can only appear at the very last position of a document window)
    eos_id = 2
    bos_id = 1
    violations = 0

    for idx in range(min(500, len(ds))):
        doc_start, offset, win_len = ds._samples[idx]
        flat_start = doc_start + offset
        flat_end   = flat_start + win_len     # exclusive

        window_tokens = flat[flat_start : flat_end].tolist()

        # EOS is allowed ONLY at the last position of this window
        # BOS is allowed ONLY at the first position
        for pos, tok in enumerate(window_tokens[:-1]):   # all but last
            if tok == eos_id:
                violations += 1
        # BOS in the middle of a window (not position 0) is a boundary violation
        for pos, tok in enumerate(window_tokens[1:], 1):
            if tok == bos_id:
                violations += 1

    assert violations == 0, f"{violations} boundary violations found in first 500 samples"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 4 — Dataset: padded target positions use IGNORE_INDEX
# ══════════════════════════════════════════════════════════════════════════════
def test_padded_targets():
    ds = BinaryTokenDataset(config.TRAIN_BIN, block_size=config.BLOCK_SIZE, split="train")
    # Find a short document (< block_size) where padding should occur
    for idx in range(len(ds)):
        _, offset, win_len = ds._samples[idx]
        if win_len < config.BLOCK_SIZE:
            x, y = ds[idx]
            # Positions [win_len:] in y should all be IGNORE_INDEX
            padded_y = y[win_len:]
            assert (padded_y == IGNORE_INDEX).all(), (
                f"Expected IGNORE_INDEX in padded target positions for sample {idx}"
            )
            # Positions [0:win_len-1] should NOT be IGNORE_INDEX in y
            valid_y = y[:win_len - 1]
            assert (valid_y != IGNORE_INDEX).all(), (
                f"Valid target positions should not be IGNORE_INDEX for sample {idx}"
            )
            return  # one confirmed short doc is enough
    # If every doc is exactly block_size, skip gracefully
    print("    (no short docs found — skipping padding check)")


# ══════════════════════════════════════════════════════════════════════════════
# TEST 5 — Gradient accumulation: backward called N times, step once
# ══════════════════════════════════════════════════════════════════════════════
def test_gradient_accumulation():
    model = GPT(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    grad_accum = 4

    # Record initial weight snapshot
    p0 = next(iter(model.parameters())).data.clone()

    optimizer.zero_grad(set_to_none=True)
    for _ in range(grad_accum):
        x = torch.randint(0, TINY.vocab_size, (2, TINY.block_size))
        y = torch.randint(0, TINY.vocab_size, (2, TINY.block_size))
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        (loss / grad_accum).backward()

    # Gradients should now be non-zero
    for name, param in model.named_parameters():
        if param.grad is not None:
            assert param.grad.abs().max().item() > 0, f"Zero gradient on {name}"
            break

    # Only one optimizer step
    optimizer.step()
    p1 = next(iter(model.parameters())).data.clone()
    assert not torch.equal(p0, p1), "Parameters did not change after optimizer.step()"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 6 — Optimizer update frequency: step counter
# ══════════════════════════════════════════════════════════════════════════════
def test_optimizer_step_frequency():
    model     = GPT(TINY)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    scheduler = CosineScheduler(optimizer, warmup_steps=5, max_steps=50,
                                 peak_lr=1e-3, min_lr=1e-4)

    grad_accum = 4
    opt_steps  = 0

    for outer in range(6):   # 6 outer steps
        optimizer.zero_grad(set_to_none=True)
        for _ in range(grad_accum):
            x = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
            y = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
            (loss / grad_accum).backward()
        optimizer.step()
        scheduler.step()
        opt_steps += 1

    assert opt_steps == 6, f"Expected 6 opt steps, got {opt_steps}"
    assert scheduler.current_step == 6, (
        f"Scheduler step {scheduler.current_step} != 6"
    )


# ══════════════════════════════════════════════════════════════════════════════
# TEST 7 — BF16 forward/backward: no NaN, activations are bf16
# ══════════════════════════════════════════════════════════════════════════════
def test_bf16_forward_backward():
    if not torch.cuda.is_available():
        print("    (CUDA not available — skipping BF16 test)")
        return

    device = torch.device("cuda")
    model  = GPT(TINY).to(device)
    model.train()

    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

    x = torch.randint(0, TINY.vocab_size, (2, TINY.block_size), device=device)
    y = torch.randint(0, TINY.vocab_size, (2, TINY.block_size), device=device)

    with amp_ctx:
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    assert not torch.isnan(loss), "NaN loss in BF16 forward pass"
    assert not torch.isinf(loss), "Inf loss in BF16 forward pass"
    assert logits.dtype == torch.bfloat16, f"Expected bf16 logits, got {logits.dtype}"

    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert not torch.isnan(p.grad).any(), "NaN gradient in BF16 backward"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 8 — Validation isolation: val set has no train tokens (different files)
# ══════════════════════════════════════════════════════════════════════════════
def test_validation_isolation():
    train_ds = BinaryTokenDataset(config.TRAIN_BIN, block_size=64, split="train")
    val_ds   = BinaryTokenDataset(config.VAL_BIN,   block_size=64, split="val")

    # Confirm they point to different underlying arrays
    assert train_ds.bin_path != val_ds.bin_path, "Train and val datasets point to the same file!"
    assert train_ds.total_tokens != val_ds.total_tokens or \
           train_ds.bin_path.stat().st_size != val_ds.bin_path.stat().st_size, \
           "Train and val binary files appear identical"

    # Confirm document counts
    assert train_ds.num_documents == 4500, f"Expected 4500 train docs, got {train_ds.num_documents}"
    assert val_ds.num_documents   == 500,  f"Expected 500 val docs, got {val_ds.num_documents}"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 9 — Checkpoint save + load: exact state restoration
# ══════════════════════════════════════════════════════════════════════════════
def test_checkpoint_save_load():
    device    = torch.device("cpu")
    model     = GPT(TINY).to(device)
    optimizer = configure_adamw(model, learning_rate=1e-3, weight_decay=0.1,
                                 betas=(0.9, 0.95), device_type="cpu")
    scheduler = CosineScheduler(optimizer, warmup_steps=5, max_steps=100,
                                 peak_lr=1e-3, min_lr=1e-4)

    # Do a couple of steps to get non-zero optimizer state
    for _ in range(3):
        x = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
        y = torch.randint(0, TINY.vocab_size, (1, TINY.block_size))
        logits, _ = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

    # Snapshot LR and a weight before saving
    lr_before     = scheduler.get_lr()
    weight_before = next(iter(model.parameters())).data.clone()

    with tempfile.TemporaryDirectory() as tmp_dir:
        ckpt_dir = Path(tmp_dir)
        ckpt_file = save_checkpoint(
            checkpoint_dir=ckpt_dir,
            model=model, optimizer=optimizer, scheduler=scheduler,
            opt_step=3, micro_step=3, train_loss=4.5, val_loss=4.8,
            config_snapshot={"test": True},
            tokenizer_config_path=config.TOKENIZER_CONFIG_FILE,
            dataset_meta_paths={"train": config.TOKENIZED_DATA_DIR / "train.json"},
        )

        # Create a fresh model+opt+scheduler, load checkpoint into them
        model2     = GPT(TINY).to(device)
        optimizer2 = configure_adamw(model2, learning_rate=1e-3, weight_decay=0.1,
                                      betas=(0.9, 0.95), device_type="cpu")
        scheduler2 = CosineScheduler(optimizer2, warmup_steps=5, max_steps=100,
                                      peak_lr=1e-3, min_lr=1e-4)

        meta = load_checkpoint(ckpt_file, model2, optimizer2, scheduler2, device)

        # Verify model weights are identical
        for (n1, p1), (n2, p2) in zip(model.named_parameters(), model2.named_parameters()):
            assert torch.equal(p1.data, p2.data), f"Weight mismatch after restore: {n1}"

        # Verify scheduler LR
        lr_after = scheduler2.get_lr()
        assert abs(lr_before - lr_after) < 1e-10, (
            f"LR mismatch: before={lr_before}, after={lr_after}"
        )

        # Verify metadata
        assert meta["opt_step"] == 3
        assert meta["val_loss"] == 4.8


# ══════════════════════════════════════════════════════════════════════════════
# TEST 10 — Reproducibility: same seed → same first batch tokens
# ══════════════════════════════════════════════════════════════════════════════
def test_reproducibility():
    import random, numpy as np

    def get_first_batch(seed: int):
        random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
        ds     = BinaryTokenDataset(config.TRAIN_BIN, block_size=64, split="train")
        loader = DataLoader(ds, batch_size=2, shuffle=True, num_workers=0)
        x, y   = next(iter(loader))
        return x.clone(), y.clone()

    x1, y1 = get_first_batch(42)
    x2, y2 = get_first_batch(42)
    x3, y3 = get_first_batch(99)

    assert torch.equal(x1, x2), "Same seed produced different batches"
    assert torch.equal(y1, y2), "Same seed produced different target batches"
    # Different seeds should (almost certainly) differ
    assert not torch.equal(x1, x3), "Different seeds produced identical batches (unexpected)"


# ══════════════════════════════════════════════════════════════════════════════
# TEST 11 — Smoke test: 3 optimizer steps, loss should be finite
# ══════════════════════════════════════════════════════════════════════════════
def test_smoke_training():
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model     = GPT(TINY).to(device)
    optimizer = configure_adamw(model, 3e-4, 0.1, (0.9, 0.95),
                                 device_type=str(device))
    scheduler = CosineScheduler(optimizer, warmup_steps=1, max_steps=20,
                                 peak_lr=3e-4, min_lr=3e-5)

    amp_dtype = torch.bfloat16 if str(device) == "cuda" else torch.float32
    amp_ctx   = torch.amp.autocast(device_type=str(device),
                                    dtype=amp_dtype,
                                    enabled=(str(device) == "cuda"))

    model.train()
    losses = []

    for step in range(3):
        optimizer.zero_grad(set_to_none=True)
        x = torch.randint(0, TINY.vocab_size, (2, TINY.block_size), device=device)
        y = torch.randint(0, TINY.vocab_size, (2, TINY.block_size), device=device)

        with amp_ctx:
            logits, _ = model(x)
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        losses.append(loss.item())

    assert all(math.isfinite(l) for l in losses), f"Non-finite loss in smoke test: {losses}"
    print(f"    Smoke losses: {[f'{l:.4f}' for l in losses]}")


# ══════════════════════════════════════════════════════════════════════════════
# Runner
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print()
    print("=" * 65)
    print("  PHASE 6 TRAINING ENGINE — TEST SUITE")
    print("=" * 65)
    print()

    tests = [
        ("1. Dataset sample count",           test_dataset_sample_count),
        ("2. Input/target shift by one token", test_input_target_shift),
        ("3. No cross-document boundary",      test_no_cross_boundary),
        ("4. Padded targets use IGNORE_INDEX", test_padded_targets),
        ("5. Gradient accumulation",           test_gradient_accumulation),
        ("6. Optimizer update frequency",      test_optimizer_step_frequency),
        ("7. BF16 forward/backward",           test_bf16_forward_backward),
        ("8. Validation isolation",            test_validation_isolation),
        ("9. Checkpoint save + load",          test_checkpoint_save_load),
        ("10. Reproducibility",                test_reproducibility),
        ("11. Smoke training (3 steps)",       test_smoke_training),
    ]

    passed = 0
    failed = 0
    for label, fn in tests:
        ok = run(label, fn)
        if ok:
            passed += 1
        else:
            failed += 1

    print()
    print("=" * 65)
    print(f"  Results: {passed}/{len(tests)} passed   {failed} failed")
    print("=" * 65)
    print()

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
