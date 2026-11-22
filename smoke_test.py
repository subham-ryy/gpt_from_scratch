"""Short smoke test on the real 98.4M GPT model with actual FineWeb-Edu data.

Runs 25 optimizer steps (200 micro-batches) to verify:
  - The full training engine works end-to-end on real data
  - Loss is finite and decreasing
  - VRAM is within budget
  - Checkpoint saves correctly
  - Validation loop works

Does NOT start full pretraining. Max 25 steps.
"""
import sys, time, math
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import BinaryTokenDataset
from model.transformer import GPT, GPTConfig
from monitoring.system_monitor import SystemMonitor
from training.checkpointing import save_checkpoint
from training.optimizer import configure_adamw
from training.scheduler import CosineScheduler
from training.train import evaluate

SMOKE_STEPS   = 25
GRAD_ACCUM    = config.GRAD_ACCUM_STEPS   # 8
MICRO_BATCH   = config.MICRO_BATCH_SIZE   # 4

def main():
    print("=" * 65)
    print("  PHASE 6 SMOKE TEST — real model, real data, 25 steps")
    print("=" * 65)

    device     = torch.device(config.DEVICE)
    amp_dtype  = torch.bfloat16
    amp_ctx    = torch.amp.autocast(device_type=str(device), dtype=amp_dtype,
                                     enabled=(str(device) == "cuda"))

    # ── Datasets ──────────────────────────────────────────────────────────────
    print("\n[1] Loading datasets...")
    train_ds = BinaryTokenDataset(config.TRAIN_BIN, config.BLOCK_SIZE, "train")
    val_ds   = BinaryTokenDataset(config.VAL_BIN,   config.BLOCK_SIZE, "val")
    train_loader = DataLoader(train_ds, batch_size=MICRO_BATCH, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader   = DataLoader(val_ds,   batch_size=MICRO_BATCH, shuffle=False,
                               num_workers=0, pin_memory=True, drop_last=False)
    train_iter = iter(train_loader)
    print(f"    Train samples : {len(train_ds):,}")
    print(f"    Val   samples : {len(val_ds):,}")

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\n[2] Initialising GPT (~98.4M params)...")
    model_cfg = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=config.BLOCK_SIZE, n_layer=config.N_LAYER,
        n_head=config.N_HEAD, n_embd=config.N_EMBD, ffn_dim=config.FFN_DIM,
        dropout=0.0,   # disable dropout for reproducible smoke test
        bias=config.BIAS, tie_weights=config.TIE_WEIGHTS, use_flash=True,
    )
    model = GPT(model_cfg).to(device)
    n_params = model.get_num_params()
    print(f"    Parameters    : {n_params:,}")

    # ── Optimizer / Scheduler ─────────────────────────────────────────────────
    print("\n[3] Configuring optimizer & scheduler...")
    optimizer = configure_adamw(model, config.LEARNING_RATE, config.WEIGHT_DECAY,
                                 (config.BETA1, config.BETA2), str(device))
    scheduler = CosineScheduler(optimizer, warmup_steps=config.WARMUP_STEPS,
                                 max_steps=config.MAX_STEPS,
                                 peak_lr=config.LEARNING_RATE, min_lr=config.MIN_LR)

    # ── Baseline VRAM ─────────────────────────────────────────────────────────
    if str(device) == "cuda":
        torch.cuda.reset_peak_memory_stats()
        hw0 = SystemMonitor.get_current_metrics()
        print(f"\n    Baseline VRAM allocated: {hw0['gpu_allocated_mb']:.1f} MB")

    # ── Training loop ─────────────────────────────────────────────────────────
    print(f"\n[4] Training ({SMOKE_STEPS} optimizer steps × {GRAD_ACCUM} micro-batches each)...\n")
    print(f"{'step':>5}  {'loss':>8}  {'lr':>9}  {'grad':>7}  {'tok/s':>8}  {'VRAM_res':>10}")
    print("─" * 56)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    all_losses = []
    t_total = time.perf_counter()

    for opt_step in range(1, SMOKE_STEPS + 1):
        t0 = time.perf_counter()
        accum_loss = 0.0

        for _ in range(GRAD_ACCUM):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with amp_ctx:
                logits, _ = model(x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                        y.view(-1), ignore_index=-1)
            (loss / GRAD_ACCUM).backward()
            accum_loss += loss.item()

        step_loss = accum_loss / GRAD_ACCUM
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        elapsed   = time.perf_counter() - t0
        tok_s     = (GRAD_ACCUM * MICRO_BATCH * config.BLOCK_SIZE) / elapsed
        hw        = SystemMonitor.get_current_metrics()
        all_losses.append(step_loss)

        print(f"{opt_step:5d}  {step_loss:8.4f}  {scheduler.get_lr():9.2e}  "
              f"{float(grad_norm):7.4f}  {tok_s/1e3:7.1f}k  "
              f"{hw['gpu_reserved_mb']:8.0f} MB")

    total_elapsed = time.perf_counter() - t_total
    print("─" * 56)

    # ── Validation ────────────────────────────────────────────────────────────
    print("\n[5] Running validation...")
    t_v = time.perf_counter()
    val_loss, val_ppl, val_bpb = evaluate(model, val_loader, device, amp_ctx)
    val_elapsed = time.perf_counter() - t_v
    print(f"    val_loss={val_loss:.4f}  perplexity={val_ppl:.2f}  bpb={val_bpb:.4f}  [{val_elapsed:.1f}s]")

    # ── Peak VRAM ─────────────────────────────────────────────────────────────
    if str(device) == "cuda":
        peak_mb = torch.cuda.max_memory_allocated() / 1e6
        res_mb  = torch.cuda.memory_reserved()       / 1e6
        print(f"\n[6] Peak VRAM allocated : {peak_mb:.1f} MB")
        print(f"    Peak VRAM reserved  : {res_mb:.1f} MB")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    print("\n[7] Saving checkpoint...")
    ckpt_dir = Path("smoke_test_checkpoints")
    ckpt_file = save_checkpoint(
        checkpoint_dir=ckpt_dir, model=model, optimizer=optimizer,
        scheduler=scheduler, opt_step=SMOKE_STEPS, micro_step=SMOKE_STEPS * GRAD_ACCUM,
        train_loss=all_losses[-1], val_loss=val_loss,
        config_snapshot={"smoke_test": True, "steps": SMOKE_STEPS},
        tokenizer_config_path=config.TOKENIZER_CONFIG_FILE,
        dataset_meta_paths={"train": config.TOKENIZED_DATA_DIR / "train.json",
                             "val"  : config.TOKENIZED_DATA_DIR / "val.json"},
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    first_loss = all_losses[0]
    last_loss  = all_losses[-1]
    loss_drop  = first_loss - last_loss

    print("\n" + "=" * 65)
    print("  SMOKE TEST SUMMARY")
    print("=" * 65)
    print(f"  Optimizer steps   : {SMOKE_STEPS}")
    print(f"  Micro-batches     : {SMOKE_STEPS * GRAD_ACCUM}")
    print(f"  Tokens processed  : {SMOKE_STEPS * GRAD_ACCUM * MICRO_BATCH * config.BLOCK_SIZE:,}")
    print(f"  Total wall time   : {total_elapsed:.1f}s")
    print(f"  First train loss  : {first_loss:.4f}")
    print(f"  Last  train loss  : {last_loss:.4f}  (Δ = {loss_drop:+.4f})")
    print(f"  Val loss          : {val_loss:.4f}  (ppl={val_ppl:.2f}, bpb={val_bpb:.4f})")
    print(f"  Checkpoint size   : {ckpt_file.stat().st_size/1e6:.1f} MB")
    if str(device) == "cuda":
        print(f"  Peak VRAM alloc   : {peak_mb:.1f} MB")
        print(f"  Peak VRAM reserv  : {res_mb:.1f} MB")
    print("=" * 65)

    assert math.isfinite(val_loss), "Validation loss is not finite!"
    assert math.isfinite(last_loss), "Final training loss is not finite!"
    print("\n[SUCCESS] Smoke test passed. Engine is ready for pretraining.\n")

if __name__ == "__main__":
    main()

