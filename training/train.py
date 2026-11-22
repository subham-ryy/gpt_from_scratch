"""GPT Pretraining Engine — Phase 6 & Track A Continued Pretraining.

Usage
-----
    # 1. Fresh pretraining from scratch (Phase 6 baseline):
    python training/train.py

    # 2. Track A Continued Pretraining from best Phase 6 checkpoint:
    python training/train.py --continued --base-checkpoint experiments/20261104_211024_gpt_pretrain/checkpoints/best.pt

    # 3. 25-step smoke test for continued pretraining verification:
    python training/train.py --continued --smoke-test

    # 4. Resume an interrupted run:
    python training/train.py --resume path/to/last.pt

Key Architecture & Optimization:
- Model: Decoder-only Transformer (12 layers, 12 heads, 768 embd, ~98.4M params).
- Dataset: PackedBinaryTokenDataset with greedy document bin-packing (>98% efficiency).
- Boundaries: Loss is NEVER computed across document boundaries (EOS target is -1).
- Regularization: Z-loss in FP32: 1e-4 * mean(logsumexp(logits)^2) on valid targets.
- Precision: BF16 AMP mixed precision.
- Token Accounting: 4 independent counters (nominal, valid, unique tokens, effective epochs).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import config
from data.dataset import BinaryTokenDataset, PackedBinaryTokenDataset
from model.transformer import GPT, GPTConfig
from monitoring.experiment_tracker import ExperimentTracker
from monitoring.system_monitor import SystemMonitor
from training.checkpointing import load_checkpoint, save_checkpoint
from training.optimizer import configure_adamw
from training.scheduler import CosineScheduler


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_val_bytes_per_token() -> float:
    """Return average UTF-8 bytes per token for the validation dataset."""
    val_meta_path = config.TOKENIZED_DATA_DIR / "val.json"
    if val_meta_path.exists():
        try:
            with open(val_meta_path, "r", encoding="utf-8") as f:
                d = json.load(f)
                if "total_bytes" in d and "total_tokens" in d and d["total_tokens"] > 0:
                    return float(d["total_bytes"]) / float(d["total_tokens"])
        except Exception:
            pass
    return 4.2514  # Calibrated on FineWeb-Edu 500-doc validation split (2,409,117 bytes / 566,663 tokens)


@torch.no_grad()
def evaluate(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    amp_ctx,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    """Run the full validation loop over the frozen validation benchmark.

    Returns
    -------
    avg_loss   : Mean cross-entropy loss over all validated tokens (nats).
    perplexity : exp(avg_loss).
    bpb        : Bits Per Byte (val_loss / (ln(2) * bytes_per_token)).
    """
    model.eval()
    total_loss   = 0.0
    total_tokens = 0

    for batch_idx, (x, y) in enumerate(val_loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with amp_ctx:
            logits, _ = model(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
                ignore_index=-1,
                reduction="sum",
            )

        valid_tokens = (y != -1).sum().item()
        total_loss   += loss.item()
        total_tokens += valid_tokens

    model.train()

    if total_tokens == 0:
        return float("inf"), float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl      = math.exp(min(avg_loss, 20.0))
    bytes_per_tok = get_val_bytes_per_token()
    bpb      = (avg_loss / math.log(2)) / bytes_per_tok

    return avg_loss, ppl, bpb


def train(
    resume_from: str | None = None,
    continued: bool = False,
    base_checkpoint: str | None = None,
    smoke_test: bool = False,
    max_steps_override: int | None = None,
    peak_lr_override: float | None = None,
    min_lr_override: float | None = None,
    warmup_steps_override: int | None = None,
    eval_interval_override: int | None = None,
    save_interval_override: int | None = None,
    z_loss_weight: float = 1e-4,
    use_packed: bool = True,
    train_bin_override: str | None = None,
    val_bin_override: str | None = None,
    eval_batches: int | None = 250,
) -> None:
    # ── Setup ─────────────────────────────────────────────────────────────────
    set_seed(config.RANDOM_SEED)
    device     = torch.device(config.DEVICE)
    device_str = config.DEVICE

    # Mode determination and hyperparameter resolution
    is_continued = continued or (base_checkpoint is not None)

    if is_continued:
        exp_tag = "gpt_continued_smoke_test" if smoke_test else "gpt_continued_pretrain"
        peak_lr = peak_lr_override if peak_lr_override is not None else 1.5e-4
        min_lr = min_lr_override if min_lr_override is not None else 1.5e-5
        max_steps = max_steps_override if max_steps_override is not None else (5 if smoke_test else 9000)
        warmup_steps = warmup_steps_override if warmup_steps_override is not None else (2 if smoke_test else 300)
        if warmup_steps >= max_steps:
            warmup_steps = max(1, max_steps // 2)
        eval_interval = eval_interval_override if eval_interval_override is not None else (5 if smoke_test else 500)
        save_interval = save_interval_override if save_interval_override is not None else (5 if smoke_test else 500)
        log_interval = 1 if smoke_test else 10
    else:
        exp_tag = "gpt_pretrain"
        peak_lr = peak_lr_override if peak_lr_override is not None else config.LEARNING_RATE
        min_lr = min_lr_override if min_lr_override is not None else config.MIN_LR
        max_steps = max_steps_override if max_steps_override is not None else (5 if smoke_test else config.MAX_STEPS)
        warmup_steps = warmup_steps_override if warmup_steps_override is not None else (2 if smoke_test else config.WARMUP_STEPS)
        if warmup_steps >= max_steps:
            warmup_steps = max(1, max_steps // 2)
        eval_interval = eval_interval_override if eval_interval_override is not None else (5 if smoke_test else config.EVAL_INTERVAL)
        save_interval = save_interval_override if save_interval_override is not None else (5 if smoke_test else config.SAVE_INTERVAL)
        log_interval = 1 if smoke_test else config.LOG_INTERVAL

    # Data paths
    if train_bin_override:
        train_bin_path = Path(train_bin_override)
    elif is_continued and (config.TOKENIZED_DATA_DIR / "train_expanded.bin").exists():
        train_bin_path = config.TOKENIZED_DATA_DIR / "train_expanded.bin"
    else:
        train_bin_path = config.TRAIN_BIN

    if val_bin_override:
        val_bin_path = Path(val_bin_override)
    else:
        val_bin_path = config.VAL_BIN

    print("=" * 68)
    print(f"  GPT PRETRAINING ENGINE — {'Track A: Continued Pretraining' if is_continued else 'Phase 6'}")
    if smoke_test:
        print("  MODE        : 25-Step Verification Smoke Test")
    print("=" * 68)
    print(f"  Device      : {device_str.upper()}")
    if device_str == "cpu":
        print("\n" + "!" * 68)
        print("  WARNING: RUNNING ON CPU! CUDA environment recommended for speed.")
        print(r"  Activate: D:\pytorch\.venv\Scripts\Activate.ps1")
        print("!" * 68 + "\n")
    print(f"  Precision   : {'BF16 AMP' if config.USE_AMP and device_str == 'cuda' else 'FP32'}")
    print(f"  Micro-batch : {config.MICRO_BATCH_SIZE}")
    print(f"  Grad accum  : {config.GRAD_ACCUM_STEPS}")
    eff_batch_seqs = config.MICRO_BATCH_SIZE * config.GRAD_ACCUM_STEPS
    eff_batch_toks = eff_batch_seqs * config.BLOCK_SIZE
    print(f"  Eff. batch  : {eff_batch_seqs} seqs ({eff_batch_toks:,} nominal tokens/step)")
    print(f"  Peak LR     : {peak_lr:.2e}")
    print(f"  Min LR      : {min_lr:.2e}")
    print(f"  Warmup steps: {warmup_steps:,}")
    print(f"  Max steps   : {max_steps:,}")
    print(f"  Z-loss      : weight = {z_loss_weight:.1e}")
    print(f"  Dataset     : {train_bin_path.name} ({'PackedBinaryTokenDataset' if use_packed else 'BinaryTokenDataset'})")
    print()

    # AMP context
    if config.USE_AMP and device_str == "cuda":
        amp_dtype = torch.bfloat16 if config.AMP_DTYPE == "bfloat16" else torch.float16
        amp_ctx   = torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
    else:
        amp_ctx = torch.amp.autocast(device_type="cpu", enabled=False)

    # ── Datasets and DataLoaders ───────────────────────────────────────────────
    print("Loading datasets...")
    if use_packed:
        train_dataset = PackedBinaryTokenDataset(
            train_bin_path, block_size=config.BLOCK_SIZE, split="train", strategy="first_fit"
        )
    else:
        train_dataset = BinaryTokenDataset(
            train_bin_path, block_size=config.BLOCK_SIZE, split="train"
        )

    # Validation dataset always uses the frozen standard benchmark
    val_dataset = BinaryTokenDataset(
        val_bin_path, block_size=config.BLOCK_SIZE, split="val"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config.MICRO_BATCH_SIZE,
        shuffle=True,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device_str == "cuda"),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.MICRO_BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=(device_str == "cuda"),
        drop_last=False,
    )

    train_iter = iter(train_loader)

    # ── Model ─────────────────────────────────────────────────────────────────
    print("\nInitialising model...")
    model_cfg = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=config.BLOCK_SIZE,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        ffn_dim=config.FFN_DIM,
        dropout=config.DROPOUT,
        bias=config.BIAS,
        tie_weights=config.TIE_WEIGHTS,
        use_flash=True,
    )
    model = GPT(model_cfg).to(device)
    n_params = model.get_num_params()
    print(f"  Parameters  : {n_params:,}")

    # ── Load Base Checkpoint (for Continued Pretraining) ─────────────────────
    base_val_loss = None
    if is_continued and not resume_from:
        if base_checkpoint is None:
            # Default to best Phase 6 checkpoint
            base_checkpoint = str(
                PROJECT_ROOT / "experiments" / "20261104_211024_gpt_pretrain" / "checkpoints" / "best.pt"
            )
        base_path = Path(base_checkpoint)
        print(f"\n[Continued Pretraining] Loading base model weights from: {base_path}")
        if not base_path.exists():
            raise FileNotFoundError(f"Base checkpoint not found: {base_path}")

        ckpt_data = torch.load(str(base_path), map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt_data["model_state"])
        base_step = ckpt_data.get("opt_step", 1200)
        base_val_loss = ckpt_data.get("val_loss", 5.2793)
        del ckpt_data
        if device_str == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
        print(f"  Base weights successfully loaded (from Phase 6 step {base_step:,}, baseline val_loss={base_val_loss:.4f})")

    # ── Optimizer and Scheduler ────────────────────────────────────────────────
    print("\nConfiguring optimizer & LR scheduler...")
    optimizer = configure_adamw(
        model,
        learning_rate=peak_lr,
        weight_decay=config.WEIGHT_DECAY,
        betas=(config.BETA1, config.BETA2),
        device_type=device_str,
    )
    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=max_steps,
        peak_lr=peak_lr,
        min_lr=min_lr,
    )

    # ── Experiment Tracker ────────────────────────────────────────────────────
    cfg_snapshot = {
        "model_name"      : "GPT-98.4M",
        "n_params"        : n_params,
        "mode"            : "continued_pretrain" if is_continued else "scratch_pretrain",
        "base_checkpoint" : str(base_checkpoint) if is_continued else None,
        "vocab_size"      : config.VOCAB_SIZE,
        "block_size"      : config.BLOCK_SIZE,
        "n_layer"         : config.N_LAYER,
        "n_head"          : config.N_HEAD,
        "n_embd"          : config.N_EMBD,
        "micro_batch"     : config.MICRO_BATCH_SIZE,
        "grad_accum"      : config.GRAD_ACCUM_STEPS,
        "effective_batch" : eff_batch_seqs,
        "peak_lr"         : peak_lr,
        "min_lr"          : min_lr,
        "warmup_steps"    : warmup_steps,
        "max_steps"       : max_steps,
        "z_loss_weight"   : z_loss_weight,
        "use_packed"      : use_packed,
        "packing_eff"     : getattr(train_dataset, "packing_efficiency", 0.0),
        "train_bin"       : str(train_bin_path),
        "val_bin"         : str(val_bin_path),
        "smoke_test"      : smoke_test,
    }
    tracker = ExperimentTracker(
        experiment_name=exp_tag,
        config_dict=cfg_snapshot,
    )
    ckpt_dir = tracker.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta_paths = {
        "train": train_bin_path.with_suffix(".json") if train_bin_path.with_suffix(".json").exists() else config.TOKENIZED_DATA_DIR / ("train_expanded.json" if is_continued else "train.json"),
        "val"  : val_bin_path.with_suffix(".json") if val_bin_path.with_suffix(".json").exists() else config.TOKENIZED_DATA_DIR / "val.json",
    }

    # ── Resumption (if requested) ─────────────────────────────────────────────
    start_opt_step  = 0
    start_micro     = 0
    best_val_loss   = base_val_loss if base_val_loss is not None else float("inf")

    if resume_from:
        meta = load_checkpoint(Path(resume_from), model, optimizer, scheduler, device)
        start_opt_step = meta["opt_step"]
        start_micro    = meta["micro_step"]
        best_val_loss  = meta["val_loss"] or float("inf")
        print(f"\n[Resume] Starting from opt_step={start_opt_step:,}")

    # ── Phase D: Independent Token Accounting Counters ────────────────────────
    nominal_tokens_processed = start_opt_step * eff_batch_toks
    valid_tokens_processed   = int(start_opt_step * eff_batch_toks * (getattr(train_dataset, "packing_efficiency", 100.0) / 100.0))
    unique_train_tokens      = train_dataset.total_tokens

    # ── Baseline Pre-Evaluation ───────────────────────────────────────────────
    if is_continued and start_opt_step == 0:
        print("\n[Baseline Check] Evaluating base model on frozen validation set before training...")
        init_loss, init_ppl, init_bpb = evaluate(model, val_loader, device, amp_ctx, max_batches=20 if smoke_test else eval_batches)
        print(f"  Base Model Benchmark: val_loss={init_loss:.4f}  ppl={init_ppl:.2f}  bpb={init_bpb:.4f}")
        if not smoke_test:
            best_val_loss = init_loss

    # ── Training State Variables ──────────────────────────────────────────────
    opt_step     = start_opt_step
    micro_step   = start_micro
    running_loss = 0.0
    t_step_start = time.perf_counter()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device_str == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'─'*68}")
    print(f"  Starting training: step {opt_step} → {max_steps} ({max_steps - opt_step:,} steps to execute)")
    print(f"  Unique train tokens in corpus: {unique_train_tokens:,}")
    print(f"{'─'*68}\n")

    # ── Main Training Loop ────────────────────────────────────────────────────
    while opt_step < max_steps:

        accum_total_loss = 0.0
        accum_ce_loss    = 0.0
        accum_z_loss     = 0.0
        actual_micro     = 0
        step_valid_toks  = 0

        for accum_idx in range(config.GRAD_ACCUM_STEPS):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            # Valid target tokens in this micro-batch
            valid_mask = (y != -1)
            num_valid = valid_mask.sum().item()
            step_valid_toks += num_valid

            # Forward pass + Loss computation under BF16 autocast
            with amp_ctx:
                logits, _ = model(x)  # [B, T, V]

                # 1. Next-token Cross-Entropy Loss
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=-1,
                )

                # 2. Z-Loss Regularization (BF16 vocab reduction before casting [B, T] to FP32)
                if z_loss_weight > 0.0 and num_valid > 0:
                    log_z = torch.logsumexp(logits, dim=-1).float()  # [B, T] in FP32 (avoids huge [B, T, V] FP32 tensor)
                    z_loss = z_loss_weight * (log_z[valid_mask] ** 2).mean()
                    total_loss = ce_loss + z_loss
                else:
                    z_loss = torch.tensor(0.0, device=logits.device)
                    total_loss = ce_loss

            # Scaled backward for gradient accumulation
            (total_loss / config.GRAD_ACCUM_STEPS).backward()

            accum_total_loss += total_loss.item()
            accum_ce_loss    += ce_loss.item()
            accum_z_loss     += z_loss.item()
            micro_step       += 1
            actual_micro     += 1

        # ── Optimizer Step ─────────────────────────────────────────────────────
        step_loss    = accum_total_loss / actual_micro
        step_ce_loss = accum_ce_loss / actual_micro
        step_z_loss  = accum_z_loss / actual_micro

        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)

        opt_step += 1
        current_lr = scheduler.get_lr(opt_step)

        # Token counters update
        nominal_tokens_processed += (actual_micro * config.MICRO_BATCH_SIZE * config.BLOCK_SIZE)
        valid_tokens_processed   += step_valid_toks
        effective_epochs = valid_tokens_processed / max(unique_train_tokens, 1)

        t_now = time.perf_counter()
        step_elapsed = t_now - t_step_start
        t_step_start = t_now
        tok_per_sec = (actual_micro * config.MICRO_BATCH_SIZE * config.BLOCK_SIZE) / max(step_elapsed, 1e-9)
        running_loss = 0.9 * running_loss + 0.1 * step_loss

        # ── Logging ───────────────────────────────────────────────────────────
        if opt_step == 1 or opt_step % log_interval == 0:
            hw = SystemMonitor.get_current_metrics()
            print(
                f"step {opt_step:5d} | loss {step_loss:.4f} (ce {step_ce_loss:.4f}, z {step_z_loss:.4f}) | "
                f"lr {current_lr:.2e} | grad {grad_norm:.3f} | "
                f"{tok_per_sec/1e3:5.1f}k tok/s | ep {effective_epochs:.3f} | "
                f"VRAM {hw['gpu_reserved_mb']:.0f}MB"
            )
            tracker.log_step(
                step=opt_step,
                train_loss=step_loss,
                lr=current_lr,
                grad_norm=float(grad_norm),
                tokens_in_step=int(actual_micro * config.MICRO_BATCH_SIZE * config.BLOCK_SIZE),
                total_tokens=valid_tokens_processed,
                ce_loss=round(float(step_ce_loss), 4),
                z_loss=round(float(step_z_loss), 5),
                nominal_tokens=nominal_tokens_processed,
                valid_tokens=valid_tokens_processed,
                effective_epochs=round(effective_epochs, 4),
                micro_step=micro_step,
                opt_step=opt_step,
            )

        # ── Validation & Checkpointing ────────────────────────────────────────
        if opt_step % eval_interval == 0 or opt_step == max_steps:
            t_eval_start = time.perf_counter()
            max_eval_b = 20 if smoke_test else eval_batches
            val_loss, val_ppl, val_bpb = evaluate(model, val_loader, device, amp_ctx, max_batches=max_eval_b)
            eval_elapsed = time.perf_counter() - t_eval_start

            if device_str == "cuda":
                torch.cuda.empty_cache()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            print(
                f"  [EVAL step {opt_step:5d}] "
                f"val_loss={val_loss:.4f}  ppl={val_ppl:.2f}  bpb={val_bpb:.4f}  "
                f"({'NEW BEST' if is_best else 'no improvement'})  "
                f"[{eval_elapsed:.1f}s]"
            )
            tracker.log_eval(
                step=opt_step,
                val_loss=val_loss,
                val_perplexity=val_ppl,
                val_bpb=round(val_bpb, 4),
                eval_seconds=round(eval_elapsed, 2),
            )

            if opt_step % save_interval == 0 or is_best or opt_step == max_steps:
                save_checkpoint(
                    checkpoint_dir=ckpt_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    opt_step=opt_step,
                    micro_step=micro_step,
                    train_loss=step_loss,
                    val_loss=val_loss,
                    config_snapshot=cfg_snapshot,
                    tokenizer_config_path=config.TOKENIZER_CONFIG_FILE,
                    dataset_meta_paths=dataset_meta_paths,
                    is_best=is_best,
                )

    # ── Final Summary ─────────────────────────────────────────────────────────
    bytes_per_tok = get_val_bytes_per_token()
    best_ppl = math.exp(min(best_val_loss, 20.0))
    best_bpb = (best_val_loss / math.log(2)) / bytes_per_tok

    print("\n" + "=" * 68)
    print(f"  TRAINING ENGINE RUN COMPLETED — {opt_step:,} STEPS")
    print(f"  Best Validation Loss    : {best_val_loss:.4f}")
    print(f"  Best Perplexity         : {best_ppl:.2f}")
    print(f"  Best Bits Per Byte      : {best_bpb:.4f}")
    print(f"  Nominal Tokens Seen     : {nominal_tokens_processed:,}")
    print(f"  Valid Tokens Processed  : {valid_tokens_processed:,}")
    print(f"  Unique Dataset Tokens   : {unique_train_tokens:,}")
    print(f"  Effective Epochs Seen   : {valid_tokens_processed / max(unique_train_tokens, 1):.4f}")
    print("=" * 68)

    tracker.finish()


def main() -> None:
    parser = argparse.ArgumentParser(description="GPT Pretraining Engine — Track A")
    parser.add_argument("--resume", type=str, default=None, help="Resume training from an existing checkpoint.")
    parser.add_argument("--continued", action="store_true", help="Launch Track A Continued Pretraining from base checkpoint.")
    parser.add_argument("--base-checkpoint", type=str, default=None, help="Path to base checkpoint for continued pretraining.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 25-step verification smoke test.")
    parser.add_argument("--max-steps", type=int, default=None, help="Total optimizer steps to run.")
    parser.add_argument("--peak-lr", type=float, default=None, help="Peak learning rate.")
    parser.add_argument("--min-lr", type=float, default=None, help="Min learning rate floor.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Linear warmup steps.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval in optimizer steps.")
    parser.add_argument("--save-interval", type=int, default=None, help="Checkpoint saving interval in optimizer steps.")
    parser.add_argument("--z-loss", type=float, default=1e-4, help="Z-loss regularization weight.")
    parser.add_argument("--train-bin", type=str, default=None, help="Explicit path to train .bin shard.")
    parser.add_argument("--val-bin", type=str, default=None, help="Explicit path to val .bin shard.")
    parser.add_argument("--eval-batches", type=int, default=250, help="Max batches to evaluate during periodic validation (default: 250 = ~1M tokens).")
    parser.add_argument("--full-eval", action="store_true", help="Evaluate full validation split on every eval step.")
    parser.add_argument("--no-packed", action="store_true", help="Disable greedy bin-packing.")
    args = parser.parse_args()

    train(
        resume_from=args.resume,
        continued=args.continued,
        base_checkpoint=args.base_checkpoint,
        smoke_test=args.smoke_test,
        max_steps_override=args.max_steps,
        peak_lr_override=args.peak_lr,
        min_lr_override=args.min_lr,
        warmup_steps_override=args.warmup_steps,
        eval_interval_override=args.eval_interval,
        save_interval_override=args.save_interval,
        z_loss_weight=args.z_loss,
        use_packed=not args.no_packed,
        train_bin_override=args.train_bin,
        val_bin_override=args.val_bin,
        eval_batches=None if args.full_eval else args.eval_batches,
    )


if __name__ == "__main__":
    main()
