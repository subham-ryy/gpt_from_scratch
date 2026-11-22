"""Supervised Fine-Tuning (SFT) Training Engine for GPT-98.4M (Phase 8).

Fine-tunes the base Track A pretrained model on factual Q&A and general instruction datasets
with strict prompt loss masking and BF16 Z-loss logit stabilization.

Usage:
    # 1. 25-step smoke test verification:
    python training/train_sft.py --smoke-test

    # 2. Full SFT run (~2.0 epochs / 4,000 steps):
    python training/train_sft.py

    # 3. Custom run with overrides:
    python training/train_sft.py --max-steps 3500 --peak-lr 5e-5 --base-checkpoint path/to/best.pt
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

import config
from data.sft_dataset import (
    SFTDataset,
    LengthGroupedBatchSampler,
    sft_collate_fn,
    BOS_ID,
    EOS_ID,
    IGNORE_INDEX,
)
from model.transformer import GPT, GPTConfig
from monitoring.experiment_tracker import ExperimentTracker
from monitoring.system_monitor import SystemMonitor
from training.checkpointing import save_checkpoint
from training.optimizer import configure_adamw
from training.scheduler import CosineScheduler


DEFAULT_BASE_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "20261106_032904_gpt_continued_pretrain"
    / "checkpoints"
    / "best.pt"
)

PROMPT_SAMPLES = [
    "What is the capital of France?",
    "Explain the process of photosynthesis.",
    "In computer science, what is an algorithm?",
    "What is machine learning?",
]


def set_seed(seed: int) -> None:
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def evaluate_sft(
    model: GPT,
    val_loader: DataLoader,
    device: torch.device,
    amp_ctx,
    max_batches: int | None = None,
) -> tuple[float, float, float]:
    """Run validation over SFT validation dataset (only active response tokens)."""
    model.eval()
    total_loss = 0.0
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
                ignore_index=IGNORE_INDEX,
                reduction="sum",
            )

        valid_tokens = (y != IGNORE_INDEX).sum().item()
        total_loss += loss.item()
        total_tokens += valid_tokens

    model.train()

    if total_tokens == 0:
        return float("inf"), float("inf"), float("inf")

    avg_loss = total_loss / total_tokens
    ppl = math.exp(min(avg_loss, 20.0))
    bpb = avg_loss / (math.log(2) * 4.25)

    return avg_loss, ppl, bpb


@torch.no_grad()
def generate_sample_answers(
    model: GPT,
    tokenizer: Tokenizer,
    prompts: list[str],
    device: torch.device,
    max_new_tokens: int = 80,
) -> list[str]:
    """Generate assistant responses for tracking qualitative alignment during training."""
    model.eval()
    outputs = []
    for q in prompts:
        prefix_str = f"### Question:\n{q}\n\n### Answer:\n"
        prefix_ids = [BOS_ID] + tokenizer.encode(prefix_str).ids
        idx = torch.tensor([prefix_ids], dtype=torch.long, device=device)

        out_idx = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=0.4,
            top_p=0.90,
            min_p=0.08,
            repetition_penalty=1.15,
            no_repeat_ngram_size=4,
            eos_token_id=EOS_ID,
        )

        gen_tokens = out_idx[0, len(prefix_ids):].tolist()
        if EOS_ID in gen_tokens:
            gen_tokens = gen_tokens[: gen_tokens.index(EOS_ID)]
        ans_text = tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
        outputs.append(f"Q: {q}\nA: {ans_text}")
    model.train()
    return outputs


def train_sft(
    base_checkpoint: str | Path = DEFAULT_BASE_CHECKPOINT,
    smoke_test: bool = False,
    max_steps_override: Optional[int] = None,
    peak_lr_override: Optional[float] = None,
    min_lr_override: Optional[float] = None,
    warmup_steps_override: Optional[int] = None,
    eval_interval_override: Optional[int] = None,
    save_interval_override: Optional[int] = None,
    micro_batch_size: int = 4,
    grad_accum_steps: int = 8,
    weight_decay: float = 0.05,
    z_loss_weight: float = 1e-4,
    train_jsonl: Path = config.DATA_DIR / "sft" / "sft_train.jsonl",
    val_jsonl: Path = config.DATA_DIR / "sft" / "sft_val.jsonl",
) -> None:
    set_seed(config.RANDOM_SEED)
    device = torch.device(config.DEVICE)
    device_str = config.DEVICE

    print("=" * 72)
    print("  PHASE 8: INSTRUCTION & FACTUAL SUPERVISED FINE-TUNING (SFT)")
    print("=" * 72)

    # 1. Hyperparameters
    peak_lr = peak_lr_override or (1e-4 if smoke_test else 5e-5)
    min_lr = min_lr_override or (1e-5 if smoke_test else 5e-6)
    warmup_steps = warmup_steps_override or (5 if smoke_test else 100)
    max_steps = max_steps_override or (25 if smoke_test else 4000)
    eval_interval = eval_interval_override or (10 if smoke_test else 250)
    save_interval = save_interval_override or (10 if smoke_test else 250)
    log_interval = 5 if smoke_test else 10

    if smoke_test:
        print("[SMOKE TEST] Running quick 25-step verification...")
        warmup_steps = min(warmup_steps, max(1, max_steps // 2))

    # 2. Tokenizer
    print(f"\n[Tokenizer] Loading from: {config.TOKENIZER_FILE.name}")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    # 3. Datasets & DataLoaders
    max_train_samples = 400 if smoke_test else None
    max_val_samples = 100 if smoke_test else None

    print(f"[Dataset] Loading SFT train dataset: {train_jsonl}")
    train_dataset = SFTDataset(
        jsonl_path=train_jsonl,
        tokenizer=tokenizer,
        block_size=config.BLOCK_SIZE,
        max_samples=max_train_samples,
        dynamic_padding=True,
    )

    print(f"[Dataset] Loading SFT val dataset: {val_jsonl}")
    val_dataset = SFTDataset(
        jsonl_path=val_jsonl,
        tokenizer=tokenizer,
        block_size=config.BLOCK_SIZE,
        max_samples=max_val_samples,
        dynamic_padding=True,
    )

    # Length-based bucketing sampler for micro-batches with minimal padding waste
    train_sampler = LengthGroupedBatchSampler(
        train_dataset,
        batch_size=micro_batch_size,
        bucket_size=16,
        shuffle=True,
        drop_last=True,
        seed=config.RANDOM_SEED,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=sft_collate_fn,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=micro_batch_size,
        shuffle=False,
        collate_fn=sft_collate_fn,
        num_workers=0,
        pin_memory=(device_str == "cuda"),
        drop_last=False,
    )

    eff_batch_samples = micro_batch_size * grad_accum_steps
    steps_per_epoch = len(train_dataset) / eff_batch_samples
    num_epochs = max_steps / max(steps_per_epoch, 1e-6)

    print(f"\nTraining Scope:")
    print(f"  Train examples  : {len(train_dataset):,}")
    print(f"  Val examples    : {len(val_dataset):,}")
    print(f"  Micro batch size: {micro_batch_size}")
    print(f"  Grad accum steps: {grad_accum_steps}")
    print(f"  Effective batch : {eff_batch_samples} samples / step")
    print(f"  Steps per epoch : {steps_per_epoch:.1f} steps")
    print(f"  Planned steps   : {max_steps:,} steps (~{num_epochs:.2f} epochs)")
    print(f"  Peak LR / Min LR: {peak_lr:.2e} / {min_lr:.2e}")
    print(f"  Warmup steps    : {warmup_steps}")

    # 4. Model Architecture & Base Weight Loading
    print(f"\n[Architecture] Building GPT-98.4M model...")
    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=config.BLOCK_SIZE,
        n_layer=config.N_LAYER,
        n_head=config.N_HEAD,
        n_embd=config.N_EMBD,
        ffn_dim=config.FFN_DIM,
        dropout=0.05,  # Slight regularizing dropout for SFT
        bias=config.BIAS,
        tie_weights=config.TIE_WEIGHTS,
        use_flash=True,
    )

    model = GPT(model_config).to(device)
    print(f"  Parameters: {model.get_num_params():,}")

    base_path = Path(base_checkpoint)
    print(f"\n[Weights] Loading base weights from: {base_path}")
    if not base_path.exists():
        raise FileNotFoundError(f"Base checkpoint does not exist: {base_path}")

    base_ckpt = torch.load(str(base_path), map_location="cpu", weights_only=False)
    model.load_state_dict(base_ckpt["model_state"])
    base_step = base_ckpt.get("opt_step", "N/A")
    base_val = base_ckpt.get("val_loss", "N/A")
    del base_ckpt
    if device_str == "cuda":
        torch.cuda.empty_cache()
    print(f"  Successfully loaded Track A weights (Pretrained step {base_step}, pretrain val_loss={base_val})")

    # 5. Optimizer & Scheduler
    optimizer = configure_adamw(
        model,
        learning_rate=peak_lr,
        weight_decay=weight_decay,
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

    # 6. Precision & Experiment Tracking
    amp_ctx = (
        torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)
        if device_str == "cuda" and config.USE_AMP
        else torch.amp.autocast(device_type="cpu", enabled=False)
    )

    exp_name = f"gpt_sft{'_smoke' if smoke_test else ''}"
    tracker = ExperimentTracker(
        experiment_name=exp_name,
        config_dict={
            "experiment_name": exp_name,
            "mode": "sft",
            "base_checkpoint": str(base_path),
            "max_steps": max_steps,
            "peak_lr": peak_lr,
            "min_lr": min_lr,
            "warmup_steps": warmup_steps,
            "eff_batch_samples": eff_batch_samples,
            "weight_decay": weight_decay,
            "z_loss_weight": z_loss_weight,
            "train_samples": len(train_dataset),
            "val_samples": len(val_dataset),
        },
        base_dir=config.EXPERIMENTS_DIR,
    )
    exp_dir = tracker.run_dir
    checkpoint_dir = exp_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # 7. Initial Baseline Validation
    print("\nRunning initial baseline SFT validation...")
    init_val_loss, init_val_ppl, init_val_bpb = evaluate_sft(
        model, val_loader, device, amp_ctx, max_batches=15 if smoke_test else 40
    )
    print(f"  [Initial SFT Baseline] val_loss={init_val_loss:.4f}  ppl={init_val_ppl:.2f}  bpb={init_val_bpb:.4f}")

    best_val_loss = init_val_loss
    opt_step = 0
    micro_step = 0
    current_epoch = 0
    train_iter = iter(train_loader)
    total_valid_tokens_processed = 0

    model.train()
    optimizer.zero_grad(set_to_none=True)
    if device_str == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    print(f"\n{'─'*70}")
    print(f"  Starting SFT Training: step 0 -> {max_steps}")
    print(f"{'─'*70}\n")

    t_train_start = time.perf_counter()

    # 8. Main SFT Training Loop
    while opt_step < max_steps:
        step_start_time = time.perf_counter()
        accum_loss = 0.0
        accum_ce_loss = 0.0
        accum_z_loss = 0.0
        step_valid_tokens = 0

        for _ in range(grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                current_epoch += 1
                train_sampler.set_epoch(current_epoch)
                train_iter = iter(train_loader)
                x, y = next(train_iter)

            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            valid_mask = (y != IGNORE_INDEX)
            num_valid = valid_mask.sum().item()
            step_valid_tokens += num_valid

            with amp_ctx:
                logits, _ = model(x)

                # Next-token loss on response tokens only
                ce_loss = F.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    y.view(-1),
                    ignore_index=IGNORE_INDEX,
                )

                # Z-loss logit regularization
                if z_loss_weight > 0.0 and num_valid > 0:
                    log_z = torch.logsumexp(logits, dim=-1).float()
                    z_loss = z_loss_weight * (log_z[valid_mask] ** 2).mean()
                    total_loss = ce_loss + z_loss
                else:
                    z_loss = torch.tensor(0.0, device=logits.device)
                    total_loss = ce_loss

            (total_loss / grad_accum_steps).backward()

            accum_loss += total_loss.item()
            accum_ce_loss += ce_loss.item()
            accum_z_loss += z_loss.item()
            micro_step += 1

        # Optimizer step
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        opt_step += 1
        current_lr = scheduler.get_lr()
        total_valid_tokens_processed += step_valid_tokens

        step_elapsed = time.perf_counter() - step_start_time
        tok_per_sec = step_valid_tokens / max(step_elapsed, 1e-6)

        avg_step_loss = accum_loss / grad_accum_steps
        avg_ce_loss = accum_ce_loss / grad_accum_steps
        avg_z_loss = accum_z_loss / grad_accum_steps

        # Logging
        if opt_step == 1 or opt_step % log_interval == 0:
            hw = SystemMonitor.get_current_metrics()
            epoch_progress = opt_step / max(steps_per_epoch, 1e-6)
            print(
                f"step {opt_step:4d} | loss {avg_step_loss:.4f} (ce {avg_ce_loss:.4f}, z {avg_z_loss:.4f}) | "
                f"lr {current_lr:.2e} | grad {grad_norm:.3f} | "
                f"{tok_per_sec/1e3:4.1f}k tok/s | ep {epoch_progress:.2f} | "
                f"VRAM {hw['gpu_reserved_mb']:.0f}MB"
            )
            tracker.log_step(
                step=opt_step,
                train_loss=avg_step_loss,
                lr=current_lr,
                grad_norm=float(grad_norm),
                tokens_in_step=step_valid_tokens,
                total_tokens=total_valid_tokens_processed,
                ce_loss=round(float(avg_ce_loss), 4),
                z_loss=round(float(avg_z_loss), 5),
                epoch_progress=round(epoch_progress, 4),
            )

        # Validation & Checkpointing
        if opt_step % eval_interval == 0 or opt_step == max_steps:
            t_eval = time.perf_counter()
            val_loss, val_ppl, val_bpb = evaluate_sft(
                model,
                val_loader,
                device,
                amp_ctx,
                max_batches=20 if smoke_test else None,
            )
            eval_elapsed = time.perf_counter() - t_eval

            if device_str == "cuda":
                torch.cuda.empty_cache()

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss

            print(
                f"  [EVAL step {opt_step:4d}] "
                f"val_loss={val_loss:.4f}  ppl={val_ppl:.2f}  bpb={val_bpb:.4f}  "
                f"({'NEW BEST' if is_best else 'no improvement'})  [{eval_elapsed:.1f}s]"
            )

            # Generate sample answers
            samples = generate_sample_answers(model, tokenizer, PROMPT_SAMPLES, device)
            print("  [Sample Assistant Generations]")
            for s in samples[:2]:
                print(f"    {s.replace(chr(10), ' | ')}")

            tracker.log_eval(
                step=opt_step,
                val_loss=val_loss,
                samples=samples,
                perplexity=val_ppl,
                bpb=val_bpb,
                is_best=is_best,
            )

            # Save checkpoint
            if opt_step % save_interval == 0 or is_best or opt_step == max_steps:
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    opt_step=opt_step,
                    micro_step=micro_step,
                    train_loss=avg_step_loss,
                    val_loss=val_loss,
                    config_snapshot={
                        "model_name": "GPT-98.4M-SFT",
                        "base_checkpoint": str(base_path),
                        "opt_step": opt_step,
                        "val_loss": val_loss,
                        "n_layer": config.N_LAYER,
                        "n_head": config.N_HEAD,
                        "n_embd": config.N_EMBD,
                        "ffn_dim": config.FFN_DIM,
                        "block_size": config.BLOCK_SIZE,
                        "vocab_size": config.VOCAB_SIZE,
                        "bias": config.BIAS,
                        "tie_weights": config.TIE_WEIGHTS,
                    },
                    tokenizer_config_path=config.TOKENIZER_CONFIG_FILE,
                    dataset_meta_paths={
                        "train": train_jsonl,
                        "val": val_jsonl,
                    },
                    is_best=is_best,
                )
                print(f"  [Checkpoint] Saved -> {checkpoint_dir.name} (best={is_best})")

    total_time = time.perf_counter() - t_train_start
    print(f"\n{'='*72}")
    print(f"  SFT Training Finished! Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
    print(f"  Best Val Loss: {best_val_loss:.4f}")
    print(f"  Checkpoints saved in: {checkpoint_dir}")
    print(f"{'='*72}\n")


def main():
    parser = argparse.ArgumentParser(description="GPT SFT Training Engine (Phase 8)")
    parser.add_argument("--base-checkpoint", type=str, default=str(DEFAULT_BASE_CHECKPOINT), help="Path to base pretrained checkpoint.")
    parser.add_argument("--smoke-test", action="store_true", help="Run 25-step quick verification smoke test.")
    parser.add_argument("--max-steps", type=int, default=None, help="Total optimizer steps override.")
    parser.add_argument("--peak-lr", type=float, default=None, help="Peak learning rate override.")
    parser.add_argument("--min-lr", type=float, default=None, help="Minimum learning rate override.")
    parser.add_argument("--warmup-steps", type=int, default=None, help="Warmup steps override.")
    parser.add_argument("--eval-interval", type=int, default=None, help="Evaluation interval override.")
    parser.add_argument("--save-interval", type=int, default=None, help="Save interval override.")
    args = parser.parse_args()

    train_sft(
        base_checkpoint=args.base_checkpoint,
        smoke_test=args.smoke_test,
        max_steps_override=args.max_steps,
        peak_lr_override=args.peak_lr,
        min_lr_override=args.min_lr,
        warmup_steps_override=args.warmup_steps,
        eval_interval_override=args.eval_interval,
        save_interval_override=args.save_interval,
    )


if __name__ == "__main__":
    main()
