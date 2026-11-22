"""Phase 4: Direct Preference Optimization (DPO) Training Engine for GPT-98.4M.

Features:
- Policy & Reference initialized from frozen SFT checkpoint.
- Float32 log_softmax over response tokens only (assistant tokens including terminal <|eos|>).
- Precomputes and caches reference log-probs to disk (0 reference VRAM overhead).
- Dropout set to 0 for policy during DPO (eval-mode dropout behavior).
- Loss: L = -logsigmoid(beta * ((pi_c - ref_c) - (pi_r - ref_r))) + alpha * NLL_c.
- Step 0 sanity check (loss ~ 0.6931, accuracy ~ 50%, margin ~ 0).
- 16-pair overfit verification mode (--overfit-test).
- TensorBoard logging every 10 steps, held-out evaluation every 25 steps.
- Saves best.pt and last.pt in timestamped experiment directories.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, List, Optional, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig
from monitoring.experiment_tracker import ExperimentTracker
from training.checkpointing import save_checkpoint
from training.optimizer import configure_adamw
from training.scheduler import CosineScheduler

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0
IGNORE_INDEX = -100

DEFAULT_SFT_CHECKPOINT = (
    PROJECT_ROOT
    / "experiments"
    / "20261121_174835_gpt_multi_turn_sft"
    / "checkpoints"
    / "best.pt"
)


def format_tokens_and_mask(
    tokenizer: Tokenizer,
    prompt_messages: List[Dict[str, str]],
    response_text: str,
    max_seq_len: int = 512,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize prompt and response piece-wise with exact IGNORE_INDEX mask on prompt."""
    user_hdr_ids = tokenizer.encode("### User:\n").ids
    asst_hdr_ids = tokenizer.encode("### Assistant:\n").ids
    sep_ids = tokenizer.encode("\n\n").ids

    full_inputs = [BOS_ID]
    full_targets = [IGNORE_INDEX]

    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            turn_ids = user_hdr_ids + tokenizer.encode(f"{content}\n\n").ids
            full_inputs.extend(turn_ids)
            full_targets.extend([IGNORE_INDEX] * len(turn_ids))
        elif role == "assistant":
            turn_ids = asst_hdr_ids + tokenizer.encode(content).ids + [EOS_ID] + sep_ids
            full_inputs.extend(turn_ids)
            full_targets.extend([IGNORE_INDEX] * len(turn_ids))

    full_inputs.extend(asst_hdr_ids)
    full_targets.extend([IGNORE_INDEX] * len(asst_hdr_ids))

    clean_resp = response_text.strip()
    resp_tokens = tokenizer.encode(clean_resp).ids + [EOS_ID]
    full_inputs.extend(resp_tokens)
    full_targets.extend(resp_tokens)

    # Truncate over-long prompts from the left if exceeding max_seq_len + 1
    if len(full_inputs) > max_seq_len + 1:
        excess = len(full_inputs) - (max_seq_len + 1)
        full_inputs = [full_inputs[0]] + full_inputs[1 + excess :]
        full_targets = [full_targets[0]] + full_targets[1 + excess :]

    x = torch.tensor(full_inputs[:-1], dtype=torch.long)
    y = torch.tensor(full_targets[1:], dtype=torch.long)
    return x, y


class DPODataset(Dataset):
    """Dataset of DPO pairs with precomputed reference log-probs."""

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer: Tokenizer,
        ref_logprobs_cache: Optional[Dict[str, Dict[str, float]]] = None,
        max_seq_len: int = 512,
        limit: Optional[int] = None,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.pairs = []

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.pairs.append(json.loads(line))
                    if limit and len(self.pairs) >= limit:
                        break

        self.ref_cache = ref_logprobs_cache or {}

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        p = self.pairs[idx]
        pair_id = p["id"]
        msgs = p["prompt_messages"]

        c_x, c_y = format_tokens_and_mask(self.tokenizer, msgs, p["chosen"], self.max_seq_len)
        r_x, r_y = format_tokens_and_mask(self.tokenizer, msgs, p["rejected"], self.max_seq_len)

        ref_c = self.ref_cache.get(pair_id, {}).get("ref_c", 0.0)
        ref_r = self.ref_cache.get(pair_id, {}).get("ref_r", 0.0)

        return {
            "id": pair_id,
            "c_x": c_x,
            "c_y": c_y,
            "r_x": r_x,
            "r_y": r_y,
            "ref_c": ref_c,
            "ref_r": ref_r,
        }


def dpo_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Collate function that dynamically pads chosen and rejected micro-batches."""
    max_c_len = max(len(item["c_x"]) for item in batch)
    max_r_len = max(len(item["r_x"]) for item in batch)

    # Pad up to multiple of 8
    pad_c_len = ((max_c_len + 7) // 8) * 8
    pad_r_len = ((max_r_len + 7) // 8) * 8

    batch_c_x, batch_c_y = [], []
    batch_r_x, batch_r_y = [], []
    batch_ref_c, batch_ref_r = [], []
    batch_ids = []

    for item in batch:
        batch_ids.append(item["id"])
        batch_ref_c.append(item["ref_c"])
        batch_ref_r.append(item["ref_r"])

        # Chosen padding
        cx = item["c_x"]
        cy = item["c_y"]
        if len(cx) < pad_c_len:
            pad_n = pad_c_len - len(cx)
            cx = torch.cat([cx, torch.full((pad_n,), PAD_ID, dtype=torch.long)])
            cy = torch.cat([cy, torch.full((pad_n,), IGNORE_INDEX, dtype=torch.long)])
        batch_c_x.append(cx)
        batch_c_y.append(cy)

        # Rejected padding
        rx = item["r_x"]
        ry = item["r_y"]
        if len(rx) < pad_r_len:
            pad_n = pad_r_len - len(rx)
            rx = torch.cat([rx, torch.full((pad_n,), PAD_ID, dtype=torch.long)])
            ry = torch.cat([ry, torch.full((pad_n,), IGNORE_INDEX, dtype=torch.long)])
        batch_r_x.append(rx)
        batch_r_y.append(ry)

    return {
        "ids": batch_ids,
        "c_x": torch.stack(batch_c_x),
        "c_y": torch.stack(batch_c_y),
        "r_x": torch.stack(batch_r_x),
        "r_y": torch.stack(batch_r_y),
        "ref_c": torch.tensor(batch_ref_c, dtype=torch.float32),
        "ref_r": torch.tensor(batch_ref_r, dtype=torch.float32),
    }


def compute_sequence_logprobs(
    model: GPT,
    x: torch.Tensor,
    y: torch.Tensor,
    amp_ctx,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute exact float32 sum of token log-probabilities over active response tokens.
    
    Returns:
        seq_logprobs: Tensor [B] of sum of response token log-probs.
        response_lengths: Tensor [B] of response token counts.
    """
    with amp_ctx:
        logits, _ = model(x)

    # Float32 log_softmax for high numerical stability
    logits_f32 = logits.float()
    log_probs = F.log_softmax(logits_f32, dim=-1)

    # Gather log-probs for target tokens
    y_clamped = y.clamp(min=0)
    target_log_probs = log_probs.gather(-1, y_clamped.unsqueeze(-1)).squeeze(-1)

    # Mask out prompt tokens (where y == -100)
    mask = (y != IGNORE_INDEX).float()
    seq_logprobs = (target_log_probs * mask).sum(dim=-1)
    response_lengths = mask.sum(dim=-1).clamp(min=1.0)

    return seq_logprobs, response_lengths


@torch.no_grad()
def precompute_reference_logprobs(
    model: GPT,
    jsonl_path: Path,
    tokenizer: Tokenizer,
    cache_path: Path,
    batch_size: int = 4,
    amp_ctx = None,
) -> Dict[str, Dict[str, float]]:
    """Precompute reference log-probs once and cache them to disk."""
    if cache_path.exists():
        print(f"[Cache] Loading precomputed reference log-probs from: {cache_path}")
        return torch.load(str(cache_path), map_location="cpu", weights_only=False)

    print(f"\n[Precompute] Precomputing reference log-probs for: {jsonl_path.name}...")
    dataset = DPODataset(jsonl_path, tokenizer, max_seq_len=512)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=dpo_collate_fn)

    model.eval()
    cache: Dict[str, Dict[str, float]] = {}
    t0 = time.perf_counter()

    for idx, batch in enumerate(loader):
        c_x = batch["c_x"].to(DEVICE, non_blocking=True)
        c_y = batch["c_y"].to(DEVICE, non_blocking=True)
        r_x = batch["r_x"].to(DEVICE, non_blocking=True)
        r_y = batch["r_y"].to(DEVICE, non_blocking=True)

        logp_c, _ = compute_sequence_logprobs(model, c_x, c_y, amp_ctx)
        logp_r, _ = compute_sequence_logprobs(model, r_x, r_y, amp_ctx)

        for b_i, p_id in enumerate(batch["ids"]):
            cache[p_id] = {
                "ref_c": float(logp_c[b_i].cpu().item()),
                "ref_r": float(logp_r[b_i].cpu().item()),
            }

        if (idx + 1) % 50 == 0 or (idx + 1) == len(loader):
            elapsed = time.perf_counter() - t0
            print(f"  Processed {idx + 1}/{len(loader)} batches ({len(cache)} pairs) in {elapsed:.1f}s")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(cache, str(cache_path))
    print(f"[Cache] Saved precomputed reference log-probs to: {cache_path}")
    return cache


@torch.no_grad()
def evaluate_dpo(
    model: GPT,
    loader: DataLoader,
    beta: float,
    amp_ctx,
    nll_alpha: float = 0.0,
    max_batches: Optional[int] = None,
) -> Dict[str, float]:
    """Evaluate DPO loss, reward accuracy, and reward margins on held-out pairs."""
    model.eval()
    total_loss = 0.0
    total_acc = 0.0
    total_margin = 0.0
    total_pi_c = 0.0
    total_pi_r = 0.0
    total_pairs = 0

    for idx, batch in enumerate(loader):
        if max_batches and idx >= max_batches:
            break

        c_x = batch["c_x"].to(DEVICE, non_blocking=True)
        c_y = batch["c_y"].to(DEVICE, non_blocking=True)
        r_x = batch["r_x"].to(DEVICE, non_blocking=True)
        r_y = batch["r_y"].to(DEVICE, non_blocking=True)
        ref_c = batch["ref_c"].to(DEVICE, non_blocking=True)
        ref_r = batch["ref_r"].to(DEVICE, non_blocking=True)

        pi_c, c_lens = compute_sequence_logprobs(model, c_x, c_y, amp_ctx)
        pi_r, _ = compute_sequence_logprobs(model, r_x, r_y, amp_ctx)

        delta_c = pi_c - ref_c
        delta_r = pi_r - ref_r

        margin = beta * (delta_c - delta_r)
        dpo_loss = -F.logsigmoid(margin)

        if nll_alpha > 0.0:
            nll_c = -(pi_c / c_lens)
            loss = dpo_loss + nll_alpha * nll_c
        else:
            loss = dpo_loss

        b_size = len(c_x)
        total_loss += loss.sum().item()
        # Pairwise accuracy with tie-breaking: margin > 0 is win (1.0), margin == 0 is tie (0.5)
        pair_acc = (margin > 0.0).float() + 0.5 * (margin == 0.0).float()
        total_acc += pair_acc.sum().item()
        total_margin += margin.sum().item()
        total_pi_c += pi_c.sum().item()
        total_pi_r += pi_r.sum().item()
        total_pairs += b_size

    model.train()
    if total_pairs == 0:
        return {}

    return {
        "val_loss": round(total_loss / total_pairs, 4),
        "val_accuracy": round((total_acc / total_pairs) * 100.0, 2),
        "val_margin": round(total_margin / total_pairs, 4),
        "val_pi_c": round(total_pi_c / total_pairs, 2),
        "val_pi_r": round(total_pi_r / total_pairs, 2),
    }


def train_dpo_run(
    experiment_name: str,
    lr: float = 3e-6,
    beta: float = 0.1,
    nll_alpha: float = 0.0,
    epochs: int = 1,
    micro_batch_size: int = 4,
    effective_batch_size: int = 32,
    overfit_test: bool = False,
    sft_checkpoint_path: Path = DEFAULT_SFT_CHECKPOINT,
) -> Dict[str, Any]:
    print("=" * 72)
    print(f"  DPO TRAINING RUN: {experiment_name}")
    print(f"  LR: {lr:.2e} | Beta: {beta} | NLL Alpha: {nll_alpha} | Overfit Test: {overfit_test}")
    print("=" * 72)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16) if DEVICE == "cuda" else torch.amp.autocast(device_type="cpu", enabled=False)

    # 1. Load Architecture & Initialize Model from SFT Checkpoint
    ckpt = torch.load(str(sft_checkpoint_path), map_location="cpu", weights_only=False)
    saved_cfg = ckpt["config_snapshot"]

    # Rule: Set dropout to 0.0 for DPO (so policy matches reference at step 0)
    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=saved_cfg.get("block_size", config.BLOCK_SIZE),
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )
    model = GPT(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    print(f"[Model] Initialized policy from SFT checkpoint: {sft_checkpoint_path.name}")
    del ckpt
    torch.cuda.empty_cache()

    # 2. Datasets & Reference Precomputation
    train_jsonl = PROJECT_ROOT / "data" / "dpo" / "dpo_train_pairs.jsonl"
    val_jsonl = PROJECT_ROOT / "data" / "dpo" / "dpo_val_pairs.jsonl"
    cache_dir = PROJECT_ROOT / "data" / "dpo" / "cache"

    ref_train_cache = precompute_reference_logprobs(
        model, train_jsonl, tokenizer, cache_dir / "ref_logprobs_train.pt", batch_size=micro_batch_size, amp_ctx=amp_ctx
    )
    ref_val_cache = precompute_reference_logprobs(
        model, val_jsonl, tokenizer, cache_dir / "ref_logprobs_val.pt", batch_size=micro_batch_size, amp_ctx=amp_ctx
    )

    limit = 16 if overfit_test else None
    train_ds = DPODataset(train_jsonl, tokenizer, ref_train_cache, max_seq_len=512, limit=limit)
    val_ds = DPODataset(val_jsonl, tokenizer, ref_val_cache, max_seq_len=512, limit=limit)

    train_loader = DataLoader(train_ds, batch_size=micro_batch_size, shuffle=True, collate_fn=dpo_collate_fn)
    val_loader = DataLoader(val_ds, batch_size=micro_batch_size, shuffle=False, collate_fn=dpo_collate_fn)

    grad_accum_steps = max(1, effective_batch_size // micro_batch_size)
    total_steps = 30 if overfit_test else math.ceil((len(train_ds) * epochs) / effective_batch_size)
    warmup_steps = 2 if overfit_test else max(10, int(total_steps * 0.10))

    # 3. Optimizer & Scheduler
    optimizer = configure_adamw(
        model,
        learning_rate=lr,
        weight_decay=0.0,
        betas=(0.9, 0.95),
        device_type=DEVICE,
    )
    scheduler = CosineScheduler(
        optimizer,
        warmup_steps=warmup_steps,
        max_steps=total_steps,
        peak_lr=lr,
        min_lr=lr * 0.10,
    )

    # 4. Tracker & Checkpointing
    tracker = ExperimentTracker(
        experiment_name=experiment_name,
        config_dict={
            "mode": "dpo",
            "lr": lr,
            "beta": beta,
            "nll_alpha": nll_alpha,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "grad_accum_steps": grad_accum_steps,
            "micro_batch_size": micro_batch_size,
            "train_pairs": len(train_ds),
            "val_pairs": len(val_ds),
            "overfit_test": overfit_test,
        },
        base_dir=config.EXPERIMENTS_DIR,
    )
    ckpt_dir = tracker.run_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 5. Step 0 Sanity Check
    print("\n[Sanity Check] Verifying Step 0 DPO baseline invariants...")
    step0_metrics = evaluate_dpo(model, val_loader, beta, amp_ctx, nll_alpha, max_batches=10)
    print(
        f"  Step 0: Loss={step0_metrics['val_loss']} | "
        f"Accuracy={step0_metrics['val_accuracy']}% (Target: ~50.0%) | "
        f"Margin={step0_metrics['val_margin']} (Target: ~0.0)"
    )

    if nll_alpha == 0.0:
        assert abs(step0_metrics["val_loss"] - 0.6931) < 0.05, f"Step 0 loss sanity failed: {step0_metrics['val_loss']}"
    else:
        assert step0_metrics["val_loss"] >= 0.6931, f"Step 0 loss with NLL regularization failed: {step0_metrics['val_loss']}"
    assert abs(step0_metrics["val_accuracy"] - 50.0) <= 15.0, f"Step 0 accuracy sanity failed: {step0_metrics['val_accuracy']}"
    print("--> PASS: Step 0 mathematical invariants verified!\n")

    # 6. Training Loop
    best_val_acc = step0_metrics["val_accuracy"]
    best_val_loss = step0_metrics["val_loss"]
    opt_step = 0
    micro_step = 0
    train_iter = iter(train_loader)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    t_start = time.perf_counter()

    while opt_step < total_steps:
        accum_loss = 0.0
        accum_acc = 0.0
        accum_margin = 0.0
        accum_pi_c = 0.0
        accum_pi_r = 0.0

        for _ in range(grad_accum_steps):
            try:
                batch = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                batch = next(train_iter)

            c_x = batch["c_x"].to(DEVICE, non_blocking=True)
            c_y = batch["c_y"].to(DEVICE, non_blocking=True)
            r_x = batch["r_x"].to(DEVICE, non_blocking=True)
            r_y = batch["r_y"].to(DEVICE, non_blocking=True)
            ref_c = batch["ref_c"].to(DEVICE, non_blocking=True)
            ref_r = batch["ref_r"].to(DEVICE, non_blocking=True)

            pi_c, c_lens = compute_sequence_logprobs(model, c_x, c_y, amp_ctx)
            pi_r, _ = compute_sequence_logprobs(model, r_x, r_y, amp_ctx)

            delta_c = pi_c - ref_c
            delta_r = pi_r - ref_r
            margin = beta * (delta_c - delta_r)

            dpo_loss = -F.logsigmoid(margin).mean()

            if nll_alpha > 0.0:
                nll_loss = (-(pi_c / c_lens)).mean()
                total_loss = dpo_loss + nll_alpha * nll_loss
            else:
                total_loss = dpo_loss

            (total_loss / grad_accum_steps).backward()

            accum_loss += total_loss.item()
            pair_acc = (margin > 0.0).float() + 0.5 * (margin == 0.0).float()
            accum_acc += pair_acc.mean().item()
            accum_margin += margin.mean().item()
            accum_pi_c += pi_c.mean().item()
            accum_pi_r += pi_r.mean().item()
            micro_step += 1

        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        opt_step += 1
        cur_lr = scheduler.get_lr()

        step_loss = accum_loss / grad_accum_steps
        step_acc = (accum_acc / grad_accum_steps) * 100.0
        step_margin = accum_margin / grad_accum_steps
        step_pi_c = accum_pi_c / grad_accum_steps
        step_pi_r = accum_pi_r / grad_accum_steps

        # Check for policy collapse failure mode
        if opt_step >= 10 and step_pi_c < -400.0:
            print(f"[ALERT] Step {opt_step}: Chosen log-prob collapsed sharply to {step_pi_c:.1f}! Stopping run.")
            break

        # Log to TensorBoard every 10 steps (or every step in overfit test)
        log_interval = 2 if overfit_test else 10
        if opt_step % log_interval == 0 or opt_step == 1 or opt_step == total_steps:
            print(
                f"step {opt_step:3d}/{total_steps} | loss {step_loss:.4f} | "
                f"acc {step_acc:5.1f}% | margin {step_margin:+.4f} | "
                f"pi_c {step_pi_c:6.1f} | pi_r {step_pi_r:6.1f} | "
                f"grad {grad_norm:.3f} | lr {cur_lr:.2e}"
            )
            tracker.tb_logger.log_scalars({
                "dpo/loss": step_loss,
                "dpo/reward_accuracy": step_acc,
                "dpo/reward_margin": step_margin,
                "dpo/pi_chosen_logprob": step_pi_c,
                "dpo/pi_rejected_logprob": step_pi_r,
                "dpo/grad_norm": float(grad_norm),
                "dpo/lr": cur_lr,
            }, opt_step)

        # Held-out evaluation every 25 steps (or every 5 steps in overfit test)
        eval_interval = 5 if overfit_test else 25
        if opt_step % eval_interval == 0 or opt_step == total_steps:
            val_res = evaluate_dpo(model, val_loader, beta, amp_ctx, nll_alpha)
            is_best = val_res["val_accuracy"] >= best_val_acc
            if is_best:
                best_val_acc = val_res["val_accuracy"]
                best_val_loss = val_res["val_loss"]

            print(
                f"  [EVAL step {opt_step:3d}] "
                f"val_loss={val_res['val_loss']:.4f} | val_acc={val_res['val_accuracy']:.1f}% | "
                f"val_margin={val_res['val_margin']:+.4f} | ({'NEW BEST' if is_best else 'no improvement'})"
            )

            tracker.tb_logger.log_scalars({
                "eval/val_loss": val_res["val_loss"],
                "eval/val_accuracy": val_res["val_accuracy"],
                "eval/val_margin": val_res["val_margin"],
            }, opt_step)

            if not overfit_test:
                dpo_cfg_snapshot = dict(saved_cfg)
                dpo_cfg_snapshot.update({
                    "model_name": f"GPT-98.4M-DPO-{experiment_name}",
                    "opt_step": opt_step,
                    "val_loss": val_res["val_loss"],
                    "val_accuracy": val_res["val_accuracy"],
                    "lr": lr,
                    "beta": beta,
                    "nll_alpha": nll_alpha,
                })
                save_checkpoint(
                    checkpoint_dir=ckpt_dir,
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    opt_step=opt_step,
                    micro_step=micro_step,
                    train_loss=step_loss,
                    val_loss=val_res["val_loss"],
                    config_snapshot=dpo_cfg_snapshot,
                    tokenizer_config_path=config.TOKENIZER_CONFIG_FILE,
                    dataset_meta_paths={"train": train_jsonl, "val": val_jsonl},
                    is_best=is_best,
                )

    if not overfit_test:
        if not (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists():
            import shutil
            shutil.copy2(ckpt_dir / "last.pt", ckpt_dir / "best.pt")

    total_time = time.perf_counter() - t_start
    print(f"\nRun {experiment_name} finished in {total_time:.1f}s ({total_time/60:.1f}m).")
    print(f"Best Val Accuracy: {best_val_acc:.1f}% | Final Loss: {best_val_loss:.4f}\n")

    try:
        tracker.finish()
    except Exception as e:
        print(f"[Warning] tracker.finish() warning: {e}")

    return {
        "experiment_name": experiment_name,
        "best_val_accuracy": best_val_acc,
        "best_val_loss": best_val_loss,
        "total_time_sec": round(total_time, 1),
        "run_dir": str(tracker.run_dir),
    }


def main():
    parser = argparse.ArgumentParser(description="DPO Training Engine")
    parser.add_argument("--exp-name", type=str, default="dpo_run", help="Experiment name")
    parser.add_argument("--lr", type=float, default=3e-6, help="Learning rate")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta temperature")
    parser.add_argument("--nll_alpha", type=float, default=0.0, help="NLL regularization weight on chosen")
    parser.add_argument("--overfit-test", action="store_true", help="Run 16-pair sanity overfit test")
    args = parser.parse_args()

    train_dpo_run(
        experiment_name=args.exp_name,
        lr=args.lr,
        beta=args.beta,
        nll_alpha=args.nll_alpha,
        overfit_test=args.overfit_test,
    )


if __name__ == "__main__":
    main()
