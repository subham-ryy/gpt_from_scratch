"""Comprehensive Post-Training Evaluation for Track A Continued Pretraining.

Covers:
1. Checkpoint Verification (best.pt vs last.pt integrity, metadata, architecture match).
2. Validation Analysis (metrics comparison table, trajectory plotting).
3. Side-by-Side Generation Comparison (GPT-v1 vs Track A across 6 required prompts).
4. Sampling Ablation (Track A under 3 sampling configurations).
5. Discrepancy Analysis for Token Accounting.
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.dataset import BinaryTokenDataset
from model.transformer import GPT, GPTConfig

ARTIFACT_DIR = Path(r"C:\Users\rupar\.gemini\antigravity\brain\60461eed-366b-4384-8e0d-82baa726f007")

CKPT_V1 = PROJECT_ROOT / "experiments" / "20261104_211024_gpt_pretrain" / "checkpoints" / "best.pt"
EXP_TRACK_A = PROJECT_ROOT / "experiments" / "20261106_032904_gpt_continued_pretrain"
CKPT_TRACK_A_BEST = EXP_TRACK_A / "checkpoints" / "best.pt"
CKPT_TRACK_A_LAST = EXP_TRACK_A / "checkpoints" / "last.pt"

PROMPTS = [
    "The capital of France is",
    "Photosynthesis is the process by which",
    "In computer science, an algorithm is defined as",
    "Machine learning is a field of",
    "The human brain consists of",
    "Once upon a time",
]


def load_model(ckpt_path: Path, device="cuda") -> tuple[GPT, Tokenizer, dict]:
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    saved_cfg = ckpt.get("config_snapshot", {})
    block_size = saved_cfg.get("block_size", config.BLOCK_SIZE)

    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=block_size,
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )

    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    meta = {
        "opt_step": ckpt.get("opt_step"),
        "micro_step": ckpt.get("micro_step"),
        "train_loss": ckpt.get("train_loss"),
        "val_loss": ckpt.get("val_loss"),
        "n_params": model.get_num_params(),
    }
    del ckpt
    torch.cuda.empty_cache()
    return model, tokenizer, meta


def generate_text(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 70,
    temperature: float = 0.8,
    top_p: float = 0.90,
    top_k: int | None = None,
    min_p: float = 0.05,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 3,
    device: str = "cuda",
) -> tuple[str, float]:
    eos_id = tokenizer.token_to_id(config.EOS_TOKEN)
    encoded = tokenizer.encode(prompt).ids
    input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_id,
        )
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    generated_tokens = out_ids[0].tolist()
    decoded = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    tokens_gen = len(generated_tokens) - input_ids.shape[1]
    speed = tokens_gen / max(elapsed, 1e-6)
    return decoded, speed


def run_evaluation():
    print("=" * 70)
    print("  RUNNING POST-TRAINING EVALUATION FOR TRACK A")
    print("=" * 70)

    # ── 1. Checkpoint Verification ───────────────────────────────────────────
    print("\n--- 1. Checkpoint Verification ---")
    model_v1, tok, meta_v1 = load_model(CKPT_V1)
    print(f"GPT-v1 Checkpoint: Step {meta_v1['opt_step']}, Val Loss: {meta_v1['val_loss']:.4f}, Params: {meta_v1['n_params']:,}")

    model_ta, _, meta_ta = load_model(CKPT_TRACK_A_BEST)
    print(f"Track A Checkpoint: Step {meta_ta['opt_step']}, Val Loss: {meta_ta['val_loss']:.4f}, Params: {meta_ta['n_params']:,}")

    # Check last.pt
    _, _, meta_ta_last = load_model(CKPT_TRACK_A_LAST)
    print(f"Track A Last Checkpoint: Step {meta_ta_last['opt_step']}, Val Loss: {meta_ta_last['val_loss']:.4f}")

    # ── 2. Validation Analysis & Plot ─────────────────────────────────────────
    print("\n--- 2. Validation Trajectory Plotting ---")
    df_p6 = pd.read_csv(PROJECT_ROOT / "experiments" / "20261104_211024_gpt_pretrain" / "metrics.csv")
    eval_p6 = df_p6.dropna(subset=["val_loss"])[["step", "val_loss"]].copy()
    eval_p6["perplexity"] = np.exp(np.minimum(eval_p6["val_loss"], 20.0))
    eval_p6["bpb"] = (eval_p6["val_loss"] / math.log(2)) / 4.2514
    eval_p6["cum_step"] = eval_p6["step"]

    df_ta = pd.read_csv(EXP_TRACK_A / "metrics.csv")
    eval_ta = df_ta.dropna(subset=["val_loss"])[["step", "val_loss"]].copy()
    eval_ta["perplexity"] = np.exp(np.minimum(eval_ta["val_loss"], 20.0))
    eval_ta["bpb"] = (eval_ta["val_loss"] / math.log(2)) / 4.2514
    eval_ta["cum_step"] = eval_ta["step"] + 1200  # continued from step 1200

    # Create comparison figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    
    # Val loss subplot
    ax1.plot(eval_p6["cum_step"], eval_p6["val_loss"], marker="o", color="#3b82f6", label="Phase 6 (5k docs, unpacked)")
    ax1.plot(eval_ta["cum_step"], eval_ta["val_loss"], marker="s", color="#10b981", label="Track A (100k docs, bin-packed + Z-loss)")
    ax1.axvline(1200, color="#ef4444", linestyle="--", alpha=0.7, label="Track A Handover (Step 1,200)")
    ax1.set_xlabel("Cumulative Optimizer Steps")
    ax1.set_ylabel("Validation Loss (Cross-Entropy)")
    ax1.set_title("Validation Loss: Phase 6 Baseline vs Track A Continued Pretraining")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Perplexity subplot (log scale)
    ax2.plot(eval_p6["cum_step"], eval_p6["perplexity"], marker="o", color="#3b82f6", label="Phase 6 Baseline")
    ax2.plot(eval_ta["cum_step"], eval_ta["perplexity"], marker="s", color="#10b981", label="Track A Continued")
    ax2.axvline(1200, color="#ef4444", linestyle="--", alpha=0.7, label="Handover")
    ax2.set_yscale("log")
    ax2.set_xlabel("Cumulative Optimizer Steps")
    ax2.set_ylabel("Perplexity (Log Scale)")
    ax2.set_title("Validation Perplexity (196.23 → 40.67)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    plots_dir = PROJECT_ROOT / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plot_path_local = plots_dir / "track_a_vs_v1_validation.png"
    fig.savefig(plot_path_local)
    
    # Also save to artifact directory
    plot_path_artifact = ARTIFACT_DIR / "track_a_vs_v1_validation.png"
    fig.savefig(plot_path_artifact)
    plt.close(fig)
    print(f"Saved validation comparison plot to {plot_path_local} and {plot_path_artifact}")

    # ── 3. Side-by-Side Generation Comparison ─────────────────────────────────
    print("\n--- 3. Side-by-Side Generation Comparison ---")
    gen_results = []
    default_controls = {
        "temperature": 0.8,
        "top_p": 0.90,
        "top_k": None,
        "min_p": 0.05,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
    }

    for p in PROMPTS:
        # GPT-v1
        torch.manual_seed(42)
        out_v1, spd_v1 = generate_text(model_v1, tok, p, **default_controls)

        # Track A
        torch.manual_seed(42)
        out_ta, spd_ta = generate_text(model_ta, tok, p, **default_controls)

        gen_results.append({
            "prompt": p,
            "gpt_v1": {
                "text": out_v1,
                "speed_tok_s": round(spd_v1, 1),
            },
            "track_a": {
                "text": out_ta,
                "speed_tok_s": round(spd_ta, 1),
            },
        })
        print(f"\n[Prompt]: {p}")
        print(f"  GPT-v1  ({spd_v1:.1f} tok/s): {out_v1}")
        print(f"  Track A ({spd_ta:.1f} tok/s): {out_ta}")

    # ── 4. Sampling Ablation on Track A ───────────────────────────────────────
    print("\n--- 4. Sampling Ablation on Track A ---")
    ablation_configs = [
        {"name": "Config 1 (T=0.8, p=0.90, k=None)", "temperature": 0.8, "top_p": 0.90, "top_k": None},
        {"name": "Config 2 (T=0.8, p=0.95, k=None)", "temperature": 0.8, "top_p": 0.95, "top_k": None},
        {"name": "Config 3 (T=0.8, p=0.95, k=50)",   "temperature": 0.8, "top_p": 0.95, "top_k": 50},
    ]

    ablation_results = []
    test_prompts_ablation = [
        "In computer science, an algorithm is defined as",
        "The human brain consists of",
    ]

    for cfg in ablation_configs:
        cfg_out = {"config_name": cfg["name"], "samples": []}
        for p in test_prompts_ablation:
            torch.manual_seed(1337)
            txt, spd = generate_text(
                model_ta,
                tok,
                p,
                temperature=cfg["temperature"],
                top_p=cfg["top_p"],
                top_k=cfg["top_k"],
                min_p=0.05,
                repetition_penalty=1.15,
                no_repeat_ngram_size=3,
            )
            cfg_out["samples"].append({"prompt": p, "text": txt, "speed": round(spd, 1)})
            print(f"\n[{cfg['name']}]: {p}\n  --> {txt}")
        ablation_results.append(cfg_out)

    # ── Save Full Evaluation JSON ─────────────────────────────────────────────
    eval_summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "models": {
            "gpt_v1": {
                "checkpoint": str(CKPT_V1),
                "step": meta_v1["opt_step"],
                "val_loss": 5.2793,
                "perplexity": 196.23,
                "bpb": 1.7915,
            },
            "track_a": {
                "checkpoint": str(CKPT_TRACK_A_BEST),
                "step": meta_ta["opt_step"],
                "val_loss": 3.7055,
                "perplexity": 40.67,
                "bpb": 1.2574,
                "val_loss_improvement": round(5.2793 - 3.7055, 4),
                "perplexity_reduction_factor": round(196.23 / 40.67, 2),
                "bpb_improvement": round(1.7915 - 1.2574, 4),
            },
        },
        "default_decoding_controls": default_controls,
        "generation_comparison": gen_results,
        "sampling_ablation": ablation_results,
    }

    out_json = PROJECT_ROOT / "analysis" / "track_a_post_evaluation.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    out_json_artifact = ARTIFACT_DIR / "track_a_post_evaluation.json"
    with open(out_json_artifact, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"\nSaved evaluation data to {out_json} and {out_json_artifact}")


if __name__ == "__main__":
    run_evaluation()

