"""Rigorous 4-Model Evolution Benchmark & Analysis (Phase 4 vs Previous Models).

Models Evaluated:
1. Model 2: Track A Continued Pretraining (100k docs packed, ~113M unique tokens, step 9,000)
2. Model 3: Phase 8 Single-Turn SFT (65k uncleaned Q&A pairs, step 4,000)
3. Model 4 Base: Phase 3 Continued Pretraining (446k docs packed, 500M unique tokens, step 15,000)
4. Model 4 Multi-Turn SFT: Phase 4 Micro-Assistant (55.6k curated dialogues, exact loss masking, step 3,400)

Evaluates:
- Training efficiency & loss scaling (Validation loss, Perplexity, Bits-per-byte)
- Refusal rate on everyday / personal prompts (Testing the elimination of "As an AI language model...")
- Multi-turn conversational memory & pronoun resolution
- Factual and scientific question answering
- Structured instruction following
- Elementary mathematical reasoning
- Python code generation
- Throughput and latency benchmarking
"""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Tuple

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import torch
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_ID = 1
EOS_ID = 2

REFUSAL_PATTERNS = [
    r"as an ai\b",
    r"as a language model\b",
    r"i don't have personal\b",
    r"i do not have access\b",
    r"i cannot feel\b",
    r"i am sorry, but\b",
    r"i'm sorry, but\b",
    r"i apologize, but\b",
    r"i cannot assist\b",
    r"i am an artificial\b",
]

MODELS_CONFIG = [
    {
        "name": "Model 2 (Track A Pretrain)",
        "short_name": "M2 (Pretrain 113M)",
        "path": PROJECT_ROOT / "experiments" / "20261106_032904_gpt_continued_pretrain" / "checkpoints" / "best.pt",
        "tokens_seen": 289_500_000,
        "step": 9000,
        "val_loss": 3.7055,
        "ppl": 40.67,
        "bpb": 1.2574,
        "format": "base",
    },
    {
        "name": "Model 3 (Phase 8 SFT)",
        "short_name": "M3 (Single-Turn SFT)",
        "path": PROJECT_ROOT / "experiments" / "20261106_115736_gpt_sft" / "checkpoints" / "best.pt",
        "tokens_seen": 305_000_000,
        "step": 4000,
        "val_loss": 2.6224,
        "ppl": 13.77,
        "bpb": 0.8902,
        "format": "single_turn_sft",
    },
    {
        "name": "Model 4 Base (500M Pretrain)",
        "short_name": "M4 Base (Pretrain 500M)",
        "path": PROJECT_ROOT / "experiments" / "20261121_023140_gpt_continued_pretrain" / "checkpoints" / "best.pt",
        "tokens_seen": 613_000_000,
        "step": 15000,
        "val_loss": 3.4810,
        "ppl": 32.49,
        "bpb": 1.1813,
        "format": "base",
    },
    {
        "name": "Model 4 Multi-Turn SFT",
        "short_name": "M4 (Multi-Turn SFT)",
        "path": PROJECT_ROOT / "experiments" / "20261121_174835_gpt_multi_turn_sft" / "checkpoints" / "best.pt",
        "tokens_seen": 635_000_000,
        "step": 3400,
        "val_loss": 2.0161,
        "ppl": 7.51,
        "bpb": 0.6844,
        "format": "multi_turn_sft",
    },
]

BENCHMARK_PROMPTS = [
    # 1. Factual Knowledge
    {
        "benchmark": "Factual Knowledge",
        "prompt": "What is the capital of France?",
        "base_prompt": "The capital of France is",
        "target_keywords": ["paris"],
    },
    {
        "benchmark": "Factual Knowledge",
        "prompt": "What is photosynthesis?",
        "base_prompt": "Photosynthesis is the process by which",
        "target_keywords": ["light", "plants", "chemical energy", "convert"],
    },
    {
        "benchmark": "Factual Knowledge",
        "prompt": "In computer science, what is an algorithm?",
        "base_prompt": "In computer science, an algorithm is",
        "target_keywords": ["step", "rules", "instructions", "solve", "procedure"],
    },
    {
        "benchmark": "Factual Knowledge",
        "prompt": "What is machine learning?",
        "base_prompt": "Machine learning is defined as",
        "target_keywords": ["data", "learn", "artificial intelligence", "algorithms"],
    },
    # 2. Refusal Test / Everyday Helpfulness
    {
        "benchmark": "Everyday Assistance",
        "prompt": "I had a stressful day at work. What are some simple ways to relax this evening?",
        "base_prompt": "Simple ways to relax after a stressful day include:",
        "target_keywords": ["bath", "breathe", "walk", "tea", "sleep", "rest"],
        "check_refusal": True,
    },
    {
        "benchmark": "Everyday Assistance",
        "prompt": "I feel unmotivated to study today. Can you give me some quick advice?",
        "base_prompt": "When feeling unmotivated to study, you should:",
        "target_keywords": ["break", "small", "goal", "start", "timer", "minutes"],
        "check_refusal": True,
    },
    {
        "benchmark": "Everyday Assistance",
        "prompt": "What are three healthy habits for better sleep?",
        "base_prompt": "Three healthy habits for better sleep are:",
        "target_keywords": ["caffeine", "schedule", "dark", "screen", "routine"],
        "check_refusal": True,
    },
    # 3. Structured Formatting
    {
        "benchmark": "Structured Formatting",
        "prompt": "List 3 benefits of drinking water daily. Format as a numbered list.",
        "base_prompt": "Three benefits of drinking water daily:\n1.",
        "target_keywords": ["1.", "2.", "3."],
    },
    # 4. Elementary Arithmetic & Reasoning
    {
        "benchmark": "Mathematical Reasoning",
        "prompt": "What is 17 multiplied by 24?",
        "base_prompt": "17 multiplied by 24 is equal to",
        "target_keywords": ["408"],
    },
    {
        "benchmark": "Mathematical Reasoning",
        "prompt": "A train travels 180 km in 3 hours. What is its average speed?",
        "base_prompt": "A train travels 180 km in 3 hours. Its average speed is",
        "target_keywords": ["60", "60 km/h"],
    },
    # 5. Basic Python Coding
    {
        "benchmark": "Code Generation",
        "prompt": "Write a Python function `is_even(n)` that returns True if n is even and False otherwise.",
        "base_prompt": "def is_even(n):\n    return",
        "target_keywords": ["def is_even", "n % 2 == 0", "return"],
    },
]

MULTI_TURN_CONVERSATIONS = [
    {
        "id": "mt_laptop",
        "benchmark": "Multi-Turn Context",
        "turn1": "My laptop is overheating when running intensive code.",
        "asst1": "Overheating can throttle performance. You should clean the exhaust vents, ensure it is on a flat surface, and check for runaway background processes.",
        "turn2": "What component should I check first?",
        "target_keywords": ["fan", "vent", "cpu", "thermal", "system", "dust"],
    },
    {
        "id": "mt_pets",
        "benchmark": "Multi-Turn Context",
        "turn1": "Alice has 2 dogs and Bob has 3 cats.",
        "asst1": "That's a nice group of pets! Together they have dogs and cats.",
        "turn2": "How many total animals do they have altogether?",
        "target_keywords": ["5", "five"],
    },
]


def load_model(checkpoint_path: Path) -> Tuple[GPT, float]:
    t0 = time.perf_counter()
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt.get("config_snapshot", {})

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
    model.eval()
    elapsed = time.perf_counter() - t0
    return model, elapsed


def format_prompt(item: Dict[str, Any], model_format: str, tokenizer: Tokenizer) -> List[int]:
    if model_format == "base":
        text = item.get("base_prompt", item["prompt"])
        return [BOS_ID] + tokenizer.encode(text).ids
    elif model_format == "single_turn_sft":
        text = f"### Question:\n{item['prompt']}\n\n### Answer:\n"
        return [BOS_ID] + tokenizer.encode(text).ids
    elif model_format == "multi_turn_sft":
        text = f"### User:\n{item['prompt']}\n\n### Assistant:\n"
        return [BOS_ID] + tokenizer.encode(text).ids
    else:
        return [BOS_ID] + tokenizer.encode(item["prompt"]).ids


def format_multi_turn_prompt(item: Dict[str, Any], model_format: str, tokenizer: Tokenizer) -> List[int]:
    if model_format == "multi_turn_sft":
        t1_user = tokenizer.encode("### User:\n").ids + tokenizer.encode(f"{item['turn1']}\n\n").ids
        t1_asst = (
            tokenizer.encode("### Assistant:\n").ids
            + tokenizer.encode(item["asst1"]).ids
            + [EOS_ID]
            + tokenizer.encode("\n\n").ids
        )
        t2_user = tokenizer.encode("### User:\n").ids + tokenizer.encode(f"{item['turn2']}\n\n").ids
        t2_asst = tokenizer.encode("### Assistant:\n").ids
        return [BOS_ID] + t1_user + t1_asst + t2_user + t2_asst
    elif model_format == "single_turn_sft":
        # Model 3 only knows single-turn Question/Answer
        comb = f"Context: {item['turn1']} Assistant replied: {item['asst1']}\nQuestion: {item['turn2']}"
        text = f"### Question:\n{comb}\n\n### Answer:\n"
        return [BOS_ID] + tokenizer.encode(text).ids
    else:
        # Base model prompt
        comb = f"{item['turn1']} {item['asst1']} Question: {item['turn2']} Answer:"
        return [BOS_ID] + tokenizer.encode(comb).ids


def clean_output(raw_text: str) -> str:
    stop_markers = ["### User:", "\n### User:", "### Assistant:", "\n###", "### Question:"]
    t = raw_text.strip()
    for sm in stop_markers:
        pos = t.find(sm)
        if pos != -1:
            t = t[:pos].strip()
    return t


def is_refusal(text: str) -> bool:
    lower = text.lower()
    return any(re.search(pat, lower) for pat in REFUSAL_PATTERNS)


def run_benchmark():
    print("=" * 75)
    print("  RIGOROUS 4-MODEL EVOLUTION BENCHMARK (Phase 4 vs Previous Checkpoints)")
    print("=" * 75)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    results = {}

    for cfg in MODELS_CONFIG:
        m_name = cfg["name"]
        print(f"\n[{m_name}] Loading checkpoint from: {cfg['path'].name}...")
        model, load_sec = load_model(cfg["path"])
        print(f"  Loaded in {load_sec:.2f}s on {DEVICE.upper()} (Params: {model.get_num_params():,})")

        m_results = {
            "config": {
                "name": cfg["name"],
                "short_name": cfg["short_name"],
                "step": cfg["step"],
                "tokens_seen": cfg["tokens_seen"],
                "val_loss": cfg["val_loss"],
                "ppl": cfg["ppl"],
                "bpb": cfg["bpb"],
                "load_time_sec": round(load_sec, 2),
            },
            "generations": [],
            "multi_turn": [],
            "metrics": {},
        }

        total_gen_tokens = 0
        total_gen_time = 0.0
        refusal_count = 0
        refusal_tests = 0
        eos_count = 0

        # 1. Single-turn benchmark prompts
        for item in BENCHMARK_PROMPTS:
            prompt_ids = format_prompt(item, cfg["format"], tokenizer)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_tensor,
                    max_new_tokens=90,
                    temperature=0.4,
                    top_p=0.90,
                    min_p=0.08,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    eos_token_id=EOS_ID,
                )
            gen_sec = time.perf_counter() - t0

            gen_tokens = out[0, len(prompt_ids):].tolist()
            has_eos = EOS_ID in gen_tokens
            if has_eos:
                eos_count += 1
                gen_tokens = gen_tokens[: gen_tokens.index(EOS_ID)]

            raw_resp = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            cleaned_resp = clean_output(raw_resp)

            tokens_num = len(gen_tokens)
            total_gen_tokens += tokens_num
            total_gen_time += gen_sec
            tok_speed = tokens_num / max(gen_sec, 1e-6)

            has_refusal = is_refusal(cleaned_resp)
            if item.get("check_refusal"):
                refusal_tests += 1
                if has_refusal:
                    refusal_count += 1

            # Keyword hit check
            hits = [kw for kw in item["target_keywords"] if kw.lower() in cleaned_resp.lower()]
            hit_rate = len(hits) / max(len(item["target_keywords"]), 1)

            m_results["generations"].append({
                "benchmark": item["benchmark"],
                "prompt": item["prompt"],
                "response": cleaned_resp,
                "has_eos": has_eos,
                "has_refusal": has_refusal,
                "speed_tok_s": round(tok_speed, 1),
                "keyword_hits": hits,
                "hit_rate": round(hit_rate, 2),
            })

        # 2. Multi-turn benchmark prompts
        for mt_item in MULTI_TURN_CONVERSATIONS:
            prompt_ids = format_multi_turn_prompt(mt_item, cfg["format"], tokenizer)
            input_tensor = torch.tensor([prompt_ids], dtype=torch.long, device=DEVICE)

            t0 = time.perf_counter()
            with torch.no_grad():
                out = model.generate(
                    input_tensor,
                    max_new_tokens=90,
                    temperature=0.4,
                    top_p=0.90,
                    min_p=0.08,
                    repetition_penalty=1.15,
                    no_repeat_ngram_size=4,
                    eos_token_id=EOS_ID,
                )
            gen_sec = time.perf_counter() - t0

            gen_tokens = out[0, len(prompt_ids):].tolist()
            has_eos = EOS_ID in gen_tokens
            if has_eos:
                gen_tokens = gen_tokens[: gen_tokens.index(EOS_ID)]

            raw_resp = tokenizer.decode(gen_tokens, skip_special_tokens=True)
            cleaned_resp = clean_output(raw_resp)
            tok_speed = len(gen_tokens) / max(gen_sec, 1e-6)

            hits = [kw for kw in mt_item["target_keywords"] if kw.lower() in cleaned_resp.lower()]

            m_results["multi_turn"].append({
                "id": mt_item["id"],
                "turn1": mt_item["turn1"],
                "turn2": mt_item["turn2"],
                "response": cleaned_resp,
                "has_eos": has_eos,
                "speed_tok_s": round(tok_speed, 1),
                "hits": hits,
            })

        # Aggregate Metrics
        total_items = len(BENCHMARK_PROMPTS)
        m_results["metrics"] = {
            "avg_throughput_tok_s": round(total_gen_tokens / max(total_gen_time, 1e-6), 1),
            "clean_eos_rate_pct": round((eos_count / total_items) * 100, 1),
            "refusal_rate_pct": round((refusal_count / max(refusal_tests, 1)) * 100, 1) if refusal_tests > 0 else 0.0,
        }

        print(f"  -> Avg Throughput  : {m_results['metrics']['avg_throughput_tok_s']} tok/s")
        print(f"  -> Clean EOS Rate  : {m_results['metrics']['clean_eos_rate_pct']}%")
        print(f"  -> Refusal Rate    : {m_results['metrics']['refusal_rate_pct']}% (Target: 0% for everyday queries)")

        results[cfg["short_name"]] = m_results
        del model
        torch.cuda.empty_cache()

    # Save JSON results
    out_json = PROJECT_ROOT / "analysis" / "model_evolution_comparison.json"
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Quantitative comparison results saved to: {out_json}")

    # Generate Visualization Charts
    generate_comparison_plots(results)


def generate_comparison_plots(results: Dict[str, Any]):
    plots_dir = PROJECT_ROOT / "analysis" / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    out_plot = plots_dir / "model_evolution_comparison.png"

    models = list(results.keys())
    val_losses = [results[m]["config"]["val_loss"] for m in models]
    perplexities = [results[m]["config"]["ppl"] for m in models]
    refusal_rates = [results[m]["metrics"]["refusal_rate_pct"] for m in models]
    throughputs = [results[m]["metrics"]["avg_throughput_tok_s"] for m in models]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Evolutionary Benchmarks: GPT-98.4M Scaling (Base -> SFT -> 500M -> Multi-Turn SFT)", fontsize=15, fontweight="bold")

    colors = ["#4A90E2", "#E2844A", "#2E7D32", "#8E24AA"]

    # 1. Validation Loss Progression
    ax1 = axes[0, 0]
    bars1 = ax1.bar(models, val_losses, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax1.set_title("Validation Loss by Checkpoint (Lower is Better)", fontweight="bold")
    ax1.set_ylabel("Validation Loss")
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars1:
        yval = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2.0, yval + 0.05, f"{yval:.2f}", ha="center", va="bottom", fontweight="bold")

    # 2. Perplexity (Log Scale)
    ax2 = axes[0, 1]
    bars2 = ax2.bar(models, perplexities, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax2.set_title("Perplexity Progression (Lower is Better)", fontweight="bold")
    ax2.set_ylabel("Perplexity (PPL)")
    ax2.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars2:
        yval = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2.0, yval + 0.5, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    # 3. Canned Refusal Rate (% on everyday scenarios)
    ax3 = axes[1, 0]
    bars3 = ax3.bar(models, refusal_rates, color=["#9E9E9E", "#D32F2F", "#9E9E9E", "#388E3C"], width=0.55, edgecolor="black", alpha=0.85)
    ax3.set_title("Canned Refusal Rate on Everyday Queries (% Lower is Better)", fontweight="bold")
    ax3.set_ylabel("Refusal Rate (%)")
    ax3.set_ylim(0, 110)
    ax3.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars3:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.0f}%", ha="center", va="bottom", fontweight="bold")

    # 4. Inference Throughput (tokens/s)
    ax4 = axes[1, 1]
    bars4 = ax4.bar(models, throughputs, color=colors, width=0.55, edgecolor="black", alpha=0.85)
    ax4.set_title("Inference Throughput on RTX 4050 (Tokens / Sec)", fontweight="bold")
    ax4.set_ylabel("Tokens / Second")
    ax4.grid(axis="y", linestyle="--", alpha=0.5)
    for bar in bars4:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width()/2.0, yval + 1, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.savefig(out_plot, dpi=200)
    plt.close()
    print(f"[Plot] Comparison plot saved to: {out_plot}")


if __name__ == "__main__":
    run_benchmark()
