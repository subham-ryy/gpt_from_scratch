"""Test the 8 user prompts under repetition penalties: 1.15, 1.0, and 0.8.

Tests both:
1. Fresh single-turn (clean prompt without multi-turn history)
2. Sequential multi-turn (accumulating history across turns 1-8, matching user's exact chat session)
"""
from __future__ import annotations

import json
from pathlib import Path
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

import config
from chat import load_assistant_model, answer_message, find_latest_checkpoint

USER_PROMPTS = [
    "what is a planet",
    "i love you",
    "what is evaporation",
    "what is condensation",
    "what is the capital of india",
    "what is the capital of britain",
    "what is the capital of germany",
    "what is an algorithm",
]

PENALTIES = [1.15, 1.0, 0.8]


def run_experiment():
    ckpt_path = find_latest_checkpoint()
    print(f"Loading checkpoint: {ckpt_path.name}")
    model, tokenizer, _ = load_assistant_model(ckpt_path)

    results = {}

    for rep in PENALTIES:
        print(f"\n{'='*70}")
        print(f"  RUNNING BENCHMARK WITH REPETITION PENALTY = {rep}")
        print(f"{'='*70}")

        # 1. Sequential Mode (Accumulated conversation history, identical to user's chat)
        print("\n--- Mode 1: Multi-Turn Conversation (Accumulated History) ---")
        history = []
        seq_outputs = []
        for q in USER_PROMPTS:
            ans, speed = answer_message(
                model=model,
                tokenizer=tokenizer,
                history=history,
                new_message=q,
                repetition_penalty=rep,
                temperature=0.4,
                top_p=0.90,
                min_p=0.08,
            )
            print(f"User > {q}")
            print(f"Assistant: {ans}\n")
            history.append({"role": "user", "content": q})
            history.append({"role": "assistant", "content": ans})
            seq_outputs.append({"prompt": q, "response": ans, "speed": round(speed, 1)})

        # 2. Fresh Single-Turn Mode (Zero prior history pollution)
        print("\n--- Mode 2: Fresh Single-Turn (Clean Context) ---")
        fresh_outputs = []
        for q in USER_PROMPTS:
            ans, speed = answer_message(
                model=model,
                tokenizer=tokenizer,
                history=[],
                new_message=q,
                repetition_penalty=rep,
                temperature=0.4,
                top_p=0.90,
                min_p=0.08,
            )
            print(f"User > {q}")
            print(f"Assistant: {ans}\n")
            fresh_outputs.append({"prompt": q, "response": ans, "speed": round(speed, 1)})

        results[f"rep_{rep}"] = {
            "sequential_history": seq_outputs,
            "fresh_single_turn": fresh_outputs,
        }

    # Save to JSON for analysis
    out_file = PROJECT_ROOT / "analysis" / "rep_penalty_experiment.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n[Saved] Results written to: {out_file}")


if __name__ == "__main__":
    run_experiment()
