"""SFT Dataset Curation Pipeline (Phase 8: Factual & Instruction Fine-Tuning).

Downloads, filters, deduplicates, and splits instruction and factual Q&A datasets:
1. yahma/alpaca-cleaned (~51k general knowledge & instruction pairs)
2. allenai/sciq (~13k foundational science questions with explanations)
3. databricks/databricks-dolly-15k (filtered for QA & extraction categories)

Filtering & Formatting:
- Combined sequence (<|bos|>### Question:\n...\n\n### Answer:\n...<|eos|>) <= 950 tokens.
- Answer tokens >= 10 tokens.
- Deduplicated by normalized question text.
- Deterministic 95% Train / 5% Val split -> data/sft/sft_train.jsonl and data/sft/sft_val.jsonl.
"""
from __future__ import annotations

import json
from pathlib import Path
import re
import sys
import time

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import concatenate_datasets, load_dataset
import numpy as np
from tokenizers import Tokenizer

import config

BOS_ID = 1
EOS_ID = 2
MAX_SEQ_LEN = 950
MIN_ANSWER_TOKENS = 10
RANDOM_SEED = 1337
SFT_DIR = config.DATA_DIR / "sft"
TRAIN_OUTPUT = SFT_DIR / "sft_train.jsonl"
VAL_OUTPUT = SFT_DIR / "sft_val.jsonl"
SPLIT_RATIO = 0.95


def normalize_question(text: str) -> str:
    """Normalize question text for deduplication."""
    text = text.strip().lower()
    text = re.sub(r"\s+", " ", text)
    # Remove trailing punctuation variations (e.g. '?', '.', '!') for deduplication
    text = re.sub(r"[?!.,;:]+$", "", text).strip()
    return text


def build_alpaca_examples() -> list[dict]:
    """Load and format yahma/alpaca-cleaned."""
    print("\n[Loading 1/3] yahma/alpaca-cleaned...")
    ds = load_dataset("yahma/alpaca-cleaned", split="train")
    examples = []
    for item in ds:
        instruction = (item.get("instruction") or "").strip()
        inp = (item.get("input") or "").strip()
        out = (item.get("output") or "").strip()

        if not instruction or not out:
            continue

        if inp:
            question = f"{instruction}\n\n{inp}"
        else:
            question = instruction

        examples.append({
            "question": question,
            "answer": out,
            "source": "alpaca-cleaned"
        })
    print(f"  Raw extracted: {len(examples):,} examples")
    return examples


def build_sciq_examples() -> list[dict]:
    """Load and format allenai/sciq across train, validation, and test splits."""
    print("\n[Loading 2/3] allenai/sciq...")
    ds_train = load_dataset("allenai/sciq", split="train")
    ds_val = load_dataset("allenai/sciq", split="validation")
    ds_test = load_dataset("allenai/sciq", split="test")
    ds = concatenate_datasets([ds_train, ds_val, ds_test])

    examples = []
    for item in ds:
        q = (item.get("question") or "").strip()
        correct = (item.get("correct_answer") or "").strip()
        support = (item.get("support") or "").strip()

        if not q or not correct:
            continue

        # Combine correct answer with explanatory support text
        if support:
            answer = f"{correct}.\n\n{support}"
        else:
            answer = correct

        examples.append({
            "question": q,
            "answer": answer,
            "source": "sciq"
        })
    print(f"  Raw extracted: {len(examples):,} examples")
    return examples


def build_dolly_examples() -> list[dict]:
    """Load and format databricks/databricks-dolly-15k for QA/extraction categories."""
    print("\n[Loading 3/3] databricks/databricks-dolly-15k...")
    ds = load_dataset("databricks/databricks-dolly-15k", split="train")
    target_categories = {"open_qa", "general_qa", "closed_qa", "information_extraction"}

    examples = []
    for item in ds:
        cat = item.get("category", "")
        if cat not in target_categories:
            continue

        instruction = (item.get("instruction") or "").strip()
        context = (item.get("context") or "").strip()
        response = (item.get("response") or "").strip()

        if not instruction or not response:
            continue

        if context:
            question = f"{instruction}\n\nContext:\n{context}"
        else:
            question = instruction

        examples.append({
            "question": question,
            "answer": response,
            "source": f"dolly-{cat}"
        })
    print(f"  Raw extracted ({len(target_categories)} target categories): {len(examples):,} examples")
    return examples


def process_and_filter(
    examples: list[dict], tokenizer: Tokenizer
) -> tuple[list[dict], dict]:
    """Filter by token lengths and deduplicate by question text."""
    print("\n[Filtering & Deduplicating]...")
    seen_questions: set[str] = set()
    cleaned: list[dict] = []

    prompt_lengths = []
    answer_lengths = []
    total_lengths = []

    for item in examples:
        q = item["question"]
        a = item["answer"]

        # Deduplicate
        norm_q = normalize_question(q)
        if not norm_q or norm_q in seen_questions:
            continue
        seen_questions.add(norm_q)

        # Check token lengths
        prefix_str = f"### Question:\n{q}\n\n### Answer:\n"
        prefix_ids = [BOS_ID] + tokenizer.encode(prefix_str).ids
        ans_ids = tokenizer.encode(a).ids + [EOS_ID]

        len_p = len(prefix_ids)
        len_a = len(ans_ids)
        len_tot = len_p + len_a

        if len_a < MIN_ANSWER_TOKENS:
            continue
        if len_tot > MAX_SEQ_LEN:
            continue

        item["prompt_tokens"] = len_p
        item["answer_tokens"] = len_a
        item["total_tokens"] = len_tot

        prompt_lengths.append(len_p)
        answer_lengths.append(len_a)
        total_lengths.append(len_tot)
        cleaned.append(item)

    stats = {
        "total_extracted": len(examples),
        "unique_cleaned": len(cleaned),
        "prompt_tokens": {
            "min": int(np.min(prompt_lengths)),
            "max": int(np.max(prompt_lengths)),
            "mean": float(np.mean(prompt_lengths)),
            "median": float(np.median(prompt_lengths)),
        },
        "answer_tokens": {
            "min": int(np.min(answer_lengths)),
            "max": int(np.max(answer_lengths)),
            "mean": float(np.mean(answer_lengths)),
            "median": float(np.median(answer_lengths)),
        },
        "total_tokens": {
            "min": int(np.min(total_lengths)),
            "max": int(np.max(total_lengths)),
            "mean": float(np.mean(total_lengths)),
            "median": float(np.median(total_lengths)),
        },
    }

    return cleaned, stats


def main():
    start_time = time.time()
    print("=" * 70)
    print("PHASE 8 SFT DATASET PREPARATION PIPELINE")
    print("=" * 70)

    SFT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from: {config.TOKENIZER_FILE}")
    if not config.TOKENIZER_FILE.exists():
        raise FileNotFoundError(f"Tokenizer not found: {config.TOKENIZER_FILE}")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    # 1. Load datasets
    all_raw: list[dict] = []
    all_raw.extend(build_alpaca_examples())
    all_raw.extend(build_sciq_examples())
    all_raw.extend(build_dolly_examples())

    print(f"\nTotal raw collected: {len(all_raw):,} examples")

    # 2. Filter & Deduplicate
    cleaned_examples, stats = process_and_filter(all_raw, tokenizer)

    print(f"\nRetained after cleaning: {len(cleaned_examples):,} unique examples")
    print(f"  Prompt tokens : min={stats['prompt_tokens']['min']}, median={stats['prompt_tokens']['median']:.1f}, mean={stats['prompt_tokens']['mean']:.1f}, max={stats['prompt_tokens']['max']}")
    print(f"  Answer tokens : min={stats['answer_tokens']['min']}, median={stats['answer_tokens']['median']:.1f}, mean={stats['answer_tokens']['mean']:.1f}, max={stats['answer_tokens']['max']}")
    print(f"  Total tokens  : min={stats['total_tokens']['min']}, median={stats['total_tokens']['median']:.1f}, mean={stats['total_tokens']['mean']:.1f}, max={stats['total_tokens']['max']}")

    # 3. Deterministic Train/Val Split (95% / 5%)
    rng = np.random.default_rng(RANDOM_SEED)
    indices = np.arange(len(cleaned_examples))
    rng.shuffle(indices)

    num_train = int(len(cleaned_examples) * SPLIT_RATIO)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]

    train_data = [cleaned_examples[i] for i in train_indices]
    val_data = [cleaned_examples[i] for i in val_indices]

    print(f"\nDataset Split (seed={RANDOM_SEED}):")
    print(f"  Train: {len(train_data):,} examples (95.0%) -> {TRAIN_OUTPUT}")
    print(f"  Val  : {len(val_data):,} examples (5.0%) -> {VAL_OUTPUT}")

    # Count by source
    for split_name, dataset_split in [("Train", train_data), ("Val", val_data)]:
        source_counts: dict[str, int] = {}
        for ex in dataset_split:
            src = ex.get("source", "other")
            source_counts[src] = source_counts.get(src, 0) + 1
        print(f"  {split_name} source composition: {source_counts}")

    # 4. Save JSONL files
    print("\nWriting JSONL files...")
    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for ex in train_data:
            record = {"question": ex["question"], "answer": ex["answer"], "source": ex.get("source", "")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for ex in val_data:
            record = {"question": ex["question"], "answer": ex["answer"], "source": ex.get("source", "")}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    # Save summary metadata
    meta_path = SFT_DIR / "sft_metadata.json"
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "random_seed": RANDOM_SEED,
        "max_seq_len": MAX_SEQ_LEN,
        "min_answer_tokens": MIN_ANSWER_TOKENS,
        "train_examples": len(train_data),
        "val_examples": len(val_data),
        "total_examples": len(cleaned_examples),
        "token_stats": stats,
    }
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - start_time
    print(f"\n[Done] Pipeline finished successfully in {elapsed:.1f}s.")
    print("=" * 70)


if __name__ == "__main__":
    main()

