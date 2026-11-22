"""SFT Dataset Refusal & Identity Leakage Cleaner (Phase 1).

Filters canned synthetic refusal strings, AI self-identification disclaimers,
OpenAI/ChatGPT identity leakage, and degenerate outputs from the SFT dataset.
Saves cleaned datasets to:
  data/sft/sft_train_cleaned.jsonl
  data/sft/sft_val_cleaned.jsonl
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

SFT_DIR = config.DATA_DIR / "sft"
TRAIN_INPUT = SFT_DIR / "sft_train.jsonl"
VAL_INPUT = SFT_DIR / "sft_val.jsonl"

TRAIN_CLEANED = SFT_DIR / "sft_train_cleaned.jsonl"
VAL_CLEANED = SFT_DIR / "sft_val_cleaned.jsonl"
METADATA_CLEANED = SFT_DIR / "sft_cleaned_metadata.json"

# Patterns to detect canned refusals and AI disclaimers (case-insensitive)
REFUSAL_PATTERNS = [
    (r"\bas an ai\b", "as an ai"),
    (r"\bas an a\.i\b", "as an a.i"),
    (r"\bas a language model\b", "as a language model"),
    (r"\bi am an ai\b", "i am an ai"),
    (r"\bi'm an ai\b", "i'm an ai"),
    (r"\bi am a language model\b", "i am a language model"),
    (r"\bi'm a language model\b", "i'm a language model"),
    (r"\bi do not have the ability\b", "i do not have the ability"),
    (r"\bi don't have the ability\b", "i don't have the ability"),
    (r"\bi do not have access\b", "i do not have access"),
    (r"\bi don't have access\b", "i don't have access"),
    (r"\bi am unable to browse\b", "i am unable to browse"),
    (r"\bi cannot browse\b", "i cannot browse"),
    (r"\bi can't browse\b", "i can't browse"),
    (r"\bi do not possess\b", "i do not possess"),
    (r"\bi don't possess\b", "i don't possess"),
    (r"\bi do not have personal opinions\b", "i do not have personal opinions"),
    (r"\bi don't have personal opinions\b", "i don't have personal opinions"),
    (r"\bi do not have feelings\b", "i do not have feelings"),
    (r"\bi don't have feelings\b", "i don't have feelings"),
    (r"\bi'm sorry, but as an ai\b", "i'm sorry, but as an ai"),
    (r"\bi am sorry, but as an ai\b", "i am sorry, but as an ai"),
    (r"\bi'm sorry, but i cannot\b", "i'm sorry, but i cannot"),
    (r"\bi am sorry, but i cannot\b", "i am sorry, but i cannot"),
    (r"\bi'm sorry, but i can't\b", "i'm sorry, but i can't"),
    (r"\bi am sorry, but i can't\b", "i am sorry, but i can't"),
    (r"\bi cannot assist with\b", "i cannot assist with"),
    (r"\bi can't assist with\b", "i can't assist with"),
    (r"\bi cannot help with\b", "i cannot help with"),
    (r"\bi can't help with\b", "i can't help with"),
    (r"\bi cannot fulfill this\b", "i cannot fulfill this"),
    (r"\bi can't fulfill this\b", "i can't fulfill this"),
    (r"\bmy knowledge cutoff\b", "my knowledge cutoff"),
]

# Patterns to detect synthetic model identity leakage
IDENTITY_PATTERNS = [
    (r"\bopenai\b", "openai"),
    (r"\bchatgpt\b", "chatgpt"),
    (r"\bgpt-3\b", "gpt-3"),
    (r"\bgpt-4\b", "gpt-4"),
    (r"\bdavinci-003\b", "davinci-003"),
]


def check_example(example: dict) -> tuple[bool, str]:
    """Check if an SFT example should be filtered out.
    
    Returns:
        (is_filtered: bool, reason: str)
    """
    answer = example.get("answer", "").strip()
    question = example.get("question", "").strip()

    # 1. Degenerate or empty
    if not answer or len(answer) < 5:
        return True, "Empty or extremely short answer (<5 chars)"
    if not question:
        return True, "Empty question"

    ans_lower = answer.lower()

    # 2. Check refusal patterns
    for pattern, name in REFUSAL_PATTERNS:
        if re.search(pattern, ans_lower):
            return True, f"Refusal/Disclaimer: {name}"

    # 3. Check identity leakage
    for pattern, name in IDENTITY_PATTERNS:
        if re.search(pattern, ans_lower):
            # Exception: if the question explicitly asks about OpenAI or GPT history, keep it
            q_lower = question.lower()
            if "openai" in q_lower or "chatgpt" in q_lower or "gpt" in q_lower:
                continue
            return True, f"Identity Leakage: {name}"

    return False, "Keep"


def process_file(input_path: Path, output_path: Path) -> tuple[int, int, list[dict], list[dict], dict[str, int]]:
    """Process a single JSONL file and return filtering statistics."""
    kept = []
    removed = []
    reason_counts = {}

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            example = json.loads(line)
            is_filtered, reason = check_example(example)

            if is_filtered:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
                removed.append({
                    "question": example.get("question", "")[:200],
                    "answer": example.get("answer", "")[:300],
                    "reason": reason,
                    "source": example.get("source", "unknown")
                })
            else:
                kept.append(example)

    # Write cleaned file
    with open(output_path, "w", encoding="utf-8") as f:
        for ex in kept:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")

    return len(kept) + len(removed), len(kept), kept, removed, reason_counts


def main():
    print("=" * 70)
    print("SFT Dataset Refusal & Identity Leakage Filter (Phase 1)")
    print("=" * 70)

    random.seed(42)

    # Process Train Set
    print(f"\nProcessing Train Set: {TRAIN_INPUT}...")
    total_tr, kept_tr, kept_examples_tr, removed_examples_tr, reasons_tr = process_file(TRAIN_INPUT, TRAIN_CLEANED)
    print(f"  Train: Scanned {total_tr:,} | Kept {kept_tr:,} ({kept_tr/total_tr*100:.2f}%) | Removed {len(removed_examples_tr):,} ({len(removed_examples_tr)/total_tr*100:.2f}%)")

    # Process Val Set
    print(f"\nProcessing Val Set: {VAL_INPUT}...")
    total_val, kept_val, kept_examples_val, removed_examples_val, reasons_val = process_file(VAL_INPUT, VAL_CLEANED)
    print(f"  Val: Scanned {total_val:,} | Kept {kept_val:,} ({kept_val/total_val*100:.2f}%) | Removed {len(removed_examples_val):,} ({len(removed_examples_val)/total_val*100:.2f}%)")

    # Aggregate Reasons
    all_reasons = {}
    for r, c in reasons_tr.items():
        all_reasons[r] = all_reasons.get(r, 0) + c
    for r, c in reasons_val.items():
        all_reasons[r] = all_reasons.get(r, 0) + c

    print("\n" + "-" * 70)
    print("Breakdown of Filtered Reasons (Train + Val):")
    print("-" * 70)
    for reason, count in sorted(all_reasons.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {reason:<45}: {count:>5} examples")

    total_removed = len(removed_examples_tr) + len(removed_examples_val)
    total_all = total_tr + total_val
    print("-" * 70)
    print(f"  TOTAL REMOVED: {total_removed:,} / {total_all:,} ({total_removed/total_all*100:.2f}%)")
    print(f"  TOTAL KEPT   : {kept_tr + kept_val:,} ({ (kept_tr + kept_val)/total_all*100:.2f}%)")

    # Print 15 Sample Filtered Examples
    print("\n" + "=" * 70)
    print("AUDIT: Sample Filtered Examples (Removed from SFT)")
    print("=" * 70)
    sample_removed = random.sample(removed_examples_tr, min(15, len(removed_examples_tr)))
    for idx, ex in enumerate(sample_removed, 1):
        print(f"\n[Filtered Sample #{idx}] [Source: {ex['source']}] [Reason: {ex['reason']}]")
        print(f"  Q: {ex['question'].replace(chr(10), ' ')[:140]}...")
        print(f"  A: {ex['answer'].replace(chr(10), ' ')[:200]}...")

    # Print 5 Sample Kept Examples (Verification of legitimate content)
    print("\n" + "=" * 70)
    print("AUDIT: Sample Preserved Examples (Kept in SFT)")
    print("=" * 70)
    sample_kept = random.sample(kept_examples_tr, min(5, len(kept_examples_tr)))
    for idx, ex in enumerate(sample_kept, 1):
        q = ex.get('question', '').replace('\n', ' ')[:140]
        a = ex.get('answer', '').replace('\n', ' ')[:200]
        src = ex.get('source', 'unknown')
        print(f"\n[Preserved Sample #{idx}] [Source: {src}]")
        print(f"  Q: {q}...")
        print(f"  A: {a}...")

    # Save metadata
    meta = {
        "original_train": total_tr,
        "original_val": total_val,
        "cleaned_train": kept_tr,
        "cleaned_val": kept_val,
        "removed_train": len(removed_examples_tr),
        "removed_val": len(removed_examples_val),
        "removed_percentage": total_removed / total_all * 100.0,
        "filter_reasons": all_reasons,
    }
    with open(METADATA_CLEANED, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print(f"\nMetadata saved to {METADATA_CLEANED}")
    print("=" * 70)


if __name__ == "__main__":
    main()

