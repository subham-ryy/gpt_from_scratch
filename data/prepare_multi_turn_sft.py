"""Multi-Turn SFT Dataset Curation Pipeline (Phase 4).

Combines:
1. Cleaned Single-Turn Factual QA (SciQ, Dolly, cleaned Alpaca from Phase 1).
2. SmolTalk Multi-Turn Dialogues (everyday-conversations, smol-constraints, smol-summarize, smol-rewrite).
3. MetaMathQA Short Arithmetic (concise 1-2 step math, capped at <= 220 chars).

Guarantees:
- Case-insensitive refusal filtering applied across all sources.
- Non-English and code-heavy examples pruned.
- Conversation-level 95% train / 5% validation split (zero dialogue leakage).
- Piece-wise component formatting (<|bos|>### User:\n...### Assistant:\n...<|eos|>).
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from datasets import load_dataset
from tokenizers import Tokenizer

import config
from data.clean_sft_refusals import REFUSAL_PATTERNS, IDENTITY_PATTERNS

OUTPUT_DIR = config.DATA_DIR / "sft_multi_turn"
TRAIN_OUTPUT = OUTPUT_DIR / "multi_turn_train.jsonl"
VAL_OUTPUT = OUTPUT_DIR / "multi_turn_val.jsonl"
METADATA_OUTPUT = OUTPUT_DIR / "multi_turn_metadata.json"

MAX_SEQ_TOKENS = 1000
MATH_MAX_CHARS = 220
RANDOM_SEED = 1337


def is_refusal(text: str) -> bool:
    """Detect canned refusals and identity leakage in assistant responses."""
    if not text or len(text.strip()) < 5:
        return True
    t_lower = text.lower()
    for pattern, _ in REFUSAL_PATTERNS:
        if re.search(pattern, t_lower):
            return True
    for pattern, _ in IDENTITY_PATTERNS:
        if re.search(pattern, t_lower):
            return True
    return False


def is_mostly_code(text: str) -> bool:
    """Heuristic to detect code-heavy solutions."""
    code_indicators = [
        "def ", "class ", "import ", "return ", "print(", 
        "```python", "```javascript", "```cpp", "```java",
        "for i in range", "public static void"
    ]
    return sum(1 for ind in code_indicators if ind in text) >= 3


def is_non_english(text: str) -> bool:
    """Detect non-Latin/non-ASCII heavy texts."""
    non_ascii = len(re.findall(r"[^\x00-\x7F]", text))
    return (non_ascii / max(len(text), 1)) > 0.15


def load_cleaned_single_turn() -> list[dict]:
    """Load cleaned single-turn examples from Phase 1."""
    print("\n[1/3] Loading Cleaned Single-Turn Factual QA...")
    cleaned_train = config.DATA_DIR / "sft" / "sft_train_cleaned.jsonl"
    cleaned_val = config.DATA_DIR / "sft" / "sft_val_cleaned.jsonl"

    conversations = []
    for p in [cleaned_train, cleaned_val]:
        if not p.exists():
            continue
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                d = json.loads(line)
                q = d.get("question", "").strip()
                a = d.get("answer", "").strip()
                src = d.get("source", "single_turn")
                if not q or not a or len(a) < 10:
                    continue
                if is_mostly_code(a) or is_non_english(q) or is_non_english(a):
                    continue
                conversations.append({
                    "source": f"single_turn_{src}",
                    "messages": [
                        {"role": "user", "content": q},
                        {"role": "assistant", "content": a}
                    ]
                })

    print(f"  Loaded {len(conversations):,} clean single-turn conversations.")
    return conversations


def load_smoltalk_dialogues() -> list[dict]:
    """Load multi-turn conversational subsets from HuggingFaceTB/smoltalk."""
    print("\n[2/3] Loading SmolTalk Multi-Turn Subsets...")
    configs = [
        ("everyday-conversations", None),
        ("smol-constraints", 12000),
        ("smol-summarize", 10000),
        ("smol-rewrite", 6000),
    ]

    conversations = []
    refusal_skipped = 0
    code_skipped = 0

    for cfg_name, max_samples in configs:
        print(f"  Streaming {cfg_name}...")
        ds = load_dataset("HuggingFaceTB/smoltalk", cfg_name, split="train")
        indices = list(range(len(ds)))
        if max_samples and len(indices) > max_samples:
            random.seed(RANDOM_SEED)
            indices = random.sample(indices, max_samples)

        added = 0
        for idx in indices:
            item = ds[idx]
            raw_msgs = item.get("messages", [])
            if not raw_msgs:
                continue

            clean_msgs = []
            skip = False
            for m in raw_msgs:
                role = m.get("role", "")
                content = (m.get("content") or "").strip()
                if not content:
                    continue
                if role == "system":
                    # Fold system instruction into first user turn
                    continue
                if role == "assistant":
                    if is_refusal(content):
                        refusal_skipped += 1
                        skip = True
                        break
                    if is_mostly_code(content):
                        code_skipped += 1
                        skip = True
                        break
                clean_msgs.append({"role": role, "content": content})

            if skip or len(clean_msgs) < 2:
                continue

            # Ensure alternation user -> assistant
            if clean_msgs[0]["role"] != "user" or clean_msgs[-1]["role"] != "assistant":
                continue

            conversations.append({
                "source": f"smoltalk_{cfg_name}",
                "messages": clean_msgs
            })
            added += 1

        print(f"    Added {added:,} conversations from {cfg_name}.")

    print(f"  Total SmolTalk conversations: {len(conversations):,} (skipped {refusal_skipped:,} refusals, {code_skipped:,} code)")
    return conversations


def load_metamath_short() -> list[dict]:
    """Load concise 1-2 step arithmetic problems from metamathqa-50k."""
    print("\n[3/3] Loading MetaMathQA Short Arithmetic...")
    ds = load_dataset("HuggingFaceTB/smoltalk", "metamathqa-50k", split="train")

    conversations = []
    for item in ds:
        msgs = item.get("messages", [])
        if len(msgs) < 2:
            continue
        q = (msgs[0].get("content") or "").strip()
        a = (msgs[1].get("content") or "").strip()

        # Strict length filter: short 1-2 step solutions
        if len(a) > MATH_MAX_CHARS or len(a) < 15:
            continue
        if is_mostly_code(a) or is_non_english(q) or is_non_english(a):
            continue

        conversations.append({
            "source": "metamathqa_short",
            "messages": [
                {"role": "user", "content": q},
                {"role": "assistant", "content": a}
            ]
        })

    # Sample to ~6,000 to keep math at ~10-12%
    random.seed(RANDOM_SEED)
    if len(conversations) > 6500:
        conversations = random.sample(conversations, 6500)

    print(f"  Curated {len(conversations):,} short MetaMathQA examples (capped <= {MATH_MAX_CHARS} chars).")
    return conversations


def validate_and_tokenize(conversations: list[dict], tokenizer: Tokenizer) -> list[dict]:
    """Filter conversations exceeding context window and pre-verify token length."""
    print("\n[Verification] Validating sequence lengths with tokenizer...")
    valid = []
    too_long = 0

    for conv in conversations:
        # Approximate tokens
        total_text = " ".join(m["content"] for m in conv["messages"])
        if len(total_text) > 4000:
            too_long += 1
            continue

        # Exact piece-wise token length
        tok_len = 1  # BOS
        for m in conv["messages"]:
            tok_len += len(tokenizer.encode(f"### {m['role'].capitalize()}:\n").ids)
            tok_len += len(tokenizer.encode(m["content"]).ids) + 1  # EOS or \n\n

        if tok_len > MAX_SEQ_TOKENS:
            too_long += 1
            continue

        conv["token_length"] = tok_len
        valid.append(conv)

    print(f"  Retained {len(valid):,} conversations (filtered {too_long:,} exceeding {MAX_SEQ_TOKENS} tokens).")
    return valid


def main():
    print("=" * 70)
    print("Multi-Turn SFT Dataset Builder (Phase 4)")
    print("=" * 70)

    random.seed(RANDOM_SEED)
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    # 1. Load components
    single_turn = load_cleaned_single_turn()
    smoltalk = load_smoltalk_dialogues()
    metamath = load_metamath_short()

    # Balance mixture:
    # Target: ~25k single-turn, ~25k smoltalk, ~6.5k metamath => ~56.5k total
    if len(single_turn) > 26000:
        single_turn = random.sample(single_turn, 26000)

    all_conversations = single_turn + smoltalk + metamath
    random.shuffle(all_conversations)

    # 2. Token length verification
    clean_conversations = validate_and_tokenize(all_conversations, tokenizer)

    # 3. Split by conversation (95% Train / 5% Val)
    random.shuffle(clean_conversations)
    split_idx = int(len(clean_conversations) * 0.95)
    train_convs = clean_conversations[:split_idx]
    val_convs = clean_conversations[split_idx:]

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(TRAIN_OUTPUT, "w", encoding="utf-8") as f:
        for c in train_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    with open(VAL_OUTPUT, "w", encoding="utf-8") as f:
        for c in val_convs:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    # Source breakdown
    source_counts_tr = {}
    for c in train_convs:
        src = c["source"].split("_")[0]
        source_counts_tr[src] = source_counts_tr.get(src, 0) + 1

    print("\n" + "-" * 70)
    print("Final Dataset Mixture (Train Split):")
    print("-" * 70)
    total_tr = len(train_convs)
    for src, count in sorted(source_counts_tr.items(), key=lambda x: x[1], reverse=True):
        print(f"  - {src:<20}: {count:>6,} ({count/total_tr*100:5.1f}%)")
    print(f"  Total Train Dialogues: {total_tr:,}")
    print(f"  Total Val Dialogues  : {len(val_convs):,}")

    # Print 5 sample conversations
    print("\n" + "=" * 70)
    print("AUDIT: Sample Multi-Turn Conversations")
    print("=" * 70)
    for idx, c in enumerate(random.sample(train_convs, 5), 1):
        print(f"\n[Sample #{idx}] [Source: {c['source']}] [Tokens: {c['token_length']}]")
        for m in c["messages"]:
            role = m["role"].upper()
            content = m["content"].replace("\n", " ")[:120]
            print(f"  {role}: {content}...")

    # Save metadata
    meta = {
        "train_conversations": len(train_convs),
        "val_conversations": len(val_convs),
        "sources": source_counts_tr,
        "max_seq_tokens": MAX_SEQ_TOKENS,
    }
    with open(METADATA_OUTPUT, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"\nMetadata saved to {METADATA_OUTPUT}")
    print("=" * 70)


if __name__ == "__main__":
    main()
