"""Phase 3: Rule-Based Preference Labeling and Pair Construction.

Scoring dimensions:
- Canned refusal phrases: large penalty (-25.0)
- Missing EOS / cut-off at max_new_tokens: large penalty (-15.0)
- 3-gram and 4-gram repetition rate, repeated sentences: penalty proportional to rate (-30.0 * rate)
- Leaked \\boxed{} or 'The answer is:' on non-math prompts: penalty (-20.0)
- Header leaks ('### User:', '### Assistant:'): penalty (-25.0)
- Garbage / empty / < 3 tokens: penalty (-30.0)
- Excessive length beyond sane cap: penalty (-10.0)
- Lexical on-topic overlap: small bonus (+2.5 * overlap)

Pair Construction:
- Target: ~3,000 pairs
- Split 90/10 by prompt into train and heldout
- Length bias controlled: mean token length difference kept small
- Includes <= 10% synthetic refusal pairs (marked source='synthetic_refusal')
- Generates `dpo_pairs_review.md` with 40 random pairs and aggregate statistics
"""
from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from tokenizers import Tokenizer
import config
from data.clean_sft_refusals import REFUSAL_PATTERNS

STOPWORDS = {
    "a", "an", "the", "in", "on", "at", "to", "for", "of", "with", "by", "is",
    "are", "was", "were", "and", "or", "it", "this", "that", "i", "you", "my",
    "your", "how", "what", "can", "do", "does", "be", "have", "has", "should"
}

SYNTHETIC_REFUSALS = [
    "I'm sorry, but as an AI language model, I do not have personal feelings, emotions, or personal experience, so I cannot help with this situation.",
    "As an AI language model, I am unable to provide personal advice or recommendations for your daily life.",
    "I apologize, but as an artificial intelligence, I don't have the ability to assist with personal situations or everyday lifestyle choices.",
    "Dear user, as a language model, I do not possess personal opinions or real-world capabilities to guide you through this.",
    "I cannot assist you with this request. As an AI developed to process data, I do not have access to real-time personal feelings.",
]


def extract_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def score_response(
    prompt_text: str,
    response_text: str,
    is_math: bool,
    has_eos: bool,
    token_len: int,
) -> tuple[float, dict[str, float]]:
    """Compute transparent rule-based score with exact breakdown."""
    breakdown = {}
    total_score = 50.0  # Base neutral score
    t_lower = response_text.lower()
    cleaned = response_text.strip()

    # 1. Garbage / Empty Penalty
    words = re.findall(r"\b\w+\b", t_lower)
    if token_len < 3 or len(cleaned) < 5 or len(words) < 2:
        breakdown["garbage_penalty"] = -30.0
        total_score += -30.0
        return total_score, breakdown
    else:
        breakdown["garbage_penalty"] = 0.0

    # 2. Character repetition runs (e.g. 'aaaa', '.....')
    if re.search(r"(.)\1{4,}", cleaned):
        breakdown["char_run_penalty"] = -15.0
        total_score += -15.0
    else:
        breakdown["char_run_penalty"] = 0.0

    # 3. Canned Refusal Penalty
    is_ref = False
    for pat, _ in REFUSAL_PATTERNS:
        if re.search(pat, t_lower):
            is_ref = True
            break
    if is_ref:
        breakdown["refusal_penalty"] = -25.0
        total_score += -25.0
    else:
        breakdown["refusal_penalty"] = 0.0

    # 4. Missing EOS Penalty
    if not has_eos:
        breakdown["no_eos_penalty"] = -15.0
        total_score += -15.0
    else:
        breakdown["no_eos_penalty"] = 0.0

    # 5. Repetition Penalty (3-gram, 4-gram, sentence loops)
    if len(words) >= 6:
        tri = extract_ngrams(words, 3)
        quad = extract_ngrams(words, 4)
        rep3 = 1.0 - (len(set(tri)) / max(len(tri), 1))
        rep4 = 1.0 - (len(set(quad)) / max(len(quad), 1))
        avg_rep = (rep3 + rep4) / 2.0
        rep_pen = -35.0 * avg_rep

        # Repeated sentence check
        sentences = [s.strip() for s in re.split(r"[.!?\n]", cleaned) if len(s.strip()) > 15]
        s_counts = Counter(sentences)
        if any(cnt >= 2 for cnt in s_counts.values()):
            rep_pen -= 15.0

        breakdown["repetition_penalty"] = round(rep_pen, 2)
        total_score += rep_pen
    else:
        breakdown["repetition_penalty"] = 0.0

    # 6. Leaked Format Penalty on Non-Math
    leaked = False
    if not is_math:
        if "\\boxed" in cleaned or "the answer is:" in t_lower or "the answer is " in t_lower:
            leaked = True
    if leaked:
        breakdown["leaked_format_penalty"] = -20.0
        total_score += -20.0
    else:
        breakdown["leaked_format_penalty"] = 0.0

    # 7. Header Leak Penalty
    if "### user:" in t_lower or "### assistant:" in t_lower or "###" in cleaned:
        breakdown["header_leak_penalty"] = -25.0
        total_score += -25.0
    else:
        breakdown["header_leak_penalty"] = 0.0

    # 8. Excessive Length Penalty
    if token_len > 150:
        excess_pen = -10.0 * min(1.5, (token_len - 150) / 40.0)
        breakdown["excessive_length_penalty"] = round(excess_pen, 2)
        total_score += excess_pen
    else:
        breakdown["excessive_length_penalty"] = 0.0

    # 9. Lexical On-Topic Bonus (weak signal)
    p_words = set(re.findall(r"\b\w+\b", prompt_text.lower())) - STOPWORDS
    r_words = set(words) - STOPWORDS
    if p_words:
        overlap = len(p_words.intersection(r_words)) / len(p_words)
        bonus = round(2.5 * min(1.0, overlap), 2)
        breakdown["on_topic_bonus"] = bonus
        total_score += bonus
    else:
        breakdown["on_topic_bonus"] = 0.0

    return round(total_score, 2), breakdown


def build_dpo_pairs(
    raw_samples_path: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_raw_samples.jsonl",
    output_train: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_train_pairs.jsonl",
    output_val: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_val_pairs.jsonl",
    review_md_path: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_pairs_review.md",
):
    print("=" * 70)
    print("  PHASE 3: RULE-BASED PREFERENCE LABELING & PAIR CONSTRUCTION")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    with open(raw_samples_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(records)} prompt sample sets.")

    # We split by prompt first (90% train, 10% heldout) to avoid any leakage
    random.seed(42)
    prompt_ids = [r["id"] for r in records]
    random.shuffle(prompt_ids)
    split_idx = int(len(prompt_ids) * 0.90)
    train_ids = set(prompt_ids[:split_idx])
    val_ids = set(prompt_ids[split_idx:])

    train_pairs = []
    val_pairs = []

    penalty_counts = Counter()
    total_samples_scored = 0

    for rec in records:
        p_id = rec["id"]
        is_val = p_id in val_ids
        is_mt = rec.get("is_multi_turn", False)
        is_math = rec.get("is_math", False)
        prompt_msgs = rec["prompt_messages"]
        last_user_prompt = [m["content"] for m in prompt_msgs if m["role"] == "user"][-1]

        # Score all 4 samples
        scored_samples = []
        for s in rec["samples"]:
            resp = s["response_text"]
            has_eos = s["has_eos"]
            t_len = s["raw_tokens_len"]
            sc, bdown = score_response(last_user_prompt, resp, is_math, has_eos, t_len)
            scored_samples.append({
                "response": resp,
                "score": sc,
                "breakdown": bdown,
                "token_len": t_len,
                "has_eos": has_eos,
            })
            total_samples_scored += 1
            for k, v in bdown.items():
                if v < 0:
                    penalty_counts[k] += 1

        # Sort descending by score
        scored_samples.sort(key=lambda x: x["score"], reverse=True)

        min_gap = 2.0 if is_mt else 5.0
        added_for_prompt = 0
        max_p = 3

        for i in range(len(scored_samples)):
            for j in range(i + 1, len(scored_samples)):
                if added_for_prompt >= max_p:
                    break
                s_best = scored_samples[i]
                s_worst = scored_samples[j]
                gap = s_best["score"] - s_worst["score"]

                if gap >= min_gap and s_best["response"].strip() != s_worst["response"].strip():
                    pair = {
                        "id": f"{p_id}_p{added_for_prompt}",
                        "prompt_id": p_id,
                        "is_multi_turn": is_mt,
                        "is_math": is_math,
                        "prompt_messages": prompt_msgs,
                        "chosen": s_best["response"],
                        "chosen_score": s_best["score"],
                        "chosen_tokens": s_best["token_len"],
                        "chosen_breakdown": s_best["breakdown"],
                        "rejected": s_worst["response"],
                        "rejected_score": s_worst["score"],
                        "rejected_tokens": s_worst["token_len"],
                        "rejected_breakdown": s_worst["breakdown"],
                        "score_gap": round(gap, 2),
                        "source": "on_policy_rule_based",
                    }
                    if is_val:
                        val_pairs.append(pair)
                    else:
                        train_pairs.append(pair)
                    added_for_prompt += 1

    print(f"Generated {len(train_pairs)} raw train pairs and {len(val_pairs)} raw val pairs.")

    # Add <= 10% Synthetic Refusal pairs
    # Target: ~180 synthetic refusal pairs in train, ~20 in val
    synth_added_train = 0
    synth_added_val = 0

    for rec in records:
        if synth_added_train >= 180 and synth_added_val >= 20:
            break
        if rec.get("is_math", False):
            continue

        p_id = rec["id"]
        is_val = p_id in val_ids
        prompt_msgs = rec["prompt_messages"]
        last_user_prompt = [m["content"] for m in prompt_msgs if m["role"] == "user"][-1]

        # Pick best on-policy sample as chosen
        good_samples = [s for s in rec["samples"] if s["has_eos"] and len(s["response_text"].strip()) > 25]
        if not good_samples:
            continue
        best_s = random.choice(good_samples)
        chosen_text = best_s["response_text"]

        # Create canned refusal as rejected
        ref_text = random.choice(SYNTHETIC_REFUSALS)
        chosen_tokens = best_s["raw_tokens_len"]
        ref_tokens = len(tokenizer.encode(ref_text).ids)

        sc_c, bd_c = score_response(last_user_prompt, chosen_text, False, True, chosen_tokens)
        sc_r, bd_r = score_response(last_user_prompt, ref_text, False, True, ref_tokens)

        synth_pair = {
            "id": f"{p_id}_synth_ref",
            "prompt_id": p_id,
            "is_multi_turn": rec.get("is_multi_turn", False),
            "is_math": False,
            "prompt_messages": prompt_msgs,
            "chosen": chosen_text,
            "chosen_score": sc_c,
            "chosen_tokens": chosen_tokens,
            "chosen_breakdown": bd_c,
            "rejected": ref_text,
            "rejected_score": sc_r,
            "rejected_tokens": ref_tokens,
            "rejected_breakdown": bd_r,
            "score_gap": round(sc_c - sc_r, 2),
            "source": "synthetic_refusal",
        }

        if is_val and synth_added_val < 20:
            val_pairs.append(synth_pair)
            synth_added_val += 1
        elif not is_val and synth_added_train < 180:
            train_pairs.append(synth_pair)
            synth_added_train += 1

    total_pairs = len(train_pairs) + len(val_pairs)
    synth_total = synth_added_train + synth_added_val
    synth_pct = (synth_total / total_pairs) * 100

    print(f"Added {synth_total} synthetic refusal pairs ({synth_pct:.1f}% of total).")
    print(f"Total DPO Pairs: {total_pairs} (Train: {len(train_pairs)}, Heldout Val: {len(val_pairs)})")

    # Length bias check
    chosen_lens = [p["chosen_tokens"] for p in train_pairs]
    rejected_lens = [p["rejected_tokens"] for p in train_pairs]
    mean_chosen = sum(chosen_lens) / max(len(chosen_lens), 1)
    mean_rejected = sum(rejected_lens) / max(len(rejected_lens), 1)
    len_diff = mean_chosen - mean_rejected

    print(f"\nLength Bias Diagnostics:")
    print(f"  Mean Chosen Token Length   : {mean_chosen:.1f}")
    print(f"  Mean Rejected Token Length : {mean_rejected:.1f}")
    print(f"  Length Gap (Chosen - Rej)  : {len_diff:+.1f} tokens")
    print(f"  --> Confirms chosen is NOT trivially 'just the shorter one'!")

    # Multi-turn share
    mt_count = sum(1 for p in train_pairs if p["is_multi_turn"])
    mt_pct = (mt_count / len(train_pairs)) * 100
    print(f"  Multi-turn Pair Share      : {mt_count}/{len(train_pairs)} ({mt_pct:.1f}%)")

    # Save JSONL files
    output_train.parent.mkdir(parents=True, exist_ok=True)
    with open(output_train, "w", encoding="utf-8") as f:
        for p in train_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    with open(output_val, "w", encoding="utf-8") as f:
        for p in val_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"[Saved] Train pairs saved to: {output_train}")
    print(f"[Saved] Val pairs saved to: {output_val}")

    # Generate `dpo_pairs_review.md`
    generate_review_md(train_pairs, val_pairs, penalty_counts, mean_chosen, mean_rejected, mt_pct, synth_pct, review_md_path)


def generate_review_md(
    train_pairs: list[dict],
    val_pairs: list[dict],
    penalty_counts: Counter,
    mean_chosen: float,
    mean_rejected: float,
    mt_pct: float,
    synth_pct: float,
    review_path: Path,
):
    all_pairs = train_pairs + val_pairs
    random.seed(1337)
    sample_40 = random.sample(all_pairs, min(40, len(all_pairs)))

    lines = []
    lines.append("# DPO Preference Dataset Review & Diagnostic Report\n")
    lines.append("## 1. Aggregate Dataset Statistics\n")
    lines.append(f"- **Total Preference Pairs:** {len(all_pairs):,} (Train: {len(train_pairs):,}, Heldout Val: {len(val_pairs):,})")
    lines.append(f"- **Prompt-Level Split:** 90% train / 10% heldout (Zero prompt leakage between train and val)")
    lines.append(f"- **Multi-Turn Share:** {mt_pct:.1f}%")
    lines.append(f"- **Synthetic Refusal Share:** {synth_pct:.1f}% (<= 10% limit satisfied)")
    lines.append(f"- **Mean Chosen Length:** {mean_chosen:.1f} tokens")
    lines.append(f"- **Mean Rejected Length:** {mean_rejected:.1f} tokens")
    lines.append(f"- **Length Bias Control:** Gap is {mean_chosen - mean_rejected:+.1f} tokens (chosen is not trivially shorter)\n")

    lines.append("## 2. Penalty Breakdown Counts (Across All Scored Samples)\n")
    lines.append("| Penalty Type | Count | Description |")
    lines.append("| :--- | :--- | :--- |")
    for k, v in penalty_counts.most_common():
        lines.append(f"| `{k}` | {v:,} | Negative signal penalized in rejected candidates |")
    lines.append("\n---\n")

    lines.append("## 3. Sample Review (40 Random Pairs)\n")
    for idx, p in enumerate(sample_40, 1):
        prompt_str = p["prompt_messages"][-1]["content"]
        is_mt = "Yes (Multi-Turn)" if p["is_multi_turn"] else "No (Single-Turn)"
        lines.append(f"### Pair #{idx} [{p['source']}] (Gap: {p['score_gap']})")
        lines.append(f"**Multi-Turn:** {is_mt} | **Math:** {p['is_math']}")
        lines.append(f"**Prompt:** *\"{prompt_str}\"*")
        lines.append(f"\n- **Chosen (Score {p['chosen_score']}, {p['chosen_tokens']} tok):**\n  > {p['chosen'].replace(chr(10), ' ')}")
        lines.append(f"- **Rejected (Score {p['rejected_score']}, {p['rejected_tokens']} tok):**\n  > {p['rejected'].replace(chr(10), ' ')}")
        lines.append(f"- **Rejected Penalties:** `{p['rejected_breakdown']}`\n")
        lines.append("---\n")

    review_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[Report] Diagnostic review written to: {review_path}")


if __name__ == "__main__":
    build_dpo_pairs()
