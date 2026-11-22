"""Phase 1: Build Non-Contaminated Prompt Pool for DPO (~2,000 Prompts).

Sources:
1. SFT Validation Split (`data/sft_multi_turn/multi_turn_val.jsonl`) — Held out from SFT training.
2. SmolTalk `everyday-conversations` test split (`test-00000-of-00001.parquet`) — Completely unused in SFT.
3. Everyday personal-situation prompts across realistic templates.

Guarantees:
- Exactly 2,000 prompts total.
- Exactly ~30% (580 - 620) multi-turn conversation prefixes ending on a user turn.
- <= 10% math prompts.
- Zero contamination with evaluation sets (asserted via normalized text matching).
- Context lengths strictly <= 400 tokens.
- All prompts strictly deduplicated.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyarrow.parquet as pq
from tokenizers import Tokenizer
import config
from dpo.eval_prompts import get_all_eval_prompts_normalized


EVERYDAY_TEMPLATES = [
    # Tech / Hardware
    "My laptop fan is running at maximum speed even when idle. What could be the cause?",
    "My smartphone battery drains from 100% to 20% in just a few hours. What should I check?",
    "My home Wi-Fi keeps disconnecting every 15 minutes. How can I troubleshoot this?",
    "I accidentally spilled a small amount of water near my mechanical keyboard. What steps should I take?",
    "My computer screen occasionally flickers when opening video applications. What could be causing it?",
    "My external hard drive is not showing up when plugged into the USB port. How can I fix it?",
    "My Bluetooth headphones keep stuttering while connected to my laptop. Any suggestions?",
    "How do I safely clean dust out of desktop computer cooling fans?",
    "My browser is consuming excessive RAM with only a few tabs open. What can I do?",
    "My phone storage is almost completely full of 'System' and cached data. How can I clean it?",
    "How do I set up a secondary monitor on Windows 11 with HDMI?",
    "What causes a laptop trackpad to become unresponsive intermittently?",
    "How can I test if my laptop charger is delivering sufficient power?",
    "What is the best way to back up essential files before formatting a hard drive?",
    "Why does my computer take five minutes to boot up on Windows?",
    
    # Study / Productivity / Career
    "I have trouble remembering concepts from dense textbook chapters. What active recall method works best?",
    "How can I structure a 2-hour daily study block to maximize retention?",
    "What are effective ways to prepare for a behavioral interview question about handling conflict?",
    "How should I organize my digital notes across multiple courses or projects?",
    "I struggle with afternoon brain fog around 2 PM. What are healthy ways to regain focus?",
    "What are some practical techniques to stop procrastinating on starting a major project?",
    "How can I efficiently summarize a 20-page research paper for quick review?",
    "What is a good strategy to prepare a concise 5-minute lightning talk?",
    "How do I politely decline an extra task when my work schedule is already at capacity?",
    "What are the key elements to include in a follow-up email after a job interview?",
    "How can I break down a 10-page essay into manageable daily writing goals?",
    "What are good techniques for staying attentive during long online lecture recordings?",
    "How do I prepare an elevator pitch for my background in under 60 seconds?",
    "What is the difference between active recall and spaced repetition in studying?",
    "How can I organize my inbox so important project emails do not get lost?",
    
    # Fitness / Health / Nutrition
    "What are beginner-friendly core exercises that do not put pressure on the lower back?",
    "How can I properly stretch tight hamstrings and calves after a running session?",
    "What are simple, high-protein breakfast ideas that take less than 10 minutes to prepare?",
    "How much water should an active adult drink on hot days versus cooler days?",
    "What are simple mobility exercises to relieve stiffness from sitting at a desk all day?",
    "How can I build up my stamina from walking to running a continuous 5K?",
    "What are the most common mistakes beginners make when starting weightlifting?",
    "What are healthy snack alternatives to chips and sugary snacks during long work sessions?",
    "How can I reduce neck strain when using a dual-monitor setup?",
    "What is an effective evening wind-down routine to fall asleep faster without screens?",
    "How many rest days per week should a beginner taking up strength training have?",
    "What are simple upper-body stretches to relieve tension in the shoulders?",
    "How can I prevent muscle cramps during long outdoor runs?",
    "What are good carbohydrate sources for sustaining energy throughout an athletic workout?",
    "How do I adjust my office chair to maintain proper ergonomic posture?",
    
    # Cooking / Kitchen Basics
    "What is the best way to store fresh leafy greens so they stay crisp for over a week?",
    "How do I properly season and maintain a cast-iron skillet?",
    "What are easy substitutes for eggs when baking cookies or muffins?",
    "Why did my homemade bread turn out dense rather than light and airy?",
    "What spices pair well with roasted root vegetables like carrots and sweet potatoes?",
    "How do I prevent chicken breast from drying out when pan-searing?",
    "What is the difference between baking powder and baking soda in baking recipes?",
    "How can I make a flavorful vegetable soup broth from kitchen scraps?",
    "What are safe food safety guidelines for defrosting meat in a refrigerator?",
    "How do I caramelize onions properly without burning them?",
    "What is the best oil for high-heat stir-frying in a wok?",
    "How can I fix a soup or sauce that turned out too salty?",
    "How long can cooked rice be safely stored in the refrigerator?",
    "What are common mistakes when cooking pasta that cause it to stick together?",
    "How do I sharpen kitchen knives safely using a whetstone?",
    
    # Home / Everyday Maintenance
    "How do I effectively remove hard water stains from glass shower doors?",
    "What are simple ways to organize a small closet with limited hanging space?",
    "How can I prevent indoor house plants from getting root rot during winter months?",
    "What is an easy method to remove grease stains from cotton clothing?",
    "How do I properly clean and descale a drip coffee maker with vinegar?",
    "What are essential tools to keep in a basic home repair tool kit?",
    "How can I keep my apartment cool during summer without central air conditioning?",
    "What is the best way to pack delicate glassware when moving to a new home?",
    "How do I stop a door from squeaking using household items?",
    "What are easy ways to organize cables and charging cords behind a desk?",
    "How do I safely clean a laptop screen without leaving streaks or scratches?",
    "What are practical tips for keeping a refrigerator clean and odor-free?",
    "How do I replace a blown fuse or reset a tripped circuit breaker?",
    "What is the proper way to care for wooden cutting boards?",
    "How can I safely unclog a slow-draining bathroom sink?",
]

EVERYDAY_VARIATIONS = [
    "Can you provide some practical tips for {action}?",
    "What are the most effective steps to {action}?",
    "I need advice on how to {action}. Where should I begin?",
    "What are common pitfalls to avoid when trying to {action}?",
]

ADDITIONAL_TOPICS = [
    "saving money on groceries without sacrificing nutrition",
    "training a puppy not to jump on guests",
    "cleaning white sneakers that have grass stains",
    "maintaining a bicycle chain and gears",
    "organizing digital photos and backing them up safely",
    "preparing for a weekend camping trip in unpredictable weather",
    "reducing daily household plastic waste",
    "learning to touch-type without looking at the keyboard",
    "packing a single carry-on suitcase for a 7-day trip",
    "improving vocal clarity when speaking in meetings",
    "setting up proper lighting for video conference calls",
    "starting a small balcony herb garden for basil and mint",
    "troubleshooting a slow home drainage sink",
    "building an emergency 72-hour family supply kit",
    "cleaning and conditioning leather boots",
    "managing time when working remotely from home",
    "creating an effective grocery shopping list to reduce food waste",
    "preventing blisters during long hiking trips",
    "safely removing mold from bathroom grout",
    "cooking dried beans from scratch without a pressure cooker",
    "creating a basic monthly personal budget spreadsheet",
    "improving posture while driving long distances",
    "protecting outdoor wooden patio furniture from rain",
    "practicing active listening in conversations with team members",
    "safely storing seasonal winter clothes to prevent moth damage",
]


def normalize_text(text: str) -> str:
    cleaned = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(cleaned.split())


def build_pool():
    print("=" * 70)
    print("  PHASE 1: BUILDING NON-CONTAMINATED DPO PROMPT POOL")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    eval_norm_set = get_all_eval_prompts_normalized()
    print(f"Loaded {len(eval_norm_set)} normalized evaluation prompts for exclusion.")

    pool: list[dict] = []
    seen_prompts: set[str] = set()

    TARGET_TOTAL = 2000
    TARGET_MULTI_TURN = 600   # 30.0%
    MAX_MATH = 150            # 7.5% <= 10%

    multi_turn_count = 0
    math_count = 0
    everyday_count = 0

    # 1. Multi-Turn Source A: Cached test split of SmolTalk `everyday-conversations`
    test_parquet = Path(
        r"C:\Users\rupar\.cache\huggingface\hub\datasets--HuggingFaceTB--smoltalk\snapshots\5feaf2fd3ffca7c237fc38d1861bc30365d48ffa\data\everyday-conversations\test-00000-of-00001.parquet"
    )
    if test_parquet.exists():
        table = pq.read_table(test_parquet)
        print(f"Loaded {len(table)} unused multi-turn conversations from SmolTalk test parquet.")
        for row in table["messages"]:
            raw_msgs = row.as_py()
            # Clean system messages
            clean_msgs = [m for m in raw_msgs if m.get("role") in ("user", "assistant") and m.get("content", "").strip()]
            if len(clean_msgs) < 4:
                continue

            for user_idx in range(2, len(clean_msgs), 2):
                if clean_msgs[user_idx]["role"] != "user":
                    continue

                prefix_msgs = clean_msgs[: user_idx + 1]
                ctx_text = "".join(f"{m['role']}: {m['content']}\n" for m in prefix_msgs)
                tok_len = len(tokenizer.encode(ctx_text).ids)
                if tok_len > 400:
                    continue

                norm = normalize_text(prefix_msgs[-1]["content"])
                if norm in eval_norm_set or norm in seen_prompts:
                    continue

                seen_prompts.add(norm)
                pool.append({
                    "id": f"pool_mt_{multi_turn_count:04d}",
                    "category": "smoltalk_test_multiturn",
                    "is_multi_turn": True,
                    "is_math": False,
                    "prompt_messages": prefix_msgs,
                    "token_length": tok_len,
                })
                multi_turn_count += 1
                if multi_turn_count >= TARGET_MULTI_TURN:
                    break
            if multi_turn_count >= TARGET_MULTI_TURN:
                break

    print(f"Extracted {multi_turn_count} multi-turn prefixes from SmolTalk test split.")

    # 2. Multi-Turn Source B: `multi_turn_val.jsonl`
    val_jsonl = PROJECT_ROOT / "data" / "sft_multi_turn" / "multi_turn_val.jsonl"
    val_records = []
    with open(val_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                val_records.append(json.loads(line))

    random.seed(42)
    random.shuffle(val_records)

    for rec in val_records:
        if multi_turn_count >= TARGET_MULTI_TURN:
            break

        messages = rec.get("messages", [])
        if len(messages) < 4:
            continue

        cat = rec.get("category", "general")
        is_math = ("math" in cat.lower() or "arithmetic" in cat.lower())
        if is_math and math_count >= MAX_MATH:
            continue

        for user_idx in range(2, len(messages), 2):
            if messages[user_idx]["role"] != "user":
                continue

            prefix_msgs = messages[: user_idx + 1]
            ctx_text = "".join(f"{m['role']}: {m['content']}\n" for m in prefix_msgs)
            tok_len = len(tokenizer.encode(ctx_text).ids)
            if tok_len > 400:
                continue

            norm = normalize_text(prefix_msgs[-1]["content"])
            if norm in eval_norm_set or norm in seen_prompts:
                continue

            seen_prompts.add(norm)
            pool.append({
                "id": f"pool_mt_{multi_turn_count:04d}",
                "category": cat,
                "is_multi_turn": True,
                "is_math": is_math,
                "prompt_messages": prefix_msgs,
                "token_length": tok_len,
            })
            multi_turn_count += 1
            if is_math:
                math_count += 1
            if multi_turn_count >= TARGET_MULTI_TURN:
                break

    print(f"Total multi-turn prefixes collected: {multi_turn_count} ({multi_turn_count / TARGET_TOTAL * 100:.1f}%)")

    # 3. Add Everyday Personal-Situation Prompts (~200 prompts)
    for q in EVERYDAY_TEMPLATES:
        norm = normalize_text(q)
        if norm not in eval_norm_set and norm not in seen_prompts:
            seen_prompts.add(norm)
            pool.append({
                "id": f"pool_st_{len(pool):04d}",
                "category": "everyday_single",
                "is_multi_turn": False,
                "is_math": False,
                "prompt_messages": [{"role": "user", "content": q}],
                "token_length": len(tokenizer.encode(q).ids),
            })
            everyday_count += 1

    for topic in ADDITIONAL_TOPICS:
        for tmpl in EVERYDAY_VARIATIONS:
            q = tmpl.format(action=topic)
            norm = normalize_text(q)
            if norm not in eval_norm_set and norm not in seen_prompts:
                seen_prompts.add(norm)
                pool.append({
                    "id": f"pool_st_{len(pool):04d}",
                    "category": "everyday_single",
                    "is_multi_turn": False,
                    "is_math": False,
                    "prompt_messages": [{"role": "user", "content": q}],
                    "token_length": len(tokenizer.encode(q).ids),
                })
                everyday_count += 1
                if everyday_count >= 200:
                    break
        if everyday_count >= 200:
            break

    print(f"Added {everyday_count} everyday personal-situation prompts.")

    # 4. Fill Remaining Single-Turn Slots from `multi_turn_val.jsonl`
    for rec in val_records:
        if len(pool) >= TARGET_TOTAL:
            break

        messages = rec.get("messages", [])
        if not messages or messages[0]["role"] != "user":
            continue

        cat = rec.get("category", "general")
        is_math = ("math" in cat.lower() or "arithmetic" in cat.lower())
        if is_math and math_count >= MAX_MATH:
            continue

        user_msg = messages[0]["content"].strip()
        tok_len = len(tokenizer.encode(user_msg).ids)
        if tok_len > 400 or tok_len < 3:
            continue

        norm = normalize_text(user_msg)
        if norm in eval_norm_set or norm in seen_prompts:
            continue

        seen_prompts.add(norm)
        pool.append({
            "id": f"pool_st_{len(pool):04d}",
            "category": cat,
            "is_multi_turn": False,
            "is_math": is_math,
            "prompt_messages": [{"role": "user", "content": user_msg}],
            "token_length": tok_len,
        })
        if is_math:
            math_count += 1

    # 5. Summary Statistics & Acceptance Checks
    print(f"\nFinal Pool Statistics:")
    print(f"  Total Prompts in Pool : {len(pool)}")
    print(f"  Multi-turn Prompts    : {multi_turn_count} ({multi_turn_count / len(pool) * 100:.1f}%) [Target: ~30%]")
    print(f"  Math Prompts          : {math_count} ({math_count / len(pool) * 100:.1f}%) [Target: <= 10%]")
    print(f"  Everyday Prompts      : {everyday_count}")

    assert 1950 <= len(pool) <= 2050, f"Expected ~2000 prompts, got {len(pool)}"
    assert 580 <= multi_turn_count <= 620, f"Expected ~600 multi-turn (30%), got {multi_turn_count}"
    assert math_count <= 200, f"Expected <= 10% math, got {math_count}"

    print("\nRunning strict zero-contamination assertion check...")
    for item in pool:
        for m in item["prompt_messages"]:
            if m["role"] == "user":
                n = normalize_text(m["content"])
                assert n not in eval_norm_set, f"CONTAMINATION DETECTED! '{m['content']}' matches eval set."
        assert item["token_length"] <= 400, f"Token length {item['token_length']} exceeds 400 cap!"

    print("--> PASS: 0% Contamination confirmed. All context lengths <= 400 tokens.")

    # 6. Save to JSONL
    out_dir = PROJECT_ROOT / "data" / "dpo"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "dpo_prompt_pool.jsonl"

    with open(out_file, "w", encoding="utf-8") as f:
        for item in pool:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"[Saved] Prompt pool saved to: {out_file}")
    print("=" * 70)


if __name__ == "__main__":
    build_pool()
