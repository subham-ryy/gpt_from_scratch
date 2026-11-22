"""Systematic Text Generation Experiments on Pretrained GPT (98.4M).

Tests two fixed prompts across 6 controlled sampling hyperparameter regimes:
1. Baseline Balanced:       temp=0.8, top_k=50, top_p=0.95
2. Low Temperature:         temp=0.2, top_k=50, top_p=0.95 (Greedy/exploitative)
3. High Temperature:        temp=1.3, top_k=50, top_p=0.95 (Creative/high-entropy)
4. Strict Top-K Truncation: temp=0.8, top_k=5,  top_p=1.0  (Only top 5 choices)
5. Pure Nucleus Sampling:   temp=0.8, top_k=0,  top_p=0.90 (Dynamic cutoff)
6. Strict Nucleus Cutoff:   temp=0.8, top_k=0,  top_p=0.50 (Top 50% mass only)

Records full text, latency, tokens/sec, unique token ratio, and saves to:
- analysis/sampling_experiments.json
- analysis/sampling_experiments_report.html
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
from generate import find_latest_checkpoint, load_model_and_tokenizer


def run_sampling_experiments():
    print("=" * 70)
    print("  PHASE 6: SYSTEMATIC SAMPLING HYPERPARAMETER EXPERIMENTS")
    print("=" * 70)

    device = config.DEVICE
    ckpt_path = find_latest_checkpoint()
    model, tokenizer = load_model_and_tokenizer(ckpt_path, device=device)

    # 1. Experimental Prompts
    prompts = [
        {
            "id": "prompt_physics",
            "category": "Science / Physics",
            "text": "The law of gravity states that",
        },
        {
            "id": "prompt_cs",
            "category": "Computer Science",
            "text": "In computer science, an algorithm is",
        },
    ]

    # 2. Experimental Configurations Grid
    configs = [
        {
            "id": "exp1_baseline",
            "name": "Baseline Balanced",
            "temp": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "hypothesis": "Balanced fluency with moderate diversity; standard LLM default.",
        },
        {
            "id": "exp2_low_temp",
            "name": "Low Temperature (Exploitative)",
            "temp": 0.2,
            "top_k": 50,
            "top_p": 0.95,
            "hypothesis": "Near-deterministic; highly focused but susceptible to repetitive loops.",
        },
        {
            "id": "exp3_high_temp",
            "name": "High Temperature (High Entropy)",
            "temp": 1.3,
            "top_k": 50,
            "top_p": 0.95,
            "hypothesis": "Increased lexical variety; risk of grammatical degradation or hallucinations.",
        },
        {
            "id": "exp4_strict_topk",
            "name": "Strict Top-K (K=5)",
            "temp": 0.8,
            "top_k": 5,
            "top_p": 1.0,
            "hypothesis": "Severe tail pruning; safe conservative vocabulary choices without rare tokens.",
        },
        {
            "id": "exp5_pure_nucleus",
            "name": "Pure Nucleus (P=0.90)",
            "temp": 0.8,
            "top_k": None,
            "top_p": 0.90,
            "hypothesis": "Dynamic candidate pool adapting to prediction confidence distribution.",
        },
        {
            "id": "exp6_strict_nucleus",
            "name": "Strict Nucleus (P=0.50)",
            "temp": 0.8,
            "top_k": None,
            "top_p": 0.50,
            "hypothesis": "Only top 50% cumulative mass considered; suppresses all speculative continuations.",
        },
    ]

    seed = 42
    max_new_tokens = 90
    eos_id = tokenizer.token_to_id(config.EOS_TOKEN)

    results = []

    print(f"\nRunning {len(configs)} configurations across {len(prompts)} prompts (Fixed Seed: {seed})...\n")

    for p_info in prompts:
        prompt_text = p_info["text"]
        print("\n" + "#" * 70)
        print(f"PROMPT: \"{prompt_text}\" ({p_info['category']})")
        print("#" * 70)

        encoded = tokenizer.encode(prompt_text).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

        for cfg in configs:
            # Set exact seed before each generation for fair controlled comparison
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)

            t0 = time.perf_counter()
            with torch.no_grad():
                out_ids = model.generate(
                    input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=cfg["temp"],
                    top_k=cfg["top_k"],
                    top_p=cfg["top_p"],
                    eos_token_id=eos_id,
                )
            elapsed = time.perf_counter() - t0

            gen_tokens = out_ids[0].tolist()
            new_tokens = gen_tokens[len(encoded):]
            decoded_completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
            full_text = prompt_text + decoded_completion

            speed = len(new_tokens) / max(elapsed, 1e-6)
            unique_tokens = len(set(new_tokens))
            diversity_ratio = unique_tokens / max(len(new_tokens), 1)

            entry = {
                "prompt_id": p_info["id"],
                "prompt_category": p_info["category"],
                "prompt_text": prompt_text,
                "config_id": cfg["id"],
                "config_name": cfg["name"],
                "temperature": cfg["temp"],
                "top_k": cfg["top_k"],
                "top_p": cfg["top_p"],
                "hypothesis": cfg["hypothesis"],
                "tokens_generated": len(new_tokens),
                "speed_tok_s": round(speed, 1),
                "elapsed_s": round(elapsed, 3),
                "unique_token_ratio": round(diversity_ratio, 3),
                "completed_text": decoded_completion.strip(),
                "full_output": full_text.strip(),
            }
            results.append(entry)

            print(f"\n[{cfg['name']}] (T={cfg['temp']}, K={cfg['top_k']}, P={cfg['top_p']})")
            print(f"Tokens/s: {speed:.1f} | Diversity: {diversity_ratio:.2f} ({unique_tokens}/{len(new_tokens)} unique)")
            print(f"Completion: {repr(decoded_completion.strip())}")
            print("-" * 70)

    # 3. Save JSON results
    out_dir = PROJECT_ROOT / "analysis"
    json_path = out_dir / "sampling_experiments.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({
            "model_checkpoint": str(ckpt_path),
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "results": results,
        }, f, indent=2)
    print(f"\n[Saved] Structured JSON data: {json_path}")

    # 4. Generate HTML Report
    html_path = generate_experiments_html(results, ckpt_path, out_dir)
    print(f"[Saved] Interactive HTML Report: {html_path}\n")

    return results


def generate_experiments_html(results: list[dict], ckpt_path: Path, out_dir: Path) -> Path:
    """Generate modern comparative HTML report for sampling experiments."""
    report_file = out_dir / "sampling_experiments_report.html"

    # Group by prompt
    prompts_map = {}
    for r in results:
        pid = r["prompt_id"]
        if pid not in prompts_map:
            prompts_map[pid] = {
                "text": r["prompt_text"],
                "category": r["prompt_category"],
                "experiments": [],
            }
        prompts_map[pid]["experiments"].append(r)

    sections_html = ""
    for pid, pdata in prompts_map.items():
        cards_html = ""
        for exp in pdata["experiments"]:
            cards_html += f"""
            <div class="card">
                <div class="card-header">
                    <span class="cfg-badge">{exp['config_name']}</span>
                    <span class="stats-badge">{exp['speed_tok_s']} tok/s &middot; Div: {exp['unique_token_ratio']}</span>
                </div>
                <div class="params">
                    <code>temp={exp['temperature']}</code>
                    <code>top_k={exp['top_k']}</code>
                    <code>top_p={exp['top_p']}</code>
                </div>
                <div class="hypothesis"><em>Hypothesis:</em> {exp['hypothesis']}</div>
                <div class="output-box">
                    <strong>{exp['prompt_text']}</strong> {exp['completed_text']}
                </div>
            </div>
            """

        sections_html += f"""
        <div class="prompt-section">
            <h2>Prompt: &ldquo;{pdata['text']}&rdquo; <span class="category-tag">{pdata['category']}</span></h2>
            <div class="cards-grid">
                {cards_html}
            </div>
        </div>
        """

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>GPT Pretraining: Sampling Experiments & Evaluation</title>
    <style>
        :root {{
            --primary: #2563EB;
            --primary-bg: #EFF6FF;
            --text-main: #1F2937;
            --text-muted: #6B7280;
            --card-bg: #FFFFFF;
            --bg: #F9FAFB;
            --border: #E5E7EB;
        }}
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background: var(--bg);
            color: var(--text-main);
            margin: 0;
            padding: 30px 20px;
            line-height: 1.6;
        }}
        .container {{ max-width: 1200px; margin: 0 auto; }}
        .header {{
            background: linear-gradient(135deg, #1E3A8A, #2563EB);
            color: white;
            padding: 28px 36px;
            border-radius: 12px;
            margin-bottom: 28px;
        }}
        .header h1 {{ margin: 0 0 8px 0; font-size: 26px; }}
        .header p {{ margin: 0; opacity: 0.9; font-size: 14px; }}
        .prompt-section {{ margin-bottom: 40px; }}
        .prompt-section h2 {{ font-size: 20px; margin-bottom: 16px; border-bottom: 2px solid var(--border); padding-bottom: 8px; }}
        .category-tag {{ font-size: 13px; font-weight: normal; background: #E0E7FF; color: #3730A3; padding: 3px 8px; border-radius: 4px; margin-left: 10px; }}
        .cards-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
            gap: 16px;
        }}
        .card {{
            background: var(--card-bg);
            border: 1px solid var(--border);
            border-radius: 10px;
            padding: 18px;
            display: flex;
            flex-direction: column;
            box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        }}
        .card-header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 10px;
        }}
        .cfg-badge {{ font-weight: 700; color: #1E3A8A; font-size: 14px; }}
        .stats-badge {{ font-size: 12px; background: #F3F4F6; padding: 2px 6px; border-radius: 4px; color: var(--text-muted); }}
        .params {{ margin-bottom: 8px; }}
        .params code {{ background: #F3F4F6; padding: 2px 6px; border-radius: 4px; font-size: 12px; margin-right: 4px; }}
        .hypothesis {{ font-size: 12px; color: var(--text-muted); margin-bottom: 12px; }}
        .output-box {{
            background: #FAFAFA;
            border: 1px solid #E5E7EB;
            border-left: 4px solid var(--primary);
            padding: 14px;
            border-radius: 6px;
            font-size: 14px;
            line-height: 1.6;
            flex-grow: 1;
        }}
    </style>
</head>
<body>
<div class="container">
    <div class="header">
        <h1>GPT Pretraining: Sampling Experiments & Analysis</h1>
        <p>Testing Temperature, Top-K, and Nucleus Top-P Variations on Pretrained GPT (98.4M Parameters)</p>
    </div>

    {sections_html}
</div>
</body>
</html>
"""
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(html)
    return report_file


if __name__ == "__main__":
    run_sampling_experiments()

