"""Tokenizer analysis and research documentation module.

Analyzes Byte-Level BPE token frequencies, compression ratios, vocabulary
distributions, encoding/decoding throughput on train and validation splits,
and generates publication-quality plots and an HTML research report.
"""
import argparse
import csv
import datetime
import json
import math
import shutil
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
from tokenizers import Tokenizer

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from monitoring.system_monitor import SystemMonitor


def analyze_corpus_tokenization(
    tokenizer: Tokenizer,
    jsonl_path: Path,
    split_name: str = "train",
) -> tuple[dict, Counter]:
    """Tokenize a JSONL corpus, track token counts, and benchmark throughput."""
    if not jsonl_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {jsonl_path.resolve()}")

    docs = []
    total_chars = 0
    total_bytes = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            txt = item.get("text", "")
            if txt:
                docs.append(txt)
                total_chars += len(txt)
                total_bytes += len(txt.encode("utf-8"))

    doc_count = len(docs)
    print(f"[{split_name.upper()}] Encoding {doc_count:,} documents ({total_chars:,} chars, {total_bytes:,} bytes)...")

    # Benchmark encoding
    t0 = time.perf_counter()
    encodings = tokenizer.encode_batch(docs)
    encode_time = max(time.perf_counter() - t0, 1e-6)

    # Collect token stats
    token_counter = Counter()
    total_tokens = 0
    all_token_ids = []

    for enc in encodings:
        ids = enc.ids
        total_tokens += len(ids)
        token_counter.update(ids)
        if len(all_token_ids) < 100_000:
            all_token_ids.extend(ids)

    # Benchmark decoding on sample
    sample_ids = all_token_ids[:min(len(all_token_ids), 50_000)]
    t_dec0 = time.perf_counter()
    _ = tokenizer.decode(sample_ids)
    decode_time = max(time.perf_counter() - t_dec0, 1e-6)
    decode_throughput = len(sample_ids) / decode_time

    chars_per_token = total_chars / max(1, total_tokens)
    bytes_per_token = total_bytes / max(1, total_tokens)
    tokens_per_sec = total_tokens / encode_time
    bytes_per_sec = total_bytes / encode_time

    metrics = {
        "split": split_name,
        "documents": doc_count,
        "total_characters": total_chars,
        "total_bytes": total_bytes,
        "total_tokens": total_tokens,
        "chars_per_token": round(chars_per_token, 3),
        "bytes_per_token": round(bytes_per_token, 3),
        "encode_duration_sec": round(encode_time, 3),
        "encode_tokens_per_sec": round(tokens_per_sec, 1),
        "encode_bytes_per_sec": round(bytes_per_sec, 1),
        "decode_tokens_per_sec": round(decode_throughput, 1),
    }

    print(f"  --> Total tokens:     {total_tokens:,}")
    print(f"  --> Chars / token:    {chars_per_token:.2f}")
    print(f"  --> Bytes / token:    {bytes_per_token:.2f}")
    print(f"  --> Encode speed:     {tokens_per_sec:,.0f} tokens/s ({bytes_per_sec / (1024*1024):.1f} MB/s)")
    print(f"  --> Decode speed:     {decode_throughput:,.0f} tokens/s\n")

    return metrics, token_counter


def compute_vocabulary_statistics(tokenizer: Tokenizer, train_counter: Counter) -> dict:
    """Analyze vocabulary frequencies, Zipfian distribution, rare tokens, and token lengths."""
    vocab_size = tokenizer.get_vocab_size()

    # Token string lengths across entire vocabulary
    token_lengths = []
    for tok_id in range(vocab_size):
        raw_token = tokenizer.id_to_token(tok_id)
        # Decode display string cleanly
        try:
            tok_str = tokenizer.decode([tok_id])
        except Exception:
            tok_str = raw_token
        token_lengths.append(len(tok_str))

    token_lengths.sort()
    mean_tok_len = sum(token_lengths) / len(token_lengths)
    median_tok_len = token_lengths[len(token_lengths) // 2]

    # Frequency analysis
    counts = [train_counter.get(i, 0) for i in range(vocab_size)]
    counts_sorted = sorted(counts, reverse=True)

    singletons = sum(1 for c in counts if c == 1)
    under_5 = sum(1 for c in counts if 0 < c <= 5)
    under_10 = sum(1 for c in counts if 0 < c <= 10)
    unused = sum(1 for c in counts if c == 0)

    # Top 20 most frequent tokens
    top_20 = []
    total_tokens = sum(train_counter.values())
    for tok_id, count in train_counter.most_common(20):
        try:
            decoded_rep = tokenizer.decode([tok_id])
        except Exception:
            decoded_rep = tokenizer.id_to_token(tok_id)

        pct = (count / total_tokens) * 100 if total_tokens > 0 else 0.0
        top_20.append({
            "token_id": tok_id,
            "raw_token": tokenizer.id_to_token(tok_id),
            "decoded_representation": repr(decoded_rep),
            "count": count,
            "percentage": round(pct, 3),
        })

    return {
        "vocab_size": vocab_size,
        "token_length_stats": {
            "min": token_lengths[0],
            "max": token_lengths[-1],
            "mean": round(mean_tok_len, 2),
            "median": median_tok_len,
        },
        "rare_tokens": {
            "singletons_count_1": singletons,
            "tokens_count_lte_5": under_5,
            "tokens_count_lte_10": under_10,
            "unused_tokens_count_0": unused,
        },
        "top_20_tokens": top_20,
        "counts_sorted": counts_sorted,
        "token_lengths": token_lengths,
    }


def generate_tokenizer_plots(
    vocab_stats: dict,
    train_metrics: dict,
    val_metrics: dict,
    plots_dir: Path,
) -> list[Path]:
    """Generate 4 publication-quality visualization charts for the tokenizer."""
    plots_dir.mkdir(parents=True, exist_ok=True)
    generated = []

    # 1. Token Frequency Histogram (log scale)
    counts = [c for c in vocab_stats["counts_sorted"] if c > 0]
    plt.figure(figsize=(8, 4.5))
    plt.hist(counts, bins=50, color="#2563eb", edgecolor="white", alpha=0.85, log=True)
    plt.xscale("log")
    plt.title("Token Frequency Distribution in Training Corpus (Log-Log)", fontsize=12, fontweight="bold")
    plt.xlabel("Token Occurrence Count (Log Scale)")
    plt.ylabel("Number of Vocabulary Tokens (Log Scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p1 = plots_dir / "tokenizer_token_freq_histogram.png"
    plt.savefig(p1, dpi=150)
    plt.close()
    generated.append(p1)

    # 2. Rank vs. Frequency Zipfian Plot
    ranks = list(range(1, len(counts) + 1))
    plt.figure(figsize=(8, 4.5))
    plt.plot(ranks, counts, color="#059669", linewidth=2.0)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Zipfian Token Rank vs. Frequency", fontsize=12, fontweight="bold")
    plt.xlabel("Token Frequency Rank (Log Scale)")
    plt.ylabel("Token Count (Log Scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    p2 = plots_dir / "tokenizer_rank_freq_plot.png"
    plt.savefig(p2, dpi=150)
    plt.close()
    generated.append(p2)

    # 3. Token Length Distribution
    token_lens = vocab_stats["token_lengths"]
    plt.figure(figsize=(8, 4.5))
    plt.hist(token_lens, bins=range(1, max(token_lens) + 2), color="#7c3aed", edgecolor="white", alpha=0.85, align="left")
    plt.axvline(vocab_stats["token_length_stats"]["mean"], color="#dc2626", linestyle="dashed", linewidth=1.5,
                label=f"Mean ({vocab_stats['token_length_stats']['mean']:.1f} chars)")
    plt.axvline(vocab_stats["token_length_stats"]["median"], color="#16a34a", linestyle="dotted", linewidth=1.5,
                label=f"Median ({vocab_stats['token_length_stats']['median']} chars)")
    plt.title("Token Length Distribution (Characters per Token)", fontsize=12, fontweight="bold")
    plt.xlabel("Decoded Token Length (Characters)")
    plt.ylabel("Vocabulary Count")
    plt.legend()
    plt.tight_layout()
    p3 = plots_dir / "tokenizer_token_length_dist.png"
    plt.savefig(p3, dpi=150)
    plt.close()
    generated.append(p3)

    # 4. Train vs Validation Compression Comparison
    labels = ["Chars / Token", "Bytes / Token"]
    train_vals = [train_metrics["chars_per_token"], train_metrics["bytes_per_token"]]
    val_vals = [val_metrics["chars_per_token"], val_metrics["bytes_per_token"]]

    x = [0, 1]
    width = 0.35
    plt.figure(figsize=(7, 4.5))
    b1 = plt.bar([p - width/2 for p in x], train_vals, width=width, label="Train Split", color="#3b82f6")
    b2 = plt.bar([p + width/2 for p in x], val_vals, width=width, label="Validation (Held-out)", color="#10b981")
    plt.title("Tokenizer Compression: Train vs. Validation Split", fontsize=12, fontweight="bold")
    plt.xticks(x, labels)
    plt.ylabel("Ratio (Higher = More Compressed)")
    plt.ylim(0, max(train_vals + val_vals) * 1.25)
    plt.legend()

    for bars in [b1, b2]:
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2, height + 0.05, f"{height:.2f}x", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    p4 = plots_dir / "tokenizer_compression_stats.png"
    plt.savefig(p4, dpi=150)
    plt.close()
    generated.append(p4)

    return generated


def generate_tokenizer_html_report(
    full_stats: dict,
    output_path: Path,
    plots: list[Path],
) -> Path:
    """Generate an accessible, publication-ready HTML report for the tokenizer."""
    train_m = full_stats["train_metrics"]
    val_m = full_stats["validation_metrics"]
    vocab_s = full_stats["vocabulary_statistics"]

    top_tokens_rows = "".join([
        f"<tr><td><code>{t['token_id']}</code></td><td><code>{t['raw_token']}</code></td><td>{t['decoded_representation']}</td><td>{t['count']:,}</td><td>{t['percentage']}%</td></tr>"
        for t in vocab_s["top_20_tokens"]
    ])

    plot_cards_html = "".join([
        f'<div class="plot-card"><img src="plots/{p.name}" alt="{p.stem}"></div>'
        for p in plots
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FineWeb-Edu Byte-Level BPE Tokenizer Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; line-height: 1.6; }}
    .container {{ max-width: 1040px; margin: 0 auto; background: #ffffff; padding: 36px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    h1 {{ color: #0f172a; margin-top: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
    h2 {{ color: #1e3a8a; margin-top: 30px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(210px, 1fr)); gap: 14px; margin: 20px 0; }}
    .card {{ background: #f1f5f9; padding: 14px 18px; border-radius: 8px; border-left: 4px solid #2563eb; }}
    .card h3 {{ margin: 0 0 4px 0; font-size: 11px; text-transform: uppercase; color: #64748b; letter-spacing: 0.5px; }}
    .card p {{ margin: 0; font-size: 22px; font-weight: 700; color: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; margin: 18px 0; }}
    th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f8fafc; color: #475569; font-weight: 600; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(460px, 1fr)); gap: 18px; margin: 20px 0; }}
    .plot-card img {{ width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }}
    .badge {{ display: inline-block; background: #e0f2fe; color: #0369a1; padding: 3px 8px; border-radius: 4px; font-weight: 600; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
    <h1>Byte-Level BPE Tokenizer Research Report</h1>
    <p>Trained strictly on FineWeb-Edu training documents with <span class="badge">16,384 Vocabulary</span> and <span class="badge">Hugging Face Tokenizers</span>.</p>

    <div class="grid">
        <div class="card"><h3>Vocabulary Size</h3><p>{vocab_s['vocab_size']:,}</p></div>
        <div class="card"><h3>Training Documents</h3><p>{train_m['documents']:,}</p></div>
        <div class="card"><h3>Training Tokens</h3><p>{train_m['total_tokens']:,}</p></div>
        <div class="card"><h3>Train Compression</h3><p>{train_m['chars_per_token']:.2f} chars/tok</p></div>
        <div class="card"><h3>Val Compression</h3><p>{val_m['chars_per_token']:.2f} chars/tok</p></div>
        <div class="card"><h3>Encode Throughput</h3><p>{train_m['encode_tokens_per_sec']:,.0f} tok/s</p></div>
    </div>

    <h2>Corpus Compression & Throughput Performance</h2>
    <table>
        <tr>
            <th>Metric</th>
            <th>Training Split (4,500 docs)</th>
            <th>Validation Split (500 docs, Held-out)</th>
        </tr>
        <tr><td>Total Characters</td><td>{train_m['total_characters']:,}</td><td>{val_m['total_characters']:,}</td></tr>
        <tr><td>Total UTF-8 Bytes</td><td>{train_m['total_bytes']:,}</td><td>{val_m['total_bytes']:,}</td></tr>
        <tr><td>Total Tokens</td><td>{train_m['total_tokens']:,}</td><td>{val_m['total_tokens']:,}</td></tr>
        <tr><td><strong>Characters per Token</strong></td><td><strong>{train_m['chars_per_token']:.3f}</strong></td><td><strong>{val_m['chars_per_token']:.3f}</strong></td></tr>
        <tr><td><strong>Bytes per Token</strong></td><td><strong>{train_m['bytes_per_token']:.3f}</strong></td><td><strong>{val_m['bytes_per_token']:.3f}</strong></td></tr>
        <tr><td>Encoding Throughput</td><td>{train_m['encode_tokens_per_sec']:,.0f} tokens/s ({train_m['encode_bytes_per_sec'] / (1024*1024):.1f} MB/s)</td><td>{val_m['encode_tokens_per_sec']:,.0f} tokens/s</td></tr>
        <tr><td>Decoding Throughput</td><td>{train_m['decode_tokens_per_sec']:,.0f} tokens/s</td><td>-</td></tr>
    </table>

    <h2>Token Distribution & Zipfian Characteristics</h2>
    <div class="plot-grid">
        {plot_cards_html}
    </div>

    <h2>Vocabulary Frequency & Rare Tokens</h2>
    <table>
        <tr><th>Category</th><th>Count</th><th>Notes</th></tr>
        <tr><td>Singletons (occurred once)</td><td>{vocab_s['rare_tokens']['singletons_count_1']:,}</td><td>Rare words/stems</td></tr>
        <tr><td>Tokens with count &le; 5</td><td>{vocab_s['rare_tokens']['tokens_count_lte_5']:,}</td><td>Low-frequency merges</td></tr>
        <tr><td>Tokens with count &le; 10</td><td>{vocab_s['rare_tokens']['tokens_count_lte_10']:,}</td><td>Infrequent tokens</td></tr>
        <tr><td>Unused tokens (count = 0)</td><td>{vocab_s['rare_tokens']['unused_tokens_count_0']:,}</td><td>Reserved special tokens (<|pad|>, <|bos|>, <|eos|>)</td></tr>
        <tr><td>Average Token Length</td><td>{vocab_s['token_length_stats']['mean']:.2f} chars</td><td>Median: {vocab_s['token_length_stats']['median']} chars (Range: {vocab_s['token_length_stats']['min']} - {vocab_s['token_length_stats']['max']} chars)</td></tr>
    </table>

    <h2>Top 20 Most Frequent Tokens</h2>
    <table>
        <tr><th>Token ID</th><th>Raw Representation</th><th>Decoded</th><th>Corpus Frequency</th><th>% of Total Tokens</th></tr>
        {top_tokens_rows}
    </table>
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path


def run_tokenizer_analysis(
    tokenizer_file: Path = config.TOKENIZER_FILE,
    train_file: Path = config.TRAIN_DOCUMENTS_FILE,
    val_file: Path = config.VAL_DOCUMENTS_FILE,
    analysis_dir: Path = config.BASE_DIR / "analysis",
    experiments_dir: Path = config.EXPERIMENTS_DIR,
) -> dict:
    """Run full analysis, generate plots, compile HTML report, and log experiment record."""
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"Tokenizer artifact not found at {tokenizer_file.resolve()}. Run train_tokenizer.py first.")

    tokenizer = Tokenizer.from_file(str(tokenizer_file))
    print(f"Loaded tokenizer from: {tokenizer_file.resolve()} (Vocab: {tokenizer.get_vocab_size():,})\n")

    # 1. Benchmark on Training and Validation splits
    train_metrics, train_counter = analyze_corpus_tokenization(tokenizer, train_file, split_name="train")
    val_metrics, _ = analyze_corpus_tokenization(tokenizer, val_file, split_name="val")

    # 2. Vocabulary Statistics
    vocab_stats = compute_vocabulary_statistics(tokenizer, train_counter)

    # 3. Generate Visualizations
    plots_dir = analysis_dir / "plots"
    plots = generate_tokenizer_plots(vocab_stats, train_metrics, val_metrics, plots_dir)

    # Clean stats for JSON export (remove raw lists for compactness)
    export_vocab_stats = {k: v for k, v in vocab_stats.items() if k not in ("counts_sorted", "token_lengths")}
    full_stats = {
        "timestamp": datetime.datetime.now().isoformat(),
        "tokenizer_file": str(tokenizer_file.resolve()),
        "vocab_size": tokenizer.get_vocab_size(),
        "train_metrics": train_metrics,
        "validation_metrics": val_metrics,
        "vocabulary_statistics": export_vocab_stats,
    }

    # 4. Save analysis JSON and HTML report
    stats_json_path = analysis_dir / "tokenizer_stats.json"
    with open(stats_json_path, "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)

    report_html_path = analysis_dir / "tokenizer_report.html"
    generate_tokenizer_html_report(full_stats, report_html_path, plots)

    print("=" * 70)
    print("SAVING TOKENIZER EXPERIMENT TRACKING RECORD")
    print("=" * 70)

    # 5. Create immutable experiment tracking record
    timestamp_str = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_id = f"{timestamp_str}_tokenizer_bpe_{tokenizer.get_vocab_size()}"
    exp_dir = experiments_dir / exp_id
    exp_plots_dir = exp_dir / "plots"
    exp_plots_dir.mkdir(parents=True, exist_ok=True)

    # Copy artifacts to experiment record
    shutil.copy2(tokenizer_file, exp_dir / "tokenizer.json")
    if (tokenizer_file.parent / "tokenizer_config.json").exists():
        shutil.copy2(tokenizer_file.parent / "tokenizer_config.json", exp_dir / "tokenizer_config.json")

    with open(exp_dir / "tokenizer_stats.json", "w", encoding="utf-8") as f:
        json.dump(full_stats, f, indent=2)

    # Snapshot system info
    sys_info = SystemMonitor.get_hardware_info()
    sys_info["experiment_id"] = exp_id
    sys_info["task"] = "tokenizer_training_and_evaluation"
    with open(exp_dir / "system_info.json", "w", encoding="utf-8") as f:
        json.dump(sys_info, f, indent=2)

    # Copy plots & report
    exp_plots = []
    for p in plots:
        dst = exp_plots_dir / p.name
        shutil.copy2(p, dst)
        exp_plots.append(dst)

    generate_tokenizer_html_report(full_stats, exp_dir / "tokenizer_report.html", exp_plots)

    # Machine-readable summary CSV
    with open(exp_dir / "metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["split", "documents", "tokens", "chars", "bytes", "chars_per_token", "bytes_per_token", "tokens_per_sec"])
        writer.writerow(["train", train_metrics["documents"], train_metrics["total_tokens"], train_metrics["total_characters"], train_metrics["total_bytes"], train_metrics["chars_per_token"], train_metrics["bytes_per_token"], train_metrics["encode_tokens_per_sec"]])
        writer.writerow(["val", val_metrics["documents"], val_metrics["total_tokens"], val_metrics["total_characters"], val_metrics["total_bytes"], val_metrics["chars_per_token"], val_metrics["bytes_per_token"], val_metrics["encode_tokens_per_sec"]])

    print(f"Created experiment record: {exp_dir.resolve()}")
    print(f"  Artifacts:    {exp_dir / 'tokenizer.json'}")
    print(f"  Metrics CSV:  {exp_dir / 'metrics.csv'}")
    print(f"  HTML Report:  {exp_dir / 'tokenizer_report.html'}")
    print(f"  Plots:        {exp_plots_dir.resolve()}")

    return full_stats


def main():
    parser = argparse.ArgumentParser(description="Analyze trained Byte-Level BPE tokenizer.")
    parser.add_argument("--tokenizer-file", type=str, default=str(config.TOKENIZER_FILE), help="Path to tokenizer.json.")
    parser.add_argument("--train-file", type=str, default=str(config.TRAIN_DOCUMENTS_FILE), help="Path to train.jsonl.")
    parser.add_argument("--val-file", type=str, default=str(config.VAL_DOCUMENTS_FILE), help="Path to val.jsonl.")
    args = parser.parse_args()

    run_tokenizer_analysis(
        tokenizer_file=Path(args.tokenizer_file),
        train_file=Path(args.train_file),
        val_file=Path(args.val_file),
    )


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    main()

