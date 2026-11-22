"""Dataset analysis module for FineWeb-Edu.

Inspects document lengths, educational scores, languages, character frequencies,
generates distribution plots, and creates a comprehensive HTML report.
"""
import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

# Add project root to sys.path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import config


def analyze_dataset(
    input_file: Path = config.RAW_DOCUMENTS_FILE,
    output_dir: Path = config.BASE_DIR / "analysis",
) -> dict:
    """Read documents from JSONL and calculate comprehensive dataset statistics."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found at: {input_file.resolve()}")

    doc_count = 0
    char_lengths = []
    word_lengths = []
    scores = []
    int_scores = Counter()
    languages = Counter()
    dumps = Counter()
    char_counts = Counter()
    empty_anomalous_count = 0
    sample_docs = []

    print(f"Analyzing documents from: {input_file.resolve()}...")

    with open(input_file, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            data = json.loads(line)
            text = data.get("text", "")

            if not text or not text.strip():
                empty_anomalous_count += 1
                continue

            doc_count += 1
            c_len = len(text)
            w_len = len(text.split())

            char_lengths.append(c_len)
            word_lengths.append(w_len)

            # Collect metadata
            score = data.get("score")
            if score is not None:
                try:
                    scores.append(float(score))
                except (ValueError, TypeError):
                    pass

            int_score = data.get("int_score")
            if int_score is not None:
                int_scores[int_score] += 1

            lang = data.get("language", "unknown")
            languages[lang] += 1

            dump = data.get("dump", "unknown")
            dumps[dump] += 1

            # Update character counts (first 1000 chars per doc to keep fast)
            char_counts.update(text[:1000])

            # Keep 3 samples for inspection
            if len(sample_docs) < 3:
                sample_docs.append({
                    "id": data.get("id"),
                    "url": data.get("url"),
                    "score": data.get("score"),
                    "chars": c_len,
                    "words": w_len,
                    "preview": text[:300].strip() + ("..." if c_len > 300 else ""),
                })

    if doc_count == 0:
        raise ValueError("No valid documents found to analyze.")

    # Calculate statistics
    char_lengths.sort()
    word_lengths.sort()

    total_chars = sum(char_lengths)
    total_words = sum(word_lengths)
    mean_char_len = total_chars / doc_count
    mean_word_len = total_words / doc_count
    median_char_len = char_lengths[doc_count // 2]
    median_word_len = word_lengths[doc_count // 2]
    min_char_len = char_lengths[0]
    max_char_len = char_lengths[-1]

    variance = sum((x - mean_char_len) ** 2 for x in char_lengths) / doc_count
    std_char_len = math.sqrt(variance)

    mean_score = sum(scores) / len(scores) if scores else 0.0

    # Character breakdown
    total_sampled_chars = sum(char_counts.values())
    ascii_chars = sum(count for char, count in char_counts.items() if ord(char) < 128)
    ascii_ratio = (ascii_chars / total_sampled_chars) if total_sampled_chars > 0 else 0.0

    stats = {
        "dataset_name": config.DATASET_NAME,
        "dataset_config": config.DATASET_CONFIG,
        "total_documents": doc_count,
        "empty_or_anomalous": empty_anomalous_count,
        "total_characters": total_chars,
        "total_words": total_words,
        "char_length": {
            "min": min_char_len,
            "max": max_char_len,
            "mean": round(mean_char_len, 2),
            "median": median_char_len,
            "std": round(std_char_len, 2),
        },
        "word_length": {
            "min": word_lengths[0],
            "max": word_lengths[-1],
            "mean": round(mean_word_len, 2),
            "median": median_word_len,
        },
        "educational_score": {
            "mean": round(mean_score, 2),
            "int_score_distribution": dict(sorted(int_scores.items())),
        },
        "language_distribution": dict(languages.most_common(5)),
        "common_crawl_dumps": dict(dumps.most_common(5)),
        "character_stats": {
            "unique_chars_sampled": len(char_counts),
            "ascii_percentage": round(ascii_ratio * 100, 2),
            "top_10_chars": [{"char": repr(c), "count": cnt} for c, cnt in char_counts.most_common(10)],
        },
        "samples": sample_docs,
    }

    # Generate Visualizations
    # 1. Document Length Histogram
    plt.figure(figsize=(9, 4.5))
    # Filter 95th percentile for readable visualization
    cutoff = char_lengths[int(doc_count * 0.95)]
    filtered_lengths = [l for l in char_lengths if l <= cutoff]
    plt.hist(filtered_lengths, bins=40, color="#2563eb", edgecolor="white", alpha=0.85)
    plt.axvline(mean_char_len, color="#dc2626", linestyle="dashed", linewidth=1.5, label=f"Mean ({mean_char_len:.0f})")
    plt.axvline(median_char_len, color="#16a34a", linestyle="dotted", linewidth=1.5, label=f"Median ({median_char_len})")
    plt.title("FineWeb-Edu: Document Length Distribution (chars)", fontsize=13, fontweight="bold")
    plt.xlabel("Characters per Document")
    plt.ylabel("Number of Documents")
    plt.legend()
    plt.tight_layout()
    hist_path = plots_dir / "doc_length_histogram.png"
    plt.savefig(hist_path, dpi=150)
    plt.close()

    # 2. Educational Score Distribution
    if scores:
        plt.figure(figsize=(9, 4.5))
        plt.hist(scores, bins=25, range=(0, 5), color="#059669", edgecolor="white", alpha=0.85)
        plt.axvline(mean_score, color="#dc2626", linestyle="dashed", linewidth=1.5, label=f"Mean ({mean_score:.2f})")
        plt.title("FineWeb-Edu: Educational Quality Score Distribution", fontsize=13, fontweight="bold")
        plt.xlabel("Quality Score (0.0 to 5.0)")
        plt.ylabel("Document Count")
        plt.legend()
        plt.tight_layout()
        score_path = plots_dir / "educational_score_distribution.png"
        plt.savefig(score_path, dpi=150)
        plt.close()

    # Save JSON Report
    json_path = output_dir / "dataset_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    # Save HTML Report
    html_path = output_dir / "dataset_report.html"
    generate_dataset_html_report(stats, html_path, plots_dir)

    print("\nDataset Analysis Complete!")
    print(f"  Summary JSON: {json_path.resolve()}")
    print(f"  HTML Report:  {html_path.resolve()}")
    print(f"  Plots Dir:    {plots_dir.resolve()}")
    return stats


def generate_dataset_html_report(stats: dict, output_path: Path, plots_dir: Path) -> None:
    """Generate an accessible, styled HTML report for the dataset exploration."""
    samples_html = "".join([
        f"""<div class="sample-box">
            <p><strong>ID:</strong> <code>{s['id']}</code> | <strong>Score:</strong> {s['score']} | <strong>Length:</strong> {s['chars']:,} chars ({s['words']:,} words)</p>
            <p><strong>URL:</strong> <a href="{s['url']}" target="_blank">{s['url']}</a></p>
            <pre class="preview">{s['preview']}</pre>
        </div>"""
        for s in stats["samples"]
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>FineWeb-Edu Dataset Exploration Report</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; line-height: 1.6; color: #1e293b; background: #f8fafc; margin: 0; padding: 24px; }}
    .container {{ max-width: 960px; margin: 0 auto; background: #ffffff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    h1 {{ color: #0f172a; margin-top: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
    h2 {{ color: #1e3a8a; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 16px; margin: 20px 0; }}
    .card {{ background: #f1f5f9; padding: 16px; border-radius: 8px; border-left: 4px solid #2563eb; }}
    .card h3 {{ margin: 0 0 4px 0; font-size: 13px; text-transform: uppercase; color: #64748b; }}
    .card p {{ margin: 0; font-size: 24px; font-weight: bold; color: #0f172a; }}
    table {{ width: 100%; border-collapse: collapse; margin: 16px 0; }}
    th, td {{ text-align: left; padding: 10px 14px; border-bottom: 1px solid #e2e8f0; }}
    th {{ background: #f8fafc; color: #475569; }}
    .plot-img {{ width: 100%; max-width: 800px; border-radius: 8px; border: 1px solid #e2e8f0; margin: 12px 0; }}
    .sample-box {{ background: #f8fafc; border: 1px solid #cbd5e1; border-radius: 8px; padding: 14px; margin-bottom: 12px; }}
    .preview {{ background: #ffffff; padding: 10px; border-radius: 6px; border: 1px solid #e2e8f0; font-size: 13px; white-space: pre-wrap; word-break: break-word; }}
</style>
</head>
<body>
<div class="container">
    <h1>FineWeb-Edu Dataset Exploration Report</h1>
    <p><strong>Dataset:</strong> <code>{stats['dataset_name']}</code> (config: <code>{stats['dataset_config']}</code>)</p>

    <div class="grid">
        <div class="card"><h3>Total Documents</h3><p>{stats['total_documents']:,}</p></div>
        <div class="card"><h3>Total Characters</h3><p>{stats['total_characters']:,}</p></div>
        <div class="card"><h3>Total Words</h3><p>{stats['total_words']:,}</p></div>
        <div class="card"><h3>Mean Doc Length</h3><p>{stats['char_length']['mean']:.0f} chars</p></div>
        <div class="card"><h3>Median Doc Length</h3><p>{stats['char_length']['median']:,} chars</p></div>
        <div class="card"><h3>Mean Edu Score</h3><p>{stats['educational_score']['mean']:.2f} / 5.0</p></div>
    </div>

    <h2>Document Length Statistics</h2>
    <table>
        <tr><th>Metric</th><th>Characters</th><th>Words</th></tr>
        <tr><td>Minimum</td><td>{stats['char_length']['min']:,}</td><td>{stats['word_length']['min']:,}</td></tr>
        <tr><td>Mean</td><td>{stats['char_length']['mean']:,}</td><td>{stats['word_length']['mean']:,}</td></tr>
        <tr><td>Median</td><td>{stats['char_length']['median']:,}</td><td>{stats['word_length']['median']:,}</td></tr>
        <tr><td>Maximum</td><td>{stats['char_length']['max']:,}</td><td>{stats['word_length']['max']:,}</td></tr>
        <tr><td>Standard Deviation</td><td>{stats['char_length']['std']:,}</td><td>-</td></tr>
    </table>

    <img class="plot-img" src="plots/doc_length_histogram.png" alt="Document Length Distribution">

    <h2>Educational Quality Score Distribution</h2>
    <p>Educational scores generated via classification models on FineWeb documents:</p>
    <table>
        <tr><th>Integer Score (Bucket)</th><th>Document Count</th></tr>
        {"".join([f"<tr><td>Score {k}</td><td>{v:,}</td></tr>" for k, v in stats['educational_score']['int_score_distribution'].items()])}
    </table>

    <img class="plot-img" src="plots/educational_score_distribution.png" alt="Educational Score Distribution">

    <h2>Character & Vocabulary Characteristics</h2>
    <p>Sampled ASCII character representation: <strong>{stats['character_stats']['ascii_percentage']}%</strong> (clean, standard English educational content).</p>

    <h2>Document Samples</h2>
    {samples_html}
</div>
</body>
</html>
"""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)


def main():
    parser = argparse.ArgumentParser(description="Analyze FineWeb-Edu sampled dataset.")
    parser.add_argument("--input-file", type=str, default=str(config.RAW_DOCUMENTS_FILE), help="Path to documents.jsonl.")
    args = parser.parse_args()

    analyze_dataset(input_file=Path(args.input_file))


if __name__ == "__main__":
    main()

