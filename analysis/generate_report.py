"""Generates a standalone HTML research report for an individual experiment."""
import argparse
import json
from pathlib import Path

from analysis.plot_training import plot_experiment_metrics


def generate_experiment_report(run_dir: Path) -> Path:
    """Generate plots and compile a self-contained HTML experiment research report."""
    run_dir = Path(run_dir)
    plots = plot_experiment_metrics(run_dir)

    config_path = run_dir / "config_snapshot.json"
    system_path = run_dir / "system_info.json"
    samples_path = run_dir / "samples.txt"
    html_path = run_dir / "experiment_report.html"

    cfg = {}
    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

    sys_info = {}
    if system_path.exists():
        with open(system_path, "r", encoding="utf-8") as f:
            sys_info = json.load(f)

    samples_text = ""
    if samples_path.exists():
        with open(samples_path, "r", encoding="utf-8") as f:
            samples_text = f.read()

    # Find plots
    plot_images_html = "".join([
        f'<div class="plot-card"><img src="plots/{p.name}" alt="{p.stem}"></div>'
        for p in plots
    ])

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Experiment Report: {run_dir.name}</title>
<style>
    body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; background: #f8fafc; color: #1e293b; margin: 0; padding: 24px; }}
    .container {{ max-width: 1000px; margin: 0 auto; background: #ffffff; padding: 32px; border-radius: 12px; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
    h1 {{ color: #0f172a; margin-top: 0; border-bottom: 2px solid #e2e8f0; padding-bottom: 12px; }}
    h2 {{ color: #1e3a8a; margin-top: 28px; border-bottom: 1px solid #e2e8f0; padding-bottom: 8px; }}
    .meta-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 20px 0; }}
    .meta-card {{ background: #f1f5f9; padding: 12px 16px; border-radius: 8px; border-left: 4px solid #3b82f6; }}
    .meta-card h3 {{ margin: 0 0 4px 0; font-size: 11px; text-transform: uppercase; color: #64748b; }}
    .meta-card p {{ margin: 0; font-size: 15px; font-weight: 600; color: #0f172a; word-break: break-all; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(450px, 1fr)); gap: 16px; margin: 20px 0; }}
    .plot-card img {{ width: 100%; border-radius: 8px; border: 1px solid #e2e8f0; }}
    pre.code {{ background: #0f172a; color: #f8fafc; padding: 16px; border-radius: 8px; font-size: 13px; overflow-x: auto; white-space: pre-wrap; }}
</style>
</head>
<body>
<div class="container">
    <h1>GPT Pretraining Experiment Report</h1>
    <p><strong>Experiment ID:</strong> <code>{run_dir.name}</code></p>

    <h2>Environment & Hardware</h2>
    <div class="meta-grid">
        <div class="meta-card"><h3>Device</h3><p>{sys_info.get('device_name', 'N/A')}</p></div>
        <div class="meta-card"><h3>Platform</h3><p>{sys_info.get('platform', 'N/A')}</p></div>
        <div class="meta-card"><h3>PyTorch / CUDA</h3><p>{sys_info.get('torch_version', 'N/A')} / {sys_info.get('cuda_version', 'N/A')}</p></div>
        <div class="meta-card"><h3>Random Seed</h3><p>{sys_info.get('random_seed', 'N/A')}</p></div>
        <div class="meta-card"><h3>Total RAM</h3><p>{sys_info.get('total_ram_gb', 'N/A')} GB</p></div>
        <div class="meta-card"><h3>Timestamp</h3><p>{sys_info.get('timestamp', 'N/A')}</p></div>
    </div>

    <h2>Hyperparameters & Configuration</h2>
    <div class="meta-grid">
        {"".join([f'<div class="meta-card"><h3>{k}</h3><p>{v}</p></div>' for k, v in list(cfg.items())[:12] if not k.startswith("experiment_")])}
    </div>

    <h2>Training & Optimization Dynamics</h2>
    <div class="plot-grid">
        {plot_images_html}
    </div>

    <h2>Generated Text Samples</h2>
    {"<pre class='code'>" + samples_text + "</pre>" if samples_text else "<p><em>No generation samples recorded for this run.</em></p>"}
</div>
</body>
</html>
"""
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"[Report] Generated research report: {html_path.resolve()}")
    return html_path


def main():
    parser = argparse.ArgumentParser(description="Generate HTML report for an experiment.")
    parser.add_argument("run_dir", type=str, help="Path to experiment directory.")
    args = parser.parse_args()
    generate_experiment_report(Path(args.run_dir))


if __name__ == "__main__":
    main()

