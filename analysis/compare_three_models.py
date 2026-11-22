"""Comprehensive 3-Model Benchmark: GPT-v1 vs Track A vs Phase 8 SFT.

Compares:
1. Model 1 (Phase 6 Baseline): 5k docs, step 1200, val_loss 5.2793
2. Model 2 (Track A Pretrained): 100k docs packed, step 9000, val_loss 3.7055
3. Model 3 (Phase 8 SFT Assistant): 65k Q&A pairs, step 4000, val_loss 2.6224
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

import matplotlib.pyplot as plt
import torch
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

MODEL_PATHS = {
    "Model 1 (Phase 6 Base)": {
        "path": PROJECT_ROOT / "experiments" / "20261104_211024_gpt_pretrain" / "checkpoints" / "best.pt",
        "step": 1200,
        "val_loss": 5.2793,
        "ppl": 196.23,
        "bpb": 1.7915,
        "type": "pretrain_base",
    },
    "Model 2 (Track A Pretrained)": {
        "path": PROJECT_ROOT / "experiments" / "20261106_032904_gpt_continued_pretrain" / "checkpoints" / "best.pt",
        "step": 9000,
        "val_loss": 3.7055,
        "ppl": 40.67,
        "bpb": 1.2574,
        "type": "continued_pretrain",
    },
    "Model 3 (Phase 8 SFT Assistant)": {
        "path": PROJECT_ROOT / "experiments" / "20261106_115736_gpt_sft" / "checkpoints" / "best.pt",
        "step": 4000,
        "val_loss": 2.6224,
        "ppl": 13.77,
        "bpb": 0.8902,
        "type": "sft_assistant",
    },
}

TEST_PROMPTS = [
    {
        "id": "france",
        "category": "Factual Geography",
        "raw_prompt": "The capital of France is",
        "qa_prompt": "What is the capital of France?",
    },
    {
        "id": "photosynthesis",
        "category": "Science / Biology",
        "raw_prompt": "Photosynthesis is the process by which",
        "qa_prompt": "What is photosynthesis?",
    },
    {
        "id": "algorithm",
        "category": "Computer Science",
        "raw_prompt": "In computer science, an algorithm is defined as",
        "qa_prompt": "In computer science, what is an algorithm?",
    },
    {
        "id": "machine_learning",
        "category": "Artificial Intelligence",
        "raw_prompt": "Machine learning is a field of",
        "qa_prompt": "What is machine learning?",
    },
    {
        "id": "brain",
        "category": "Neuroscience",
        "raw_prompt": "The human brain consists of",
        "qa_prompt": "What are the main functions of the human brain?",
    },
    {
        "id": "habits",
        "category": "Instruction Following",
        "raw_prompt": "Three healthy habits for better sleep are",
        "qa_prompt": "List three healthy habits for better sleep.",
    },
    {
        "id": "story",
        "category": "Narrative Prose",
        "raw_prompt": "Once upon a time",
        "qa_prompt": "Tell me a short story starting with: Once upon a time",
    },
]


def load_model(checkpoint_path: Path) -> GPT:
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt.get("config_snapshot", {})
    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=saved_cfg.get("block_size", config.BLOCK_SIZE),
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )
    model = GPT(model_config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def generate_for_model(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    is_sft_format: bool = False,
    max_new_tokens: int = 90,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: float = 0.08,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
) -> tuple[str, float]:
    bos_id = 1
    eos_id = 2

    if is_sft_format:
        prefix_str = f"### Question:\n{prompt.strip()}\n\n### Answer:\n"
        encoded = [bos_id] + tokenizer.encode(prefix_str).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=DEVICE)
    else:
        encoded = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=DEVICE)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=None,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_id,
        )
    elapsed = time.perf_counter() - t0

    generated = out_ids[0].tolist()

    if is_sft_format:
        resp = generated[input_ids.shape[1] :]
        if eos_id in resp:
            resp = resp[: resp.index(eos_id)]
        text = tokenizer.decode(resp, skip_special_tokens=True).strip()
        tok_count = len(resp)
    else:
        text = tokenizer.decode(generated, skip_special_tokens=True).strip()
        tok_count = len(generated) - input_ids.shape[1]

    speed = tok_count / max(elapsed, 1e-6)
    return text, speed


def create_comparison_plot(output_path: Path):
    """Generate bar charts for validation metrics across all 3 models."""
    models = ["Model 1\n(Phase 6 Base)", "Model 2\n(Track A Pretrain)", "Model 3\n(Phase 8 SFT)"]
    losses = [5.2793, 3.7055, 2.6224]
    ppls = [196.23, 40.67, 13.77]
    bpbs = [1.7915, 1.2574, 0.8902]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    colors = ["#4A90E2", "#50E3C2", "#F5A623"]

    # 1. Validation Loss
    bars1 = axes[0].bar(models, losses, color=colors, width=0.55)
    axes[0].set_title("Validation Loss (nats) ↓", fontsize=12, fontweight="bold")
    axes[0].set_ylim(0, 6.0)
    for bar in bars1:
        yval = bar.get_height()
        axes[0].text(bar.get_x() + bar.get_width() / 2, yval + 0.1, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

    # 2. Perplexity
    bars2 = axes[1].bar(models, ppls, color=colors, width=0.55)
    axes[1].set_title("Perplexity (PPL) ↓", fontsize=12, fontweight="bold")
    axes[1].set_ylim(0, 220)
    for bar in bars2:
        yval = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width() / 2, yval + 3.0, f"{yval:.1f}", ha="center", va="bottom", fontweight="bold")

    # 3. Bits Per Byte
    bars3 = axes[2].bar(models, bpbs, color=colors, width=0.55)
    axes[2].set_title("Bits Per Byte (BPB) ↓", fontsize=12, fontweight="bold")
    axes[2].set_ylim(0, 2.1)
    for bar in bars3:
        yval = bar.get_height()
        axes[2].text(bar.get_x() + bar.get_width() / 2, yval + 0.04, f"{yval:.4f}", ha="center", va="bottom", fontweight="bold")

    for ax in axes:
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.set_axisbelow(True)

    plt.suptitle("Evolution of 98.4M GPT: Base Pretraining → Track A Continued Pretraining → Phase 8 SFT", fontsize=13, fontweight="bold", y=1.03)
    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"[Plot Saved] {output_path}")


def main():
    print("=" * 70)
    print("  3-MODEL COMPARATIVE EVALUATION BENCHMARK")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    serializable_models = {
        m_name: {**m_info, "path": str(m_info["path"])}
        for m_name, m_info in MODEL_PATHS.items()
    }
    results = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "models": serializable_models, "prompts": []}

    # Store loaded model instances
    model_completions: dict[str, dict[str, dict]] = {m_name: {} for m_name in MODEL_PATHS}

    for model_name, model_info in MODEL_PATHS.items():
        ckpt_path = model_info["path"]
        print(f"\n[Benchmarking] Loading {model_name} from {ckpt_path.name}...")
        if not ckpt_path.exists():
            print(f"  Warning: Checkpoint not found: {ckpt_path}")
            continue

        model = load_model(ckpt_path)
        is_sft = (model_info["type"] == "sft_assistant")

        for p_item in TEST_PROMPTS:
            p_id = p_item["id"]
            # For SFT, test both Q&A format and raw prompt; for base models, test raw prompt and formatted Q&A
            prompt_to_use = p_item["qa_prompt"] if is_sft else p_item["raw_prompt"]
            text, speed = generate_for_model(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt_to_use,
                is_sft_format=is_sft,
            )
            model_completions[model_name][p_id] = {
                "prompt_used": prompt_to_use,
                "text": text,
                "speed": round(speed, 1),
            }
            print(f"  [{p_item['category']}] ({speed:.1f} tok/s): {text[:90]}...")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    for p_item in TEST_PROMPTS:
        p_id = p_item["id"]
        results["prompts"].append({
            "id": p_id,
            "category": p_item["category"],
            "raw_prompt": p_item["raw_prompt"],
            "qa_prompt": p_item["qa_prompt"],
            "completions": {
                m_name: model_completions[m_name].get(p_id, {})
                for m_name in MODEL_PATHS
            },
        })

    # Save JSON results
    out_json = PROJECT_ROOT / "analysis" / "three_models_comparison.json"
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[Saved JSON] {out_json}")

    # Generate comparison plot
    plot_path = PROJECT_ROOT / "analysis" / "plots" / "three_models_metrics_comparison.png"
    create_comparison_plot(plot_path)

    # Copy plot to brain artifact directory
    artifact_dir = Path(r"C:\Users\rupar\.gemini\antigravity\brain\60461eed-366b-4384-8e0d-82baa726f007")
    if artifact_dir.exists():
        import shutil
        shutil.copy(plot_path, artifact_dir / "three_models_metrics_comparison.png")
        shutil.copy(out_json, artifact_dir / "three_models_comparison.json")
        print(f"[Copied to Artifact Dir] {artifact_dir / 'three_models_metrics_comparison.png'}")

    print("\n[Done] 3-Model evaluation completed successfully!")


if __name__ == "__main__":
    main()
