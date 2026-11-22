"""Autoregressive text generation and sampling for pretrained GPT (Track A Upgraded).

Features:
- Automatic checkpoint discovery (picks latest best.pt from experiments/)
- Temperature scaling, top-k filtering, and nucleus (top-p) sampling
- Min-P sampling (default: 0.05)
- Repetition penalty (default: 1.15, CTRL/HF formulation)
- No-repeat n-gram blocking (default: 3)
- Single prompt mode, multi-sample mode, and interactive REPL mode (-i)
- Clean console display with prompt highlighting

Usage examples:
    # 1. Quick run with default Track A parameters (T=0.8, top_p=0.9, min_p=0.05, rep=1.15, ngram=3):
    python generate.py

    # 2. Custom prompt with upgraded sampling:
    python generate.py --prompt "Photosynthesis is the process by which"

    # 3. Interactive prompt mode:
    python generate.py -i

    # 4. Custom sampling parameters:
    python generate.py -p "In machine learning," -t 0.7 --min-p 0.08 --rep-penalty 1.2
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

# Ensure UTF-8 output on Windows consoles
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig


STOP_STRINGS = [
    "\n### User:",
    "### User:",
    "\n### Assistant:",
    "### Assistant:",
    "\n###",
]


def clean_stop_strings(text: str) -> str:
    """Rule 7: Stop generation cleanly if the model emits stop markers."""
    for s in STOP_STRINGS:
        pos = text.find(s)
        if pos != -1:
            text = text[:pos].strip()
    return text


def find_latest_checkpoint(explicit_path: Optional[str] = None) -> Path:
    """Find the checkpoint to load: either explicit or latest best.pt from experiments/."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Specified checkpoint not found: {p}")
        return p

    # Search experiments directory in reverse chronological order
    exp_dir = config.EXPERIMENTS_DIR
    if exp_dir.exists():
        # Check subdirectories sorted latest first
        subdirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        for exp in subdirs:
            best_ckpt = exp / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                return best_ckpt
            last_ckpt = exp / "checkpoints" / "last.pt"
            if last_ckpt.exists():
                return last_ckpt

    # Fallback to root checkpoints directory
    if config.CHECKPOINTS_DIR.exists():
        for name in ["best.pt", "last.pt"]:
            p = config.CHECKPOINTS_DIR / name
            if p.exists():
                return p

    raise FileNotFoundError(
        "No checkpoint found. Please provide an explicit path via --checkpoint."
    )


def load_model_and_tokenizer(checkpoint_path: Path, device: str = config.DEVICE):
    """Load tokenizer and GPT model from checkpoint."""
    print(f"[Loading] Tokenizer from: {config.TOKENIZER_FILE.name}")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    print(f"[Loading] Checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    # Reconstruct GPTConfig
    saved_cfg = ckpt.get("config_snapshot", {})
    block_size = saved_cfg.get("block_size", config.BLOCK_SIZE)

    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=block_size,
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,  # Inference mode (deterministic evaluation without dropout)
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )

    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    # Print summary metadata
    opt_step = ckpt.get("opt_step", "N/A")
    val_loss = ckpt.get("val_loss")
    loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else "N/A"
    print(f"[Model] Ready on {device.upper()} | Step: {opt_step} | Val Loss: {loss_str} | Params: {model.get_num_params():,}\n")

    return model, tokenizer


def generate_completion(
    model: GPT,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int = 120,
    temperature: float = config.GEN_TEMPERATURE,
    top_k: Optional[int] = config.GEN_TOP_K,
    top_p: Optional[float] = config.GEN_TOP_P,
    min_p: Optional[float] = config.GEN_MIN_P,
    repetition_penalty: float = config.GEN_REPETITION_PENALTY,
    no_repeat_ngram_size: Optional[int] = config.GEN_NO_REPEAT_NGRAM_SIZE,
    device: str = config.DEVICE,
    chat_format: bool = False,
) -> tuple[str, float]:
    """Generate completion for a given prompt string with full Track A controls."""
    eos_id = tokenizer.token_to_id(config.EOS_TOKEN)

    if chat_format and prompt.strip():
        # SFT Assistant prompt formatting (Rule 7: Support both ### User: and legacy ### Question:)
        bos_id = tokenizer.token_to_id(config.BOS_TOKEN)
        prefix_str = f"### User:\n{prompt.strip()}\n\n### Assistant:\n"
        encoded = [bos_id] + tokenizer.encode(prefix_str).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)
    elif not prompt.strip():
        # Empty prompt fallback: start from BOS token
        bos_id = tokenizer.token_to_id(config.BOS_TOKEN)
        input_ids = torch.tensor([[bos_id]], dtype=torch.long, device=device)
    else:
        encoded = tokenizer.encode(prompt).ids
        input_ids = torch.tensor([encoded], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=eos_id,
        )
    elapsed = time.perf_counter() - t0

    generated_tokens = out_ids[0].tolist()

    if chat_format and prompt.strip():
        # For chat format, extract only the response portion after prefix
        resp_tokens = generated_tokens[input_ids.shape[1] :]
        if eos_id in resp_tokens:
            resp_tokens = resp_tokens[: resp_tokens.index(eos_id)]
        raw_text = tokenizer.decode(resp_tokens, skip_special_tokens=True).strip()
        decoded_text = clean_stop_strings(raw_text)
        tokens_generated = len(resp_tokens)
    else:
        decoded_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
        tokens_generated = len(generated_tokens) - input_ids.shape[1]

    speed = tokens_generated / max(elapsed, 1e-6)
    return decoded_text, speed


def interactive_session(
    model: GPT,
    tokenizer: Tokenizer,
    max_new_tokens: int,
    temperature: float,
    top_k: Optional[int],
    top_p: Optional[float],
    min_p: Optional[float],
    repetition_penalty: float,
    no_repeat_ngram_size: Optional[int],
    device: str,
    chat_format: bool = False,
):
    """Run an interactive console REPL loop."""
    print("=" * 65)
    if chat_format:
        print("  INTERACTIVE GPT ASSISTANT CHAT (Phase 8 SFT)")
        print("  Ask a question and press Enter. (Type 'quit' or 'exit' to stop)")
    else:
        print("  INTERACTIVE GPT SAMPLING SESSION (Track A Upgraded)")
        print("  Type your prompt and press Enter. (Type 'quit' or 'exit' to stop)")
    print(
        f"  Settings: temp={temperature}, top_p={top_p}, top_k={top_k}, "
        f"min_p={min_p}, rep_penalty={repetition_penalty}, ngram_block={no_repeat_ngram_size}"
    )
    print("=" * 65 + "\n")

    while True:
        try:
            prompt_label = "User > " if chat_format else "Prompt >>> "
            prompt = input(prompt_label).strip()
            if not prompt:
                continue
            if prompt.lower() in ["quit", "exit", "q"]:
                print("\nExiting interactive session.")
                break

            print("\nGenerating...", end="\r", flush=True)
            output, speed = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                device=device,
                chat_format=chat_format,
            )

            if chat_format:
                print(f"Assistant: {output}")
                print(f"({speed:.1f} tokens/sec)\n")
            else:
                print("─" * 65)
                print(output)
                print("─" * 65)
                print(f"({speed:.1f} tokens/sec)\n")

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="GPT Autoregressive Text Generation & Assistant Chat")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="Input prompt string.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Path to model checkpoint (.pt).")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=120, help="Max new tokens to generate.")
    parser.add_argument("-t", "--temperature", type=float, default=config.GEN_TEMPERATURE, help="Sampling temperature.")
    parser.add_argument("-k", "--top-k", type=int, default=None, help="Top-K filtering limit (None = disabled).")
    parser.add_argument("--top-p", type=float, default=config.GEN_TOP_P, help="Top-P nucleus probability threshold.")
    parser.add_argument("--min-p", type=float, default=config.GEN_MIN_P, help="Min-P sampling relative threshold.")
    parser.add_argument("--rep-penalty", "--repetition-penalty", type=float, default=config.GEN_REPETITION_PENALTY, help="Repetition penalty (1.0 = disabled).")
    parser.add_argument("--ngram-block", "--no-repeat-ngram-size", type=int, default=config.GEN_NO_REPEAT_NGRAM_SIZE, help="No-repeat n-gram size (0 = disabled).")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive prompt REPL session.")
    parser.add_argument("--chat", action="store_true", help="Format input as SFT assistant prompt (### Question: ... ### Answer:).")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of completions per prompt.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed.")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Execution device (cuda/cpu).")
    args = parser.parse_args()

    if args.seed is not None:
        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)

    # Convert 0 or negative values to None for disabling
    top_k = args.top_k if args.top_k and args.top_k > 0 else None
    ngram_block = args.ngram_block if args.ngram_block and args.ngram_block > 0 else None
    min_p = args.min_p if args.min_p and 0.0 < args.min_p < 1.0 else None

    ckpt_path = find_latest_checkpoint(args.checkpoint)
    model, tokenizer = load_model_and_tokenizer(ckpt_path, device=args.device)

    # 1. Interactive Mode
    if args.interactive:
        interactive_session(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=top_k,
            top_p=args.top_p,
            min_p=min_p,
            repetition_penalty=args.rep_penalty,
            no_repeat_ngram_size=ngram_block,
            device=args.device,
            chat_format=args.chat,
        )
        return

    # 2. Single or Custom Prompt Mode
    if args.prompt is not None:
        prompts = [args.prompt]
    else:
        # Default educational prompts reflecting FineWeb-Edu training corpus
        prompts = [
            "The solar system consists of",
            "In computer science, an algorithm is",
            "Photosynthesis is the biological process where",
            "One of the most significant events in world history was",
        ]
        print(f"No prompt provided. Running demonstration across {len(prompts)} sample prompts.\n")

    for prompt in prompts:
        print("=" * 65)
        print(f"[PROMPT]: {prompt}")
        print("=" * 65)

        for s in range(args.num_samples):
            if args.num_samples > 1:
                print(f"--- Sample {s + 1} ---")

            completion, speed = generate_completion(
                model=model,
                tokenizer=tokenizer,
                prompt=prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=top_k,
                top_p=args.top_p,
                min_p=min_p,
                repetition_penalty=args.rep_penalty,
                no_repeat_ngram_size=ngram_block,
                device=args.device,
                chat_format=args.chat,
            )
            print(completion)
            print(f"\n[Generated at {speed:.1f} tokens/s]")
            print("-" * 65)
        print()


if __name__ == "__main__":
    main()
