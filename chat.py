"""Interactive Assistant & Multi-Turn Chat Console (Phase 4 / Phase 8 SFT).

Features:
- Full Multi-Turn Conversation Memory: Tracks context across turns up to 1,024 tokens.
- Dual Format Support:
    Multi-Turn (Phase 4):
        ### User:\n{msg}\n\n### Assistant:\n{reply}<|eos|>\n\n### User:\n{msg2}...
    Single-Turn (Phase 8):
        ### Question:\n{msg}\n\n### Answer:\n
- Rule 7 Stop Criteria:
    - Stops on <|eos|> (id 2)
    - Truncates on stop sequences ("### User:", "\n### User:", "### Assistant:")
- Factual & Balanced Decoding Controls:
    - Temperature: 0.4
    - Top-P: 0.90
    - Min-P: 0.08
    - Repetition penalty: 1.15
    - No-repeat 4-gram blocking

Commands during chat:
    /clear or /reset : Clear conversation history
    exit or quit     : Exit chat

Usage:
    python chat.py
    python chat.py -p "What is photosynthesis?"
    python chat.py -c experiments/.../checkpoints/best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Optional

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer

import config
from model.transformer import GPT, GPTConfig

BOS_ID = 1
EOS_ID = 2
FINAL_DPO_CHECKPOINT = Path("D:/pytorch/gpt_pretraining/experiments/20261122_014711_dpo_lr1e6_b01_20261122_014642/checkpoints/best.pt")

STOP_STRINGS = [
    "\n### User:",
    "### User:",
    "\n### Assistant:",
    "### Assistant:",
    "\n###",
]


def find_latest_checkpoint(explicit_path: Optional[str] = None) -> Path:
    """Find latest checkpoint: explicit -> multi-turn SFT best.pt -> single-turn SFT -> general best.pt."""
    if explicit_path:
        p = Path(explicit_path)
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")
        return p

    if "FINAL_DPO_CHECKPOINT" in globals() and FINAL_DPO_CHECKPOINT.exists():
        return FINAL_DPO_CHECKPOINT

    exp_dir = config.EXPERIMENTS_DIR
    if exp_dir.exists():
        # First preference: multi-turn SFT experiments
        multi_exps = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and "multi_turn" in d.name.lower()],
            reverse=True,
        )
        for exp in multi_exps:
            best_ckpt = exp / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                return best_ckpt
            last_ckpt = exp / "checkpoints" / "last.pt"
            if last_ckpt.exists():
                return last_ckpt

        # Second preference: other SFT experiments
        sft_exps = sorted(
            [d for d in exp_dir.iterdir() if d.is_dir() and "sft" in d.name.lower()],
            reverse=True,
        )
        for exp in sft_exps:
            best_ckpt = exp / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                return best_ckpt
            last_ckpt = exp / "checkpoints" / "last.pt"
            if last_ckpt.exists():
                return last_ckpt

        # Third preference: any recent experiment
        all_exps = sorted([d for d in exp_dir.iterdir() if d.is_dir()], reverse=True)
        for exp in all_exps:
            best_ckpt = exp / "checkpoints" / "best.pt"
            if best_ckpt.exists():
                return best_ckpt

    if config.CHECKPOINTS_DIR.exists():
        p = config.CHECKPOINTS_DIR / "best.pt"
        if p.exists():
            return p

    raise FileNotFoundError("No checkpoint found. Please provide path via --checkpoint.")


def load_assistant_model(checkpoint_path: Path, device: str = config.DEVICE) -> tuple[GPT, Tokenizer, dict]:
    """Load tokenizer and GPT model configured for inference."""
    print(f"[Loading] Tokenizer from: {config.TOKENIZER_FILE.name}")
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))

    print(f"[Loading] Checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location=device, weights_only=False)

    saved_cfg = ckpt.get("config_snapshot", {})
    block_size = saved_cfg.get("block_size", config.BLOCK_SIZE)

    model_config = GPTConfig.from_tokenizer_config(
        config_path=config.TOKENIZER_CONFIG_FILE,
        block_size=block_size,
        n_layer=saved_cfg.get("n_layer", config.N_LAYER),
        n_head=saved_cfg.get("n_head", config.N_HEAD),
        n_embd=saved_cfg.get("n_embd", config.N_EMBD),
        ffn_dim=saved_cfg.get("ffn_dim", config.FFN_DIM),
        dropout=0.0,
        bias=saved_cfg.get("bias", config.BIAS),
        tie_weights=saved_cfg.get("tie_weights", config.TIE_WEIGHTS),
        use_flash=True,
    )

    model = GPT(model_config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    opt_step = ckpt.get("opt_step", "N/A")
    val_loss = ckpt.get("val_loss", "N/A")
    loss_str = f"{val_loss:.4f}" if isinstance(val_loss, (int, float)) else str(val_loss)
    model_name = saved_cfg.get("model_name", "GPT-98.4M")
    print(f"[Model] {model_name} ready on {device.upper()} | Step: {opt_step} | Val Loss: {loss_str} | Params: {model.get_num_params():,}\n")

    return model, tokenizer, saved_cfg


def format_conversation_tokens(
    tokenizer: Tokenizer,
    history: list[dict[str, str]],
    new_user_message: str,
    max_context_len: int = 1024,
    use_multi_turn_format: bool = True,
) -> list[int]:
    """Format full multi-turn dialogue into token IDs, managing context window length."""
    if not use_multi_turn_format:
        # Legacy Phase 8 Single Turn Format
        prefix_str = f"### Question:\n{new_user_message.strip()}\n\n### Answer:\n"
        return [BOS_ID] + tokenizer.encode(prefix_str).ids

    input_ids = [BOS_ID]

    # Pre-encode tokens for turns
    for turn in history:
        role = turn["role"]
        content = turn["content"].strip()
        if role == "user":
            turn_ids = tokenizer.encode("### User:\n").ids + tokenizer.encode(f"{content}\n\n").ids
            input_ids.extend(turn_ids)
        elif role == "assistant":
            turn_ids = (
                tokenizer.encode("### Assistant:\n").ids
                + tokenizer.encode(content).ids
                + [EOS_ID]
                + tokenizer.encode("\n\n").ids
            )
            input_ids.extend(turn_ids)

    # Append current user message and assistant header
    cur_user_ids = tokenizer.encode("### User:\n").ids + tokenizer.encode(f"{new_user_message.strip()}\n\n").ids
    asst_header_ids = tokenizer.encode("### Assistant:\n").ids
    input_ids.extend(cur_user_ids + asst_header_ids)

    # If context exceeds max_context_len - 150 (leaving room for response), truncate oldest turns
    max_prompt_len = max_context_len - 150
    if len(input_ids) > max_prompt_len:
        # Keep BOS and slice from tail
        input_ids = [BOS_ID] + input_ids[-(max_prompt_len - 1):]

    return input_ids


def clean_response(raw_text: str) -> str:
    """Apply Rule 7 stop criteria: stop at EOS and truncate any self-hallucinated headers."""
    text = raw_text.strip()
    for stop_str in STOP_STRINGS:
        pos = text.find(stop_str)
        if pos != -1:
            text = text[:pos].strip()
    return text


def answer_message(
    model: GPT,
    tokenizer: Tokenizer,
    history: list[dict[str, str]],
    new_message: str,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: float = 0.08,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    device: str = config.DEVICE,
    use_multi_turn_format: bool = True,
) -> tuple[str, float]:
    """Generate assistant answer with context tracking and Rule 7 stop criteria."""
    token_ids = format_conversation_tokens(
        tokenizer=tokenizer,
        history=history,
        new_user_message=new_message,
        max_context_len=config.BLOCK_SIZE,
        use_multi_turn_format=use_multi_turn_format,
    )
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)

    t0 = time.perf_counter()
    with torch.no_grad():
        out_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            min_p=min_p,
            repetition_penalty=repetition_penalty,
            no_repeat_ngram_size=no_repeat_ngram_size,
            eos_token_id=EOS_ID,
        )
    elapsed = time.perf_counter() - t0

    generated_tokens = out_ids[0, len(token_ids):].tolist()

    # Rule 1: Truncate at terminal EOS token if present
    if EOS_ID in generated_tokens:
        generated_tokens = generated_tokens[: generated_tokens.index(EOS_ID)]

    raw_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)
    # Rule 7: Clean any stop strings
    clean_text = clean_response(raw_text)

    speed = len(generated_tokens) / max(elapsed, 1e-6)
    return clean_text, speed


def interactive_chat_session(
    model: GPT,
    tokenizer: Tokenizer,
    max_new_tokens: int = 256,
    temperature: float = 0.4,
    top_p: float = 0.90,
    min_p: float = 0.08,
    repetition_penalty: float = 1.15,
    no_repeat_ngram_size: int = 4,
    device: str = config.DEVICE,
    use_multi_turn_format: bool = True,
) -> None:
    """Run an interactive multi-turn chat REPL in the terminal."""
    print("=" * 72)
    print("  GPT-98.4M INTERACTIVE MULTI-TURN ASSISTANT CHAT")
    print("  Commands: '/clear' to reset dialogue memory | 'exit' or 'quit' to quit")
    print(f"  Factual Controls: temp={temperature}, top_p={top_p}, min_p={min_p}, rep_pen={repetition_penalty}")
    print("=" * 72 + "\n")

    history: list[dict[str, str]] = []

    while True:
        try:
            user_input = input("User > ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit", "q"]:
                print("\nExiting chat session.")
                break

            if user_input.lower() in ["/clear", "/reset"]:
                history.clear()
                print("[Context memory cleared. Starting fresh dialogue.]\n")
                continue

            print("\nAssistant: ", end="", flush=True)
            answer, speed = answer_message(
                model=model,
                tokenizer=tokenizer,
                history=history,
                new_message=user_input,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                min_p=min_p,
                repetition_penalty=repetition_penalty,
                no_repeat_ngram_size=no_repeat_ngram_size,
                device=device,
                use_multi_turn_format=use_multi_turn_format,
            )

            print(answer)
            print(f"[{speed:.1f} tokens/s | {len(history)//2 + 1} turns in memory]\n")

            # Update conversation history
            history.append({"role": "user", "content": user_input})
            history.append({"role": "assistant", "content": answer})

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break


def main():
    parser = argparse.ArgumentParser(description="Interactive Assistant Q&A Console (Phase 4 / Phase 8 SFT)")
    parser.add_argument("-p", "--prompt", type=str, default=None, help="One-shot prompt.")
    parser.add_argument("-c", "--checkpoint", type=str, default=None, help="Explicit checkpoint path.")
    parser.add_argument("-n", "--max-new-tokens", type=int, default=256, help="Max new tokens to generate.")
    parser.add_argument("-t", "--temperature", type=float, default=0.4, help="Sampling temperature.")
    parser.add_argument("--top-p", type=float, default=0.90, help="Top-P nucleus probability threshold.")
    parser.add_argument("--min-p", type=float, default=0.08, help="Min-P sampling relative threshold.")
    parser.add_argument("--rep-penalty", type=float, default=1.15, help="Repetition penalty.")
    parser.add_argument("--ngram-block", type=int, default=4, help="No-repeat n-gram size.")
    parser.add_argument("--device", type=str, default=config.DEVICE, help="Execution device (cuda/cpu).")
    parser.add_argument("--single-turn", action="store_true", help="Use legacy Phase 8 single-turn format.")
    args = parser.parse_args()

    ckpt_path = find_latest_checkpoint(args.checkpoint)
    model, tokenizer, saved_cfg = load_assistant_model(ckpt_path, device=args.device)

    is_multi_turn = not args.single_turn and (
        "multiturn" in saved_cfg.get("model_name", "").lower()
        or "multi_turn" in str(ckpt_path).lower()
        or "dpo" in str(ckpt_path).lower()
        or "dpo" in saved_cfg.get("model_name", "").lower()
    )
    if "multi_turn" in str(ckpt_path).lower() or "dpo" in str(ckpt_path).lower():
        is_multi_turn = True

    if args.prompt:
        answer, speed = answer_message(
            model=model,
            tokenizer=tokenizer,
            history=[],
            new_message=args.prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.rep_penalty,
            no_repeat_ngram_size=args.ngram_block,
            device=args.device,
            use_multi_turn_format=is_multi_turn,
        )
        print(f"\nUser: {args.prompt}")
        print(f"Assistant: {answer}")
        print(f"[{speed:.1f} tokens/s]\n")
    else:
        interactive_chat_session(
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            min_p=args.min_p,
            repetition_penalty=args.rep_penalty,
            no_repeat_ngram_size=args.ngram_block,
            device=args.device,
            use_multi_turn_format=is_multi_turn,
        )


if __name__ == "__main__":
    main()
