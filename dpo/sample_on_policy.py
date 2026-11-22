"""Phase 2: On-Policy Batched Sampling from Frozen SFT Checkpoint (Robust VRAM & Resume).

Settings:
- Checkpoint: experiments/20261121_174835_gpt_multi_turn_sft/checkpoints/best.pt
- 4 samples per prompt
- Temperature: 0.8, Top-P: 0.95, Max new tokens: 200
- Stop on <|eos|> or '### User:'
- Batch size: 2 prompts x 4 samples = 8 parallel sequences (VRAM ~1.4GB on 6GB RTX 4050)
- Automatic OOM fallback to 1 prompt at a time if memory spike occurs
- BF16 autocast, fixed seed (42)
- Output: data/dpo/dpo_raw_samples.jsonl
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer
import config
from model.transformer import GPT, GPTConfig

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BOS_ID = 1
EOS_ID = 2
PAD_ID = 0

STOP_MARKERS = ["### User:", "\n### User:", "### Assistant:", "\n###"]


def format_prompt_tokens(tokenizer: Tokenizer, prompt_messages: list[dict[str, str]]) -> list[int]:
    """Format single-turn or multi-turn prompt into exact token IDs."""
    user_hdr_ids = tokenizer.encode("### User:\n").ids
    asst_hdr_ids = tokenizer.encode("### Assistant:\n").ids
    sep_ids = tokenizer.encode("\n\n").ids

    input_ids = [BOS_ID]
    for msg in prompt_messages:
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            m_tokens = tokenizer.encode(f"{content}\n\n").ids
            input_ids.extend(user_hdr_ids + m_tokens)
        elif role == "assistant":
            m_tokens = tokenizer.encode(content).ids + [EOS_ID]
            input_ids.extend(asst_hdr_ids + m_tokens + sep_ids)

    # Append the final assistant header to elicit response
    input_ids.extend(asst_hdr_ids)
    return input_ids


def clean_response(text: str) -> str:
    cleaned = text.strip()
    for sm in STOP_MARKERS:
        pos = cleaned.find(sm)
        if pos != -1:
            cleaned = cleaned[:pos].strip()
    return cleaned


def generate_batch_samples(
    model: GPT,
    tokenizer: Tokenizer,
    batch_slice: list[dict],
    samples_per_prompt: int = 4,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
) -> tuple[list[dict], int]:
    """Run generation on a slice of prompts and return records and token count."""
    batch_prompt_tokens = [format_prompt_tokens(tokenizer, item["prompt_messages"]) for item in batch_slice]
    max_prompt_len = max(len(ids) for ids in batch_prompt_tokens)
    flat_input_tensors = []
    orig_lens = []

    for p_ids in batch_prompt_tokens:
        pad_amt = max_prompt_len - len(p_ids)
        padded_ids = [PAD_ID] * pad_amt + p_ids
        orig_lens.append(len(p_ids))
        for _ in range(samples_per_prompt):
            flat_input_tensors.append(padded_ids)

    inp = torch.tensor(flat_input_tensors, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                inp,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
            )

    flat_out = out.cpu().tolist()
    records = []
    seq_idx = 0
    total_tokens = 0

    for p_idx, item in enumerate(batch_slice):
        p_len = orig_lens[p_idx]
        samples = []

        for _ in range(samples_per_prompt):
            full_seq = flat_out[seq_idx]
            gen_slice = full_seq[max_prompt_len:]
            total_tokens += len(gen_slice)

            has_eos = EOS_ID in gen_slice
            if has_eos:
                eos_pos = gen_slice.index(EOS_ID)
                active_tokens = gen_slice[:eos_pos]
            else:
                active_tokens = gen_slice

            raw_text = tokenizer.decode(active_tokens, skip_special_tokens=True)
            cleaned_text = clean_response(raw_text)

            samples.append({
                "response_text": cleaned_text,
                "raw_tokens_len": len(active_tokens),
                "has_eos": has_eos,
            })
            seq_idx += 1

        records.append({
            "id": item["id"],
            "category": item.get("category", "general"),
            "is_multi_turn": item.get("is_multi_turn", False),
            "is_math": item.get("is_math", False),
            "prompt_messages": item["prompt_messages"],
            "samples": samples,
        })

    return records, total_tokens


def sample_on_policy(
    pool_path: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_prompt_pool.jsonl",
    output_path: Path = PROJECT_ROOT / "data" / "dpo" / "dpo_raw_samples.jsonl",
    checkpoint_path: Path = PROJECT_ROOT / "experiments" / "20261121_174835_gpt_multi_turn_sft" / "checkpoints" / "best.pt",
    batch_prompts: int = 2,
    samples_per_prompt: int = 4,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
):
    print("=" * 70, flush=True)
    print("  PHASE 2: ON-POLICY BATCHED SAMPLING (4 SAMPLES PER PROMPT)", flush=True)
    print("=" * 70, flush=True)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.cuda.empty_cache()

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    print(f"Loading frozen SFT checkpoint: {checkpoint_path.name}", flush=True)
    ckpt = torch.load(str(checkpoint_path), map_location=DEVICE, weights_only=False)
    saved_cfg = ckpt["config_snapshot"]

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
    del ckpt
    torch.cuda.empty_cache()

    # Load prompt pool
    with open(pool_path, "r", encoding="utf-8") as f:
        prompts_data = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(prompts_data)} prompts from pool.", flush=True)

    # Check already processed prompts for resume
    output_path.parent.mkdir(parents=True, exist_ok=True)
    done_ids = set()
    valid_lines = []
    if output_path.exists():
        with open(output_path, "r", encoding="utf-8") as f:
            for line in f:
                line_str = line.strip()
                if line_str:
                    try:
                        d = json.loads(line_str)
                        if "id" in d and len(d.get("samples", [])) == samples_per_prompt:
                            done_ids.add(d["id"])
                            valid_lines.append(line_str)
                    except Exception:
                        pass
        # Rewrite to clean any truncated trailing lines
        with open(output_path, "w", encoding="utf-8") as f:
            for v in valid_lines:
                f.write(v + "\n")
        print(f"Found {len(done_ids)} existing valid sampled prompts. Resuming...", flush=True)

    remaining = [p for p in prompts_data if p["id"] not in done_ids]
    print(f"Remaining prompts to process: {len(remaining)} (batch size: {batch_prompts})", flush=True)

    if not remaining:
        print("All prompts already sampled! Exiting Phase 2.", flush=True)
        return

    total_batches = (len(remaining) + batch_prompts - 1) // batch_prompts
    t0_start = time.perf_counter()
    total_tokens_generated = 0

    with open(output_path, "a", encoding="utf-8") as out_f:
        for b_idx in range(0, len(remaining), batch_prompts):
            batch_slice = remaining[b_idx : b_idx + batch_prompts]

            try:
                records, num_toks = generate_batch_samples(
                    model, tokenizer, batch_slice, samples_per_prompt, max_new_tokens, temperature, top_p
                )
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                print(f"\n[Warning] OOM/RuntimeError encountered for batch: {e}. Falling back to 1 prompt at a time...", flush=True)
                torch.cuda.empty_cache()
                records = []
                num_toks = 0
                for single_item in batch_slice:
                    s_records, s_toks = generate_batch_samples(
                        model, tokenizer, [single_item], samples_per_prompt, max_new_tokens, temperature, top_p
                    )
                    records.extend(s_records)
                    num_toks += s_toks
                    torch.cuda.empty_cache()

            total_tokens_generated += num_toks
            for r in records:
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
            out_f.flush()

            # Empty cache periodically
            cur_batch_num = b_idx // batch_prompts + 1
            if cur_batch_num % 10 == 0:
                torch.cuda.empty_cache()

            if cur_batch_num == 1 or cur_batch_num % 10 == 0 or cur_batch_num == total_batches:
                elapsed = time.perf_counter() - t0_start
                prompts_done = min(b_idx + batch_prompts, len(remaining))
                tok_s = total_tokens_generated / max(elapsed, 1e-6)
                pct = prompts_done / len(remaining) * 100
                rem_sec = (elapsed / max(prompts_done, 1)) * (len(remaining) - prompts_done)
                print(
                    f"Batch {cur_batch_num:4d}/{total_batches} | Prompts: {prompts_done:4d}/{len(remaining)} ({pct:5.1f}%) | "
                    f"Throughput: {tok_s:5.1f} tok/s | Elapsed: {elapsed/60:4.1f}m | ETA: {rem_sec/60:4.1f}m",
                    flush=True,
                )

    total_time = time.perf_counter() - t0_start
    print(f"\nPhase 2 Sampling Completed in {total_time:.1f}s ({total_time/60:.1f}m)!", flush=True)
    print(f"Total tokens generated: {total_tokens_generated:,}", flush=True)
    print(f"Raw samples saved to: {output_path}", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_prompts", type=int, default=2)
    args = parser.parse_args()
    sample_on_policy(batch_prompts=args.batch_prompts)
