"""Benchmark batched sampling speed for Phase 2."""
import time
import sys
from pathlib import Path

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

def benchmark():
    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    sft_ckpt = r"D:\pytorch\gpt_pretraining\experiments\20261121_174835_gpt_multi_turn_sft\checkpoints\best.pt"
    ckpt = torch.load(sft_ckpt, map_location=DEVICE, weights_only=False)
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

    test_prompts = [
        "What is photosynthesis?",
        "How do I clean hard water stains from glass shower doors?",
        "My smartphone battery drains from 100% to 20% in just a few hours. What should I check?",
        "What are simple mobility exercises to relieve stiffness from sitting at a desk all day?",
        "How do I properly season and maintain a cast-iron skillet?",
    ]

    print("Timing 5 prompts with 4 parallel samples each (20 samples total)...")
    t0 = time.perf_counter()
    total_tokens = 0

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            for p in test_prompts:
                p_text = f"### User:\n{p}\n\n### Assistant:\n"
                p_ids = [BOS_ID] + tokenizer.encode(p_text).ids
                inp = torch.tensor([p_ids] * 4, dtype=torch.long, device=DEVICE)

                out = model.generate(
                    inp,
                    max_new_tokens=200,
                    temperature=0.8,
                    top_p=0.95,
                    eos_token_id=EOS_ID,
                )
                total_tokens += (out.shape[1] - len(p_ids)) * 4

    elapsed = time.perf_counter() - t0
    tok_s = total_tokens / max(elapsed, 1e-6)
    print(f"Elapsed: {elapsed:.2f}s ({elapsed/5:.2f}s per prompt)")
    print(f"Total tokens generated: {total_tokens}")
    print(f"Throughput: {tok_s:.1f} tokens/s")
    est_total_sec = (elapsed / 5) * 2000
    print(f"Estimated time for 2,000 prompts: {est_total_sec:.1f}s ({est_total_sec/60:.1f} minutes)")

if __name__ == "__main__":
    benchmark()
