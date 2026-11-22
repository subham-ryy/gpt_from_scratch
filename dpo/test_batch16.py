"""Test batch size 16 (4 prompts x 4 samples) speed and VRAM."""
import sys
import time
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

def test_batch16():
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

    # 4 prompts, repeated 4 times = 16 sequences
    prompts = [
        "What is photosynthesis?",
        "How do I clean hard water stains from glass shower doors?",
        "My smartphone battery drains from 100% to 20% in just a few hours. What should I check?",
        "What are simple mobility exercises to relieve stiffness from sitting at a desk all day?",
    ]

    encoded = []
    for p in prompts:
        p_text = f"### User:\n{p}\n\n### Assistant:\n"
        encoded.append([BOS_ID] + tokenizer.encode(p_text).ids)

    # Left-pad to max prompt length in this micro-batch
    max_len = max(len(ids) for ids in encoded)
    batch_tensors = []
    for ids in encoded:
        pad_len = max_len - len(ids)
        padded = [0] * pad_len + ids
        # 4 samples each
        for _ in range(4):
            batch_tensors.append(padded)

    inp = torch.tensor(batch_tensors, dtype=torch.long, device=DEVICE)
    print(f"Batch shape: {inp.shape} (16 sequences in parallel)")

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()

    with torch.no_grad():
        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = model.generate(
                inp,
                max_new_tokens=200,
                temperature=0.8,
                top_p=0.95,
            )

    elapsed = time.perf_counter() - t0
    peak_vram = torch.cuda.max_memory_allocated() / 1e6
    tokens = 16 * 200
    print(f"16 sequences generated in: {elapsed:.2f}s ({tokens/elapsed:.1f} tokens/s)")
    print(f"Peak VRAM: {peak_vram:.0f} MB")
    print(f"Time per 4 prompts: {elapsed:.2f}s -> 2,000 prompts = {(elapsed / 4) * 2000 / 60:.1f} minutes")

if __name__ == "__main__":
    test_batch16()
