# Model Card: GPT-98.4M-Assistant-DPO

## Model Overview
- **Architecture**: 12-layer, 768-dim, 12-head decoder-only Transformer
- **Parameter Count**: 98,425,344 parameters (tied embedding weights)
- **Context Length (block_size)**: 1,024 tokens
- **Tokenizer**: Custom byte-level BPE, vocabulary size 16,384 (`tokenizer/tokenizer.json`)
- **Format**: Plaintext conversational formatting with assistant-only loss masking:
  `### User:\n{prompt}\n\n### Assistant:\n{reply}<|eos|>`
- **Active Model**: `dpo_lr1e6_b01`

## Training Progression
1. **Pretraining (Phase 1-3)**: Base causal language modeling on FineWeb-Edu (~15,000 steps, val loss 3.48).
2. **Continued Pretraining**: Refined base fluency (~500M tokens).
3. **Multi-Turn SFT (Phase 4)**: 3,500 steps on multi-turn dialogue with refusal filtering and assistant loss masking (val loss 2.0161).
4. **Direct Preference Optimization (DPO, Phase 5-6)**: On-policy preference alignment on 3,239 preference pairs (dpo_lr1e6_b01).

## Empirical Performance & Benchmark Results

| Evaluation Metric | SFT Baseline | DPO Final | Delta |
| :--- | :---: | :---: | :---: |
| **Refusal Rate** | 0.0% | 0.0% | +0.0% |
| **3-Gram Repetition Rate** | 0.0153 | 0.0078 | -0.0075 |
| **Clean EOS Stop Rate** | 71.0% | 81.0% | +10.0% |
| **Leaked Format Rate** (`\boxed{}`) | 0.0% | 0.0% | +0.0% |
| **Multi-Turn Context Retention** | 28.0% | 20.0% | -8.0% |
| **Trivia Accuracy** | 29.0% | 28.0% | -1.0% |
| **Mean Response Tokens** | 78.1 | 69.9 | - |

## Recommended Sampling Configuration
```python
# Recommended interactive inference settings
temperature = 0.4
top_p = 0.90
min_p = 0.08
repetition_penalty = 1.15
no_repeat_ngram_size = 4
max_new_tokens = 200
```

## Intended Use & Limitations
- **Intended Use**: Casual conversational assistant, multi-turn dialogue, general creative and structured text generation.
- **Known Limitations**: At 98.4M parameters, world knowledge is limited. The model will produce factual errors on obscure trivia or arithmetic. It is NOT intended as a source of verified truth or safety-critical computation.
- **Running Interactive Chat**:
  ```powershell
  cd D:\pytorch\gpt_pretraining
  ..\.venv\Scripts\python.exe chat.py
  ```