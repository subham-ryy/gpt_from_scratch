# Master Game Plan: From First SFT to a Production Micro-Assistant (GPT-98.4M)

**Hardware:** NVIDIA GeForce RTX 4050 Laptop GPU (6.0 GB VRAM, Windows 11)  
**Model Architecture:** 98,425,344 parameters, 12 layers, 768 hidden dim, 12 heads, 1,024 context, 16,384 BPE vocab (frozen `tokenizer.json`).  
**Current Baseline Status:**
- **Model 1 (Initial Base Pretrain):** Step 1,200 | Val Loss 5.2793 | PPL 196.23
- **Model 2 (Track A Continued Pretrain):** Step 9,000 | Val Loss 3.7055 | PPL 40.67 (289.5M valid tokens seen, ~113M unique tokens)
- **Model 3 (Phase 8 Single-Turn SFT):** Step 4,000 | Val Loss 2.6224 | PPL 13.77 (89% clean EOS termination)
- **Model 4 Base (Phase 3 500M Pretrain):** Step 15,250 | Val Loss 3.4810 | PPL 32.49 | BPB 1.1813 (490.6M valid tokens seen, 613M lifetime tokens)

---

## 0. Theoretical Foundation & Core Diagnostic

### 0.1 The 100k Document Under-Training Reality
- **Chinchilla Compute-Optimal Law ($N \approx 20 \times P$):** A 98.4M model requires $\approx \mathbf{1.97\text{ Billion tokens}}$ for compute-optimal pretraining. Modern efficient models (SmolLM, TinyLlama) train up to 500–3,000 tokens/param.
- **Model 2 Empirical Reality:** Trained on 100,000 FineWeb-Edu documents ($\approx 113\text{M}$ unique tokens), which is only **1.15 tokens per parameter**. Validation loss was still declining steeply at step 9,000.
- **Fundamental Law:** *Pretraining builds latent world knowledge, representations, and capabilities. SFT only teaches style, format, and elicitation.* You cannot SFT a model to answer what it has never seen in pretraining.

### 0.2 The "As an AI language model..." Refusal Leakage
- Alpaca was synthetically generated using OpenAI `text-davinci-003` with baked-in RLHF safety refusals.
- SFT is unconditional token imitation: the model simply mimics the statistical prompt-response pattern.
- **Solution:** Filter out refusal phrases directly from the training data before fine-tuning.

---

## 1. Master Strategy & Priority Roadmap

| Priority | Phase | Core Objective | Estimated Cost / Duration |
| :--- | :--- | :--- | :--- |
| **P1** | **Phase 1: SFT Data Scrubbing** | Filter canned refusals and identity leakage from existing SFT corpus. | ✅ **COMPLETED** (1,235 refusals removed, 67,560 kept) |
| **P2** | **Phase 2: 500M Pretraining Pipeline** | Download, stream, tokenize, and pack ~450k FineWeb-Edu docs into binary format. | ✅ **COMPLETED** (500,000,026 tokens packed, 98.26% efficiency, val_500m verified) |
| **P3** | **Phase 3: 500M Continued Pretrain** | Resume pretraining from Model 2 `best.pt` to push validation loss < 3.20. | ✅ **COMPLETED** (15,250 steps, 490.6M valid tokens, Val Loss 3.4810, PPL 32.49, BPB 1.1813) |
| **P4** | **Phase 4: Multi-Turn SFT** | Mix cleaned SFT data + SmolTalk dialogues with assistant-only loss masking. | ✅ **COMPLETED** (3,500 steps, 71.7m, Val Loss 2.0161, PPL 7.51, BPB 0.6844) |
| **P5** | **Phase 5: Rigorous $N \ge 20$ Eval** | Evaluate Model 4 vs Models 1–3 across 7 benchmarks; track Refusal Rate. | 🔄 **IN PROGRESS** (Comparative evaluation executing) |
| **P6** | **Phase 6: Targeted DPO** | Align model on conciseness, formatting, and hallucination reduction. | ~1.0 hour |

---

## 2. Phase 1: SFT Data Cleaning & Refusal Scrubbing

### 2.1 Refusal & Leakage Filters
Inspect `data/sft_train.jsonl` and remove responses containing:
- `"as an ai language model"`, `"as an ai"`, `"as a language model"`
- `"i'm sorry, but"`, `"i am sorry, but"`, `"i apologize, but"`
- `"i cannot assist"`, `"i cannot help"`, `"i do not have access"`, `"i don't have the ability"`
- Synthetic identity leakage: `"OpenAI"`, `"ChatGPT"`, `"GPT-3"`, `"GPT-4"`

### 2.2 Quality & Length Hygiene
- Remove empty responses or trivial one-word non-answers.
- Cap response length to avoid ultra-long outlier sequences dominating batch padding.
- Log exact counts: Number of raw examples vs. filtered examples.
- **Verification:** Print 30 random removed examples and 30 preserved examples to prevent removing legitimate answers.

---

## 3. Phase 2 & 3: Continued Pretraining to 500M Tokens

### 3.1 Data Preparation
- Stream high-quality educational documents from `HuggingFaceTB/smollm-corpus` (FineWeb-Edu extracts) or `HuggingFaceFW/fineweb-edu`.
- Target: **500,000,000 tokens** (approx 450,000–500,000 documents).
- Save as memory-mapped `uint16` binary tokens (`train_500m.bin`, ~1.0 GB).
- Carve out an independent, frozen 10,000-document validation set (`val_500m.bin`).
- Maintain document boundary loss masking to prevent cross-document attention bleed.

### 3.2 Calibrated Training Hyperparameters
*Do NOT restart at high learning rates ($3\times 10^{-4}$ will cause loss spikes on a converged model).*
- **Starting Checkpoint:** `experiments/20261106_032904_gpt_continued_pretrain/checkpoints/best.pt` (Val Loss 3.7055).
- **Sequence Length:** 1,024 tokens.
- **Batch Size:** 32,768 tokens per optimizer step (micro-batch 4 $\times$ gradient accumulation 8 $\times$ 1024 context).
- **Total Steps:** 15,250 steps ($\approx 500\text{M}$ tokens).
- **Learning Rate Schedule:**
  - 300-step linear warmup from $1.5\times 10^{-5} \to 1.0\times 10^{-4}$.
  - Cosine decay from $1.0\times 10^{-4} \to 1.0\times 10^{-5}$ at step 15,250.
- **Precision & Stability:** BF16 mixed precision + BF16 Z-loss ($\alpha = 1\times 10^{-4}$), gradient clipping = 1.0.
- **Checkpoints & Logging:** Save `last.pt` and `best.pt` only. Evaluate validation loss every 500 steps.

---

## 4. Phase 4: Multi-Turn SFT (The Micro-Assistant)

### 4.1 Critical Architecture Rule: Zero Tokenizer Alteration
- **Strict Rule:** Vocabulary remains fixed at 16,384 tokens in `tokenizer.json` to preserve 100% weight compatibility with pretraining checkpoints.
- **Format (Standard BPE Role Headers):**
  ```text
  ### User:
  {user_message}

  ### Assistant:
  {assistant_response}<|eos|>
  ```
- **Multi-Turn Format:**
  ```text
  ### User:
  {turn_1_user}

  ### Assistant:
  {turn_1_assistant}

  ### User:
  {turn_2_user}

  ### Assistant:
  {turn_2_assistant}<|eos|>
  ```

### 4.2 Dataset Mixture
1. **SmolTalk (`HuggingFaceTB/smoltalk`):** ~50% (Everyday conversations, multi-turn interactions, filtered for sequence length < 512).
2. **Cleaned Phase 1 SFT Corpus:** ~30% (SciQ factual explanations, Dolly, cleaned Alpaca).
3. **MetaMathQA (Elementary slice):** ~10% (Simple 1-2 step arithmetic; avoid long chains that confuse 98M models).
4. **Instruction / Code Formats:** ~10% (Basic markdown structuring and bulleted reasoning).

### 4.3 Training Mechanics
- **Assistant-Only Loss Masking:** Target label set to `IGNORE_INDEX = -100` for all `### User:\n...### Assistant:\n` headers and user turns. Cross-entropy computed **only on assistant responses across all turns**. Terminal `<|eos|>` is unmasked (target = 2) so the model learns when to terminate generation.
- **Piece-Wise Tokenization:** Every header, turn, and separator is tokenized separately before joining token IDs, preventing BPE boundary merge drift.
- **Dynamic Batch Padding:** Pad micro-batches only up to maximum sequence length in that micro-batch (rounded to multiple of 8, max 1024).
- **Length-Grouped Sampler:** Minimizes padding token waste.
- **LR & Schedule:** $8\times 10^{-5} \to 8\times 10^{-6}$, 100 warmup steps, cosine decay over 2 epochs (~3,500 steps).
- **Inference Stop Criteria:** Stops on `<|eos|>` and cleanly truncates if stop sequences (`### User:`, `\n### User:`) are generated.
- **Status:** ✅ Tested & Verified with 25-step smoke test (`train_multi_turn_sft.py`). Peak VRAM: 3.4GB / 6.0GB. Initial Val Loss 2.7950 $\to$ 2.5710.

---

## 5. Phase 5: Rigorous $N \ge 20$ Evaluation & Analysis

- Expand evaluation suites from $N=5 \to N=20$ prompts per category across Benchmarks 7–13:
  1. Mathematical Reasoning (Arithmetic & Word Problems)
  2. Python Code Generation (Syntax validity & Function execution)
  3. Text Summarization (Factual compression)
  4. Multi-Step Instruction Following
  5. Multi-Turn Conversational Memory & Pronoun Resolution
  6. Uncertainty Calibration & Hallucination Resistance
  7. Output Stability Across Temperatures
- **Dedicated Metric:** **Canned Refusal Rate (%)** on 20 personal assistance and everyday scenarios.
- **Model Progression Visualizations:** Generate publication-quality loss vs. tokens plots (5M $\to$ 113M $\to$ 500M tokens) and benchmark score comparison radar/bar charts.

---

## 6. Phase 6: Scoped DPO (Direct Preference Optimization)

- **Targets:** Eliminate remaining refusals, enforce conciseness, suppress repetitive loops, and penalize hallucinations.
- **Non-Targets:** Do not attempt to teach advanced math or code via DPO (a 98M model cannot sample correct references).
- **Dataset:** 3,000–5,000 high-quality preference pairs.
- **Settings:** $\beta = 0.05 - 0.1$, LR $2\times 10^{-6}$ with cosine decay, reference model frozen from Phase 4 best checkpoint.

---

## 7. Deliverables & Showcase

1. **Clean Codebase:** Fully documented training, inference, and interactive chat CLI (`chat.py`).
2. **Comprehensive Report & Visuals:** Scaling curve artifact showing continuous validation loss drops as data scaled 5x.
3. **Interactive Local Demo:** Fast, local CPU/GPU assistant running under 15ms per token on an RTX 4050.

