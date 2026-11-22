# Direct Preference Optimization (DPO) Final Report: GPT-98.4M

**Selected Winning Policy**: `dpo_lr1e6_b01`

**Status**: DPO_CANDIDATE_SELECTED

## Executive Summary
Following Supervised Fine-Tuning (SFT), the 98.4M parameter GPT model exhibited conversational capability but suffered from:
1. Residual canned refusals (*'As an AI language model...'*).
2. Repetition loops and wandering text on open-ended prompts.
3. Missing `<|eos|>` termination at context bounds.
4. Leaked math formatting tokens (`\boxed{}`, `'The answer is:'`) on general knowledge queries.

Through Direct Preference Optimization (DPO), we constructed an on-policy preference dataset of 3,239 pairs scored with transparent heuristic reward functions, precomputed reference log-probs to eliminate VRAM overhead, and performed a systematic hyperparameter sweep across learning rates, beta reward scaling, and NLL regularization.

## Hyperparameter Sweep Comparison

| Candidate Run | LR | Beta | NLL Alpha | Val Acc (%) | Val Loss | Val Margin | NLL Drift (%) | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `dpo_beta03` | - | - | - | - | - | - | +3.01% | **Regressed** |
| `dpo_nll01` | - | - | - | - | - | - | +6.62% | **Regressed** |
| `dpo_lr1e6_b01` | - | - | - | - | - | - | +0.76% | **Selected Winner** |
| `dpo_lr1e5_b01` | - | - | - | - | - | - | +8.42% | **Regressed** |
| `dpo_lr3e6_b01` | - | - | - | - | - | - | +3.71% | **Regressed** |

## Benchmark Evaluation (Default Decoding: `rep_pen=1.15`, `min_p=0.08`)

| Metric | SFT Baseline | Winner (dpo_lr1e6_b01) | Delta |
| :--- | :---: | :---: | :---: |
| **Refusal Rate** | 0.0% | 0.0% | +0.0% |
| **Clean EOS Stop Rate** | 71.0% | 81.0% | +10.0% |
| **3-Gram Repetition Rate** | 0.0153 | 0.0078 | -0.0075 |
| **Leaked Format Rate** | 0.0% | 0.0% | +0.0% |
| **Multi-Turn Retention** | 28.0% | 20.0% | -8.0% |
| **Trivia Accuracy** | 29.0% | 28.0% | -1.0% |
| **Mean Response Tokens** | 78.1 | 69.9 | - |

## Unconfounded Evaluation (Raw Softmax: `rep_pen=1.0`, `min_p=0.0`)

| Metric | SFT Baseline (Unconfounded) | Winner (Unconfounded) | Delta |
| :--- | :---: | :---: | :---: |
| **Refusal Rate** | 0.0% | 0.0% | +0.0% |
| **Clean EOS Stop Rate** | 72.0% | 82.0% | +10.0% |
| **3-Gram Repetition Rate** | 0.0476 | 0.0359 | -0.0117 |
| **Leaked Format Rate** | 0.0% | 0.0% | +0.0% |
| **Multi-Turn Retention** | 56.0% | 28.0% | -28.0% |
| **Trivia Accuracy** | 27.0% | 30.0% | +3.0% |

## Acceptance Checks Verification

| Requirement | Target | Achieved | Status |
| :--- | :--- | :--- | :---: |
| Phase 0: SFT backup checkpoint | Backup existing best.pt | `checkpoints/backup_sft_model4/best.pt` (1.18 GB) | PASS |
| Phase 0: IGNORE_INDEX verification | Use -100 or -1 | Verified `IGNORE_INDEX = -100` | PASS |
| Phase 1: Zero Contamination | 0% overlap with eval sets | Verified 0% overlap across 2,000 prompts | PASS |
| Phase 1: Multi-turn prompt share | ~30% multi-turn | Exactly 600 / 2,000 (30.0%) | PASS |
| Phase 2: On-Policy Sampling | 4 samples/prompt, 2,000 prompts | Exactly 8,000 on-policy samples in JSONL | PASS |
| Phase 3: Total Preference Pairs | ~3,000 pairs | 3,239 pairs | PASS |
| Phase 3: Synthetic Refusals | <= 10% | 6.2% (200 / 3,239) | PASS |
| Phase 4: Step 0 Invariants | Loss ~0.6931, Acc ~50%, Margin ~0 | Verified numerically | PASS |
| Phase 4: Reference VRAM | No second model in GPU memory | Precomputed & cached to disk | PASS |
| Phase 4: Sweep Configurations | 5 distinct hyperparameter runs | Completed 5 runs | PASS |
| Phase 5: Held-out Evaluation | Refusals, Repetition, EOS, Trivia, Format | Full automated test suite completed | PASS |
| Phase 6: Model Card & Report | Root MODEL_CARD.md + dpo_report | Generated and saved | PASS |