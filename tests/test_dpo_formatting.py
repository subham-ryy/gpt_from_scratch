"""Phase 0 Verification Test: DPO Exact Loss Masking and Piece-Wise Tokenization."""
from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import torch
from tokenizers import Tokenizer
import config

BOS_ID = 1
EOS_ID = 2
IGNORE_INDEX = -100


def format_dpo_example(
    tokenizer: Tokenizer,
    prompt_messages: list[dict[str, str]],
    response_text: str,
    max_seq_len: int = 1024,
) -> tuple[torch.Tensor, torch.Tensor, int, int]:
    """Tokenize prompt and response piece-wise and apply exact loss masking.
    
    Returns:
        input_ids (Tensor [T]): Autoregressive input tokens x
        target_ids (Tensor [T]): Targets y with IGNORE_INDEX (-100) on prompt tokens
        prompt_len: Number of tokens belonging to prompt
        response_len: Number of response tokens (including EOS)
    """
    user_header_ids = tokenizer.encode("### User:\n").ids
    asst_header_ids = tokenizer.encode("### Assistant:\n").ids
    sep_ids = tokenizer.encode("\n\n").ids

    full_inputs = [BOS_ID]
    full_targets = [IGNORE_INDEX]

    # 1. Process prompt history
    for i, msg in enumerate(prompt_messages):
        role = msg["role"]
        content = msg["content"].strip()
        if role == "user":
            m_tokens = tokenizer.encode(f"{content}\n\n").ids
            turn_ids = user_header_ids + m_tokens
            full_inputs.extend(turn_ids)
            full_targets.extend([IGNORE_INDEX] * len(turn_ids))
        elif role == "assistant":
            m_tokens = tokenizer.encode(content).ids + [EOS_ID]
            turn_ids = asst_header_ids + m_tokens + sep_ids
            full_inputs.extend(turn_ids)
            # Prior assistant turns in prompt history are masked in DPO
            # because DPO evaluates the log-likelihood of the CURRENT generation only
            full_targets.extend([IGNORE_INDEX] * len(turn_ids))

    # Append the final assistant header
    full_inputs.extend(asst_header_ids)
    full_targets.extend([IGNORE_INDEX] * len(asst_header_ids))
    prompt_len = len(full_inputs)

    # 2. Append the current response and terminal EOS (UNMASKED)
    clean_resp = response_text.strip()
    resp_tokens = tokenizer.encode(clean_resp).ids + [EOS_ID]
    full_inputs.extend(resp_tokens)
    full_targets.extend(resp_tokens)
    response_len = len(resp_tokens)

    # Truncate if exceeds max_seq_len + 1
    if len(full_inputs) > max_seq_len + 1:
        # If prompt is too long, truncate prompt from left at turn boundary
        excess = len(full_inputs) - (max_seq_len + 1)
        full_inputs = [full_inputs[0]] + full_inputs[1 + excess :]
        full_targets = [full_targets[0]] + full_targets[1 + excess :]

    # Autoregressive shift: x predicts y
    x = torch.tensor(full_inputs[:-1], dtype=torch.long)
    y = torch.tensor(full_targets[1:], dtype=torch.long)

    return x, y, prompt_len, response_len


def test_dpo_masking():
    print("=" * 70)
    print("  PHASE 0: VERIFYING DPO LOSS MASKING & PIECE-WISE TOKENIZATION")
    print("=" * 70)

    tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
    assert IGNORE_INDEX == -100, f"Expected IGNORE_INDEX -100, got {IGNORE_INDEX}"

    examples = [
        # Single-turn
        (
            [{"role": "user", "content": "What is photosynthesis?"}],
            "Photosynthesis is the process by which plants convert sunlight into chemical energy.",
        ),
        # Multi-turn
        (
            [
                {"role": "user", "content": "My laptop is overheating when running intensive code."},
                {"role": "assistant", "content": "You should check the vents and thermal fans."},
                {"role": "user", "content": "What else can I check?"},
            ],
            "You can also check for runaway background processes and replace the thermal paste.",
        ),
        # Math prompt
        (
            [{"role": "user", "content": "What is 15 + 27?"}],
            "15 + 27 = 42.",
        ),
    ]

    for idx, (prompt_msgs, resp_text) in enumerate(examples, 1):
        x, y, prompt_len, resp_len = format_dpo_example(tokenizer, prompt_msgs, resp_text)

        # Check mask
        unmasked_mask = (y != IGNORE_INDEX)
        unmasked_targets = y[unmasked_mask].tolist()

        # Target must end with EOS_ID
        assert unmasked_targets[-1] == EOS_ID, f"Example {idx}: Expected target to end with EOS_ID (2), got {unmasked_targets[-1]}"

        # Decode unmasked targets (excluding final EOS)
        decoded_unmasked = tokenizer.decode(unmasked_targets[:-1], skip_special_tokens=True).strip()

        print(f"\n[Test Case {idx}]")
        print(f"  Prompt Turns      : {len(prompt_msgs)}")
        print(f"  Input Tokens (x)  : {len(x)}")
        print(f"  Unmasked Tokens   : {len(unmasked_targets)}")
        print(f"  Expected Response : {resp_text}")
        print(f"  Decoded Target    : {decoded_unmasked}")

        # Verification assertion: Decoded target must match input response text exactly!
        assert decoded_unmasked == resp_text.strip(), (
            f"Mismatch!\nExpected: '{resp_text.strip()}'\nGot:      '{decoded_unmasked}'"
        )

        # Assert no header leak in unmasked tokens
        assert "### User:" not in decoded_unmasked, "Leak: '### User:' found in unmasked targets!"
        assert "### Assistant:" not in decoded_unmasked, "Leak: '### Assistant:' found in unmasked targets!"

        print("  --> PASS: Exact match, no header leak, unmasked terminal EOS present.")

    print("\n" + "=" * 70)
    print("  PHASE 0 ACCEPTANCE CHECK: ALL MASK TESTS PASSED (100% EXACT)")
    print("=" * 70)


if __name__ == "__main__":
    test_dpo_masking()
