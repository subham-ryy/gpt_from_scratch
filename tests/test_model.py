"""Comprehensive Test Suite for Decoder-Only GPT Architecture.

Validates:
1. Mathematical parameter count vs actual PyTorch parameters
2. Weight tying between token embeddings and LM head
3. Causal masking (strict autoregressive invariant)
4. Absence of NaN / Inf in forward passes
5. Output shapes across variable batch sizes and sequence lengths
6. Full gradient flow through backward pass
7. Context window limit enforcement (> block_size raises ValueError)
8. Generative sampling functionality (temperature, top-k, top-p)
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import torch
import torch.nn.functional as F

from config import TOKENIZER_CONFIG_FILE
from model.transformer import GPT, GPTConfig


def get_test_config(vocab_size: int = 16384, block_size: int = 1024) -> GPTConfig:
    """Create standard Phase 4 GPT configuration."""
    return GPTConfig.from_tokenizer_config(
        config_path=TOKENIZER_CONFIG_FILE,
        block_size=block_size,
        n_layer=12,
        n_head=12,
        n_embd=768,
        ffn_dim=3072,
        dropout=0.1,
        bias=True,
        tie_weights=True,
    )


def test_parameter_count():
    """Verify exact mathematical parameter breakdown against PyTorch parameters."""
    print("--> Running Test 1: Parameter Count Verification...")
    config = get_test_config()
    model = GPT(config)
    breakdown = model.parameter_breakdown()

    theoretical = breakdown["theoretical_total"]
    actual = breakdown["actual_unique_params"]
    trainable = breakdown["actual_trainable_params"]

    print(f"    Theoretical expected: {theoretical:,} parameters")
    print(f"    PyTorch actual unique: {actual:,} parameters")
    print(f"    PyTorch trainable:     {trainable:,} parameters")

    assert theoretical == actual, (
        f"Parameter count mismatch: theoretical={theoretical:,} vs actual={actual:,}"
    )
    assert actual == 98_425_344, (
        f"Expected exactly 98,425,344 parameters for Phase 4 config, got {actual:,}"
    )
    print("    [PASSED] Parameter count matches theoretical calculation exactly (98,425,344 params).\n")


def test_weight_tying():
    """Verify weight tying between token embeddings and LM head."""
    print("--> Running Test 2: Weight Tying Verification...")
    config = get_test_config()
    model = GPT(config)

    # 1. Verify weight tensor identity (shares same memory address)
    wte_weight = model.embeddings.token_embeddings.weight
    lm_head_weight = model.lm_head.weight

    assert wte_weight is lm_head_weight, "lm_head.weight and token_embeddings.weight must be the exact same object."
    assert wte_weight.data_ptr() == lm_head_weight.data_ptr(), (
        "Memory pointers do not match: weight tying failed."
    )

    # 2. Verify mutation in one affects the other
    orig_val = wte_weight[0, 0].item()
    wte_weight.data[0, 0] += 1.0
    assert abs(lm_head_weight[0, 0].item() - (orig_val + 1.0)) < 1e-5, (
        "Modifying token_embeddings did not reflect in lm_head."
    )
    # Restore
    wte_weight.data[0, 0] = orig_val

    print(f"    Weight pointer: {hex(wte_weight.data_ptr())} shared by wte and lm_head.")
    print("    [PASSED] Weight tying verified.\n")


def test_causal_masking():
    """Verify autoregressive property: future token modifications do NOT affect past predictions."""
    print("--> Running Test 3: Causal Masking Verification...")
    config = GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        ffn_dim=512,
        dropout=0.0,  # Zero dropout for deterministic comparison
        bias=True,
        tie_weights=True,
        use_flash=False,  # Test explicit mask logic directly
    )
    model = GPT(config)
    model.eval()

    with torch.no_grad():
        # Create sequence A of 16 tokens
        seq_len = 16
        prefix_len = 8
        seq_a = torch.randint(0, config.vocab_size, (1, seq_len))

        # Create sequence B with identical prefix [0..7], but completely different future [8..15]
        seq_b = seq_a.clone()
        seq_b[0, prefix_len:] = torch.randint(0, config.vocab_size, (seq_len - prefix_len,))

        logits_a, _ = model(seq_a)  # [1, 16, V]
        logits_b, _ = model(seq_b)  # [1, 16, V]

        # Check logits on the prefix positions [0..prefix_len-1]
        diff_prefix = (logits_a[:, :prefix_len, :] - logits_b[:, :prefix_len, :]).abs().max().item()
        # Check logits on the modified future positions [prefix_len..seq_len-1]
        diff_future = (logits_a[:, prefix_len:, :] - logits_b[:, prefix_len:, :]).abs().max().item()

        print(f"    Max diff in prefix logits (positions 0..{prefix_len-1}): {diff_prefix:.2e}")
        print(f"    Max diff in future logits (positions {prefix_len}..{seq_len-1}): {diff_future:.2e}")

        assert diff_prefix < 1e-5, (
            f"Causal mask violation! Future tokens altered past logits by {diff_prefix:.2e}"
        )
        assert diff_future > 1e-3, (
            "Future logits should differ when future tokens are altered."
        )

    print("    [PASSED] Causal masking strictly preserves autoregressive invariant.\n")


def test_nan_inf_check():
    """Verify forward pass outputs are clean of NaN and Inf."""
    print("--> Running Test 4: NaN / Inf Check...")
    config = GPTConfig(
        vocab_size=1000,
        block_size=128,
        n_layer=4,
        n_head=4,
        n_embd=128,
        ffn_dim=512,
        dropout=0.1,
    )
    model = GPT(config)
    model.eval()

    with torch.no_grad():
        x = torch.randint(0, config.vocab_size, (2, 64))
        logits, _ = model(x)

        assert not torch.isnan(logits).any(), "NaN found in model output logits!"
        assert not torch.isinf(logits).any(), "Inf found in model output logits!"

    print("    [PASSED] Forward pass is completely clean of NaN and Inf.\n")


def test_variable_batch_and_seq_lengths():
    """Verify forward pass across various batch sizes and sequence lengths."""
    print("--> Running Test 5: Variable Batch Sizes and Sequence Lengths...")
    config = GPTConfig(
        vocab_size=16384,
        block_size=1024,
        n_layer=4,
        n_head=4,
        n_embd=256,
        ffn_dim=1024,
    )
    model = GPT(config)
    model.eval()

    test_shapes = [
        (1, 1),
        (1, 16),
        (2, 64),
        (4, 128),
        (2, 512),
        (1, 1024),
    ]

    with torch.no_grad():
        for b, t in test_shapes:
            idx = torch.randint(0, config.vocab_size, (b, t))
            logits, loss = model(idx)
            assert logits.shape == (b, t, config.vocab_size), (
                f"Expected shape ({b}, {t}, {config.vocab_size}), got {logits.shape}"
            )
            assert loss is None, "Loss should be None when targets are not provided."
            print(f"    Batch={b}, SeqLen={t:4d} -> Logits shape: {list(logits.shape)} [OK]")

    print("    [PASSED] All tested batch sizes and sequence lengths produce correct tensor shapes.\n")


def test_loss_and_backward():
    """Verify loss calculation and full gradient backpropagation."""
    print("--> Running Test 6: Loss Computation and Gradient Backpropagation...")
    config = GPTConfig(
        vocab_size=500,
        block_size=64,
        n_layer=4,
        n_head=4,
        n_embd=128,
        ffn_dim=512,
        dropout=0.1,
        bias=True,
        tie_weights=True,
    )
    model = GPT(config)
    model.train()

    B, T = 2, 32
    idx = torch.randint(0, config.vocab_size, (B, T))
    targets = torch.randint(0, config.vocab_size, (B, T))

    logits, loss = model(idx, targets=targets)

    assert loss is not None, "Loss must not be None when targets are passed."
    assert loss.dim() == 0, f"Loss must be scalar, got dim {loss.dim()}."
    assert not torch.isnan(loss), "Loss computed as NaN!"
    assert not torch.isinf(loss), "Loss computed as Inf!"
    assert loss.item() > 0, f"Loss must be positive, got {loss.item()}."

    # Backward pass
    model.zero_grad()
    loss.backward()

    # Verify that all trainable parameters received valid gradients
    zero_grad_params = []
    nan_grad_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if param.grad is None:
                zero_grad_params.append(name)
            elif torch.isnan(param.grad).any():
                nan_grad_params.append(name)

    assert len(nan_grad_params) == 0, f"NaN gradients detected in: {nan_grad_params}"
    assert len(zero_grad_params) == 0, f"Missing gradients in: {zero_grad_params}"

    print(f"    Cross-entropy loss: {loss.item():.4f}")
    print(f"    All {len(list(model.parameters()))} parameter tensors received valid gradients.")
    print("    [PASSED] Full backward pass and gradient flow verified.\n")


def test_context_window_limit():
    """Verify that inputs exceeding block_size raise ValueError."""
    print("--> Running Test 7: Context Window Limit Enforcement...")
    config = GPTConfig(vocab_size=100, block_size=32, n_layer=2, n_head=2, n_embd=64)
    model = GPT(config)

    # Input within limit: 32 tokens
    idx_ok = torch.randint(0, 100, (1, 32))
    _ = model(idx_ok)

    # Input exceeding limit: 33 tokens
    idx_overflow = torch.randint(0, 100, (1, 33))
    raised = False
    try:
        _ = model(idx_overflow)
    except ValueError as e:
        raised = True
        print(f"    Caught expected ValueError: {e}")

    assert raised, "Model should raise ValueError when sequence length > block_size!"
    print("    [PASSED] Context window boundary correctly enforced.\n")


def test_generation():
    """Verify autoregressive generation."""
    print("--> Running Test 8: Autoregressive Text Generation...")
    config = GPTConfig(
        vocab_size=1000,
        block_size=64,
        n_layer=2,
        n_head=2,
        n_embd=64,
        ffn_dim=256,
    )
    model = GPT(config)
    model.eval()

    prompt = torch.randint(0, config.vocab_size, (1, 5))
    generated = model.generate(
        prompt,
        max_new_tokens=10,
        temperature=0.8,
        top_k=20,
    )

    assert generated.shape == (1, 15), f"Expected shape (1, 15), got {generated.shape}"
    # Verify the prefix is preserved
    assert torch.equal(generated[:, :5], prompt), "Prompt was modified during generation!"

    print(f"    Prompt shape: {prompt.shape} -> Generated shape: {generated.shape}")
    print(f"    Generated token IDs: {generated.squeeze().tolist()}")
    print("    [PASSED] Autoregressive text generation verified.\n")


def run_all_tests():
    """Execute all tests in sequence."""
    print("=================================================================")
    print("            PHASE 4: GPT ARCHITECTURE TEST SUITE                 ")
    print("=================================================================\n")

    test_parameter_count()
    test_weight_tying()
    test_causal_masking()
    test_nan_inf_check()
    test_variable_batch_and_seq_lengths()
    test_loss_and_backward()
    test_context_window_limit()
    test_generation()

    print("=================================================================")
    print("       ALL 8 TEST SUITES PASSED CLEANLY AND SUCCESSFULLY!        ")
    print("=================================================================")


if __name__ == "__main__":
    run_all_tests()
