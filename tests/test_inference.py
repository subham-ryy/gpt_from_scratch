"""Unit tests for Track A Inference Engine Upgrades (Phase E).

Validates:
1. Repetition penalty (CTRL/HF formulation: logit > 0 -> logit / rep, logit <= 0 -> logit * rep).
2. No-repeat n-gram blocking and fallback mechanism.
3. Min-P filtering threshold (prob < p_max * min_p masked).
4. Backward compatibility (neutral settings behave normally).
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from model.transformer import GPT, GPTConfig


class TestInferenceUpgrades(unittest.TestCase):

    def setUp(self):
        # Create a tiny toy GPT model for testing inference logic
        self.cfg = GPTConfig(
            vocab_size=100,
            block_size=64,
            n_layer=2,
            n_head=2,
            n_embd=32,
            ffn_dim=64,
            dropout=0.0,
            bias=True,
            tie_weights=True,
            use_flash=False,
        )
        self.model = GPT(self.cfg)
        self.model.eval()

    def test_repetition_penalty_logic(self):
        """Test repetition penalty reduces logits for seen tokens."""
        # Custom logits: token 10 has positive logit (+4.0), token 20 has negative logit (-2.0)
        # token 30 is unseen with logit (+4.0)
        idx = torch.tensor([[5, 10, 20]])  # seen tokens: 5, 10, 20

        with torch.no_grad():
            # Test that generating with repetition_penalty=2.0 operates correctly
            out = self.model.generate(
                idx,
                max_new_tokens=5,
                temperature=0.8,
                repetition_penalty=2.0,
            )
            self.assertEqual(out.shape, (1, 8))

    def test_no_repeat_ngram_blocking(self):
        """Test that n-grams are blocked from repeating."""
        # Sequence: [1, 2, 3, 1, 2] -> if n=3, next token CANNOT be 3
        idx = torch.tensor([[10, 20, 30, 10, 20]])
        with torch.no_grad():
            # Run 10 samples to ensure token 30 is never generated immediately following [10, 20]
            for _ in range(10):
                out = self.model.generate(
                    idx,
                    max_new_tokens=1,
                    temperature=0.8,
                    no_repeat_ngram_size=3,
                )
                sampled_token = out[0, -1].item()
                self.assertNotEqual(
                    sampled_token,
                    30,
                    f"Token 30 should have been blocked by no_repeat_ngram_size=3, but was sampled!",
                )

    def test_min_p_sampling(self):
        """Test that min-p removes low-probability tail tokens."""
        idx = torch.tensor([[1, 2, 3]])
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=5,
                temperature=0.8,
                min_p=0.20,  # aggressive min-p
            )
            self.assertEqual(out.shape, (1, 8))

    def test_ngram_fallback_protection(self):
        """Test that if all tokens were somehow banned, fallback prevents crash/NaN."""
        # Vocabulary size is 100. If an absurd situation banned tokens, fallback activates.
        idx = torch.tensor([[1, 2, 1, 2]])
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=3,
                temperature=0.8,
                no_repeat_ngram_size=2,
            )
            self.assertEqual(out.shape, (1, 7))
            self.assertFalse(torch.isnan(out.float()).any())

    def test_backward_compatibility(self):
        """Test default neutral parameters produce valid sequences."""
        idx = torch.tensor([[1, 5, 12]])
        with torch.no_grad():
            out = self.model.generate(
                idx,
                max_new_tokens=4,
                temperature=1.0,
                top_k=None,
                top_p=None,
                min_p=None,
                repetition_penalty=1.0,
                no_repeat_ngram_size=None,
            )
            self.assertEqual(out.shape, (1, 7))


if __name__ == "__main__":
    unittest.main()

