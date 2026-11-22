"""Unit tests for SFT Dataset prompt loss masking and sequence alignment."""
from pathlib import Path
import sys
import unittest

import torch
from tokenizers import Tokenizer

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.sft_dataset import SFTDataset, IGNORE_INDEX, PAD_ID, BOS_ID, EOS_ID


class TestSFTLossMasking(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
        cls.val_jsonl = config.DATA_DIR / "sft" / "sft_val.jsonl"
        cls.assertTrue(cls.val_jsonl.exists(), f"Val JSONL missing: {cls.val_jsonl}")

    def test_sft_masking_properties(self):
        dataset = SFTDataset(
            jsonl_path=self.val_jsonl,
            tokenizer=self.tokenizer,
            block_size=1024,
            max_samples=50,
        )

        for idx in range(min(len(dataset), 50)):
            x, y = dataset[idx]
            raw_item = dataset.raw_items[idx]
            sample_meta = dataset.samples[idx]

            P = len(sample_meta["prefix_ids"])
            R = len(sample_meta["answer_ids"])
            total_tokens = P + R

            # 1. Check shapes
            self.assertEqual(x.shape, (1024,))
            self.assertEqual(y.shape, (1024,))

            # 2. Check BOS token at start of input
            self.assertEqual(x[0].item(), BOS_ID)

            # 3. Check Prompt Masking:
            # Positions 0 to P-2 in y correspond to prompt tokens -> MUST be IGNORE_INDEX (-1)
            prompt_targets = y[: P - 1]
            self.assertTrue(
                (prompt_targets == IGNORE_INDEX).all().item(),
                f"Sample {idx}: Prompt targets not masked: {prompt_targets}",
            )

            # 4. Check Response Active Targets:
            # Positions P-1 to P+R-2 in y correspond to response tokens + EOS -> MUST be >= 0 and != -1
            response_targets = y[P - 1 : P + R - 1]
            self.assertEqual(
                len(response_targets),
                R,
                f"Sample {idx}: Expected {R} active response targets, got {len(response_targets)}",
            )
            self.assertTrue(
                (response_targets != IGNORE_INDEX).all().item(),
                f"Sample {idx}: Response targets contains IGNORE_INDEX: {response_targets}",
            )

            # 5. Check first active target matches first answer token
            expected_first_ans_tok = sample_meta["answer_ids"][0]
            self.assertEqual(
                y[P - 1].item(),
                expected_first_ans_tok,
                f"Sample {idx}: First target {y[P-1].item()} does not match first answer token {expected_first_ans_tok}",
            )

            # 6. Check last active target matches EOS_ID (2)
            self.assertEqual(
                y[P + R - 2].item(),
                EOS_ID,
                f"Sample {idx}: Last target {y[P+R-2].item()} is not EOS_ID ({EOS_ID})",
            )

            # 7. Check Padding Masking:
            # All positions from P+R-1 to 1023 in y MUST be IGNORE_INDEX (-1)
            # All positions from P+R-1 to 1023 in x MUST be PAD_ID (0)
            if P + R - 1 < 1024:
                pad_targets = y[P + R - 1 :]
                pad_inputs = x[P + R - 1 :]
                self.assertTrue(
                    (pad_targets == IGNORE_INDEX).all().item(),
                    f"Sample {idx}: Padding in y not masked with IGNORE_INDEX",
                )
                self.assertTrue(
                    (pad_inputs == PAD_ID).all().item(),
                    f"Sample {idx}: Padding in x not filled with PAD_ID",
                )

        print(f"\n[Test Passed] Successfully verified prompt loss masking across 50 samples!")


if __name__ == "__main__":
    unittest.main()

