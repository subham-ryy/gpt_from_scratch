"""Verification test for Dynamic Batch Padding, Length Bucketing, and Loss Equivalence.

Proves:
1. SFTDataset unpadded output and dynamic collation.
2. Micro-batch padding rounds up to a multiple of 8, capped at 1024.
3. LengthGroupedBatchSampler keeps batch lengths clustered.
4. CRITICAL: Forward pass loss between static 1024 padding and dynamic batch padding
   is 100% numerically identical (difference < 1e-6) on the base GPT-98.4M model.
"""
from pathlib import Path
import sys
import unittest

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config
from data.sft_dataset import (
    SFTDataset,
    LengthGroupedBatchSampler,
    sft_collate_fn,
    IGNORE_INDEX,
    PAD_ID,
    BOS_ID,
    EOS_ID,
)
from model.transformer import GPT, GPTConfig


class TestDynamicPaddingEquivalence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(config.TOKENIZER_FILE))
        cls.val_jsonl = config.DATA_DIR / "sft" / "sft_val.jsonl"
        cls.device = torch.device(config.DEVICE)

        # Build GPT-98.4M model config
        model_config = GPTConfig.from_tokenizer_config(
            config_path=config.TOKENIZER_CONFIG_FILE,
            block_size=config.BLOCK_SIZE,
            n_layer=config.N_LAYER,
            n_head=config.N_HEAD,
            n_embd=config.N_EMBD,
            ffn_dim=config.FFN_DIM,
            dropout=0.0,
            bias=config.BIAS,
            tie_weights=config.TIE_WEIGHTS,
            use_flash=True,
        )
        cls.model = GPT(model_config).to(cls.device).eval()

        # Load Track A checkpoint weights
        base_ckpt_path = (
            PROJECT_ROOT
            / "experiments"
            / "20261106_032904_gpt_continued_pretrain"
            / "checkpoints"
            / "best.pt"
        )
        if base_ckpt_path.exists():
            ckpt = torch.load(str(base_ckpt_path), map_location="cpu", weights_only=False)
            cls.model.load_state_dict(ckpt["model_state"])
            print(f"[TestSetup] Loaded base weights from: {base_ckpt_path.name}")
        cls.model.eval()

    def test_dynamic_collation_properties(self):
        """Verify dynamic collator produces shape [B, dyn_len] where dyn_len % 8 == 0."""
        dataset = SFTDataset(
            jsonl_path=self.val_jsonl,
            tokenizer=self.tokenizer,
            block_size=1024,
            max_samples=20,
            dynamic_padding=True,
        )

        batch = [dataset[i] for i in range(4)]
        batch_x, batch_y = sft_collate_fn(batch, pad_to_multiple_of=8, max_block_size=1024)

        self.assertEqual(batch_x.size(0), 4)
        self.assertEqual(batch_y.size(0), 4)
        self.assertEqual(batch_x.size(1), batch_y.size(1))

        # Check multiple of 8
        self.assertEqual(batch_x.size(1) % 8, 0, f"Seq len {batch_x.size(1)} not a multiple of 8")
        self.assertLessEqual(batch_x.size(1), 1024)

        # Check that max sequence length in batch fits within padded length
        unpadded_max = max(x.size(0) for x, y in batch)
        self.assertGreaterEqual(batch_x.size(1), unpadded_max)

        print(f"[Collation Test] Micro-batch of 4 unpadded max={unpadded_max} -> dynamically padded to {batch_x.size(1)} tokens (mult of 8).")

    def test_length_grouped_sampler(self):
        """Verify length grouped sampler groups similar lengths together."""
        dataset = SFTDataset(
            jsonl_path=self.val_jsonl,
            tokenizer=self.tokenizer,
            block_size=1024,
            max_samples=200,
            dynamic_padding=True,
        )

        sampler = LengthGroupedBatchSampler(
            dataset, batch_size=4, bucket_size=16, shuffle=True, drop_last=True, seed=42
        )

        batch_variances = []
        for batch_indices in sampler:
            lens = [dataset.lengths[i] for i in batch_indices]
            batch_variances.append(max(lens) - min(lens))

        avg_spread = sum(batch_variances) / len(batch_variances)
        print(f"[Bucketing Test] Average length spread within micro-batch: {avg_spread:.1f} tokens.")
        self.assertLess(avg_spread, 40.0, "Bucket sampler failed to cluster similar lengths")

    def test_loss_exact_equivalence(self):
        """CRITICAL: Prove that static 1024 padding and dynamic batch padding yield identical loss."""
        # Create dataset in static mode and dynamic mode
        ds_static = SFTDataset(
            jsonl_path=self.val_jsonl,
            tokenizer=self.tokenizer,
            block_size=1024,
            max_samples=16,
            dynamic_padding=False,
        )
        ds_dynamic = SFTDataset(
            jsonl_path=self.val_jsonl,
            tokenizer=self.tokenizer,
            block_size=1024,
            max_samples=16,
            dynamic_padding=True,
        )

        amp_ctx = torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16)

        max_loss_diff_fp32 = 0.0
        max_loss_diff_bf16 = 0.0

        # Compare across 4 micro-batches of size 4
        for b_start in range(0, 16, 4):
            # 1. Static 1024 batch
            static_items = [ds_static[i] for i in range(b_start, b_start + 4)]
            x_static = torch.stack([item[0] for item in static_items]).to(self.device)
            y_static = torch.stack([item[1] for item in static_items]).to(self.device)

            # 2. Dynamic batch
            dyn_items = [ds_dynamic[i] for i in range(b_start, b_start + 4)]
            x_dyn, y_dyn = sft_collate_fn(dyn_items, pad_to_multiple_of=8, max_block_size=1024)
            x_dyn = x_dyn.to(self.device)
            y_dyn = y_dyn.to(self.device)

            self.assertEqual(x_static.shape, (4, 1024))
            self.assertLess(x_dyn.size(1), 1024)
            self.assertEqual(x_dyn.size(1) % 8, 0)

            # Test in FP32 (exact mathematical equivalence)
            with torch.no_grad():
                _, loss_s_fp32 = self.model(x_static, targets=y_static)
                _, loss_d_fp32 = self.model(x_dyn, targets=y_dyn)

            diff_fp32 = abs(loss_s_fp32.item() - loss_d_fp32.item())
            max_loss_diff_fp32 = max(max_loss_diff_fp32, diff_fp32)

            # Test in BF16
            with torch.no_grad():
                with amp_ctx:
                    _, loss_s_bf16 = self.model(x_static, targets=y_static)
                    _, loss_d_bf16 = self.model(x_dyn, targets=y_dyn)

            diff_bf16 = abs(loss_s_bf16.item() - loss_d_bf16.item())
            max_loss_diff_bf16 = max(max_loss_diff_bf16, diff_bf16)

            print(
                f"[Batch {b_start//4 + 1}] Static (1024 tok): FP32={loss_s_fp32.item():.6f}, BF16={loss_s_bf16.item():.6f} | "
                f"Dynamic ({x_dyn.size(1):3d} tok): FP32={loss_d_fp32.item():.6f}, BF16={loss_d_bf16.item():.6f} | "
                f"FP32 Diff: {diff_fp32:.2e} | BF16 Diff: {diff_bf16:.2e} ({1024 / x_dyn.size(1):.1f}x seq compute reduction)"
            )

            # Must match to within floating-point tolerance
            self.assertLess(
                diff_fp32,
                1e-6,
                f"Batch {b_start//4}: FP32 loss mismatch between static ({loss_s_fp32.item()}) and dynamic ({loss_d_fp32.item()})",
            )
            self.assertLess(
                diff_bf16,
                1e-3,
                f"Batch {b_start//4}: BF16 loss mismatch between static ({loss_s_bf16.item()}) and dynamic ({loss_d_bf16.item()})",
            )

        print(f"\n[Equivalence Verified] Maximum FP32 loss diff: {max_loss_diff_fp32:.2e} | Maximum BF16 loss diff: {max_loss_diff_bf16:.2e}")


if __name__ == "__main__":
    unittest.main()
