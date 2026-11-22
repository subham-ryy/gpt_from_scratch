"""Unit tests for Greedy Document Bin-Packing (Phase B).

Validates:
1. Short document packing (multiple docs in one window).
2. Long document chunking (> block_size).
3. Mixed document packing.
4. Document boundary transition masks (target at EOS is IGNORE_INDEX=-1).
5. Padding position IDs (input=PAD_ID=0, target=IGNORE_INDEX=-1).
6. Packing efficiency exceeds the >92% requirement.
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.dataset import PackedBinaryTokenDataset, IGNORE_INDEX, PAD_ID, BOS_ID, EOS_ID


def create_synthetic_bin(doc_lengths: list[int]) -> tuple[Path, int]:
    """Helper to create a temporary uint16 binary shard with given document lengths."""
    tokens = []
    token_val = 100
    for length in doc_lengths:
        # Each document: [BOS=1] + tokens + [EOS=2]
        doc = [BOS_ID]
        for _ in range(length - 2):
            doc.append(token_val)
            token_val = (token_val + 1) % 15000 + 10  # stay above special IDs
        doc.append(EOS_ID)
        tokens.extend(doc)

    temp_file = tempfile.NamedTemporaryFile(suffix=".bin", delete=False)
    temp_path = Path(temp_file.name)
    temp_file.close()

    arr = np.array(tokens, dtype=np.uint16)
    arr.tofile(temp_path)
    return temp_path, len(tokens)


class TestPackedBinaryTokenDataset(unittest.TestCase):

    def test_short_doc_packing(self):
        """Test that multiple short documents pack into a single 1024 sequence."""
        doc_lens = [200, 300, 400]  # Sum = 900 <= 1024
        bin_path, _ = create_synthetic_bin(doc_lens)
        ds = None
        try:
            ds = PackedBinaryTokenDataset(bin_path, block_size=1024, strategy="first_fit")
            self.assertEqual(len(ds), 1, f"Expected 1 packed sample, got {len(ds)}")

            x, y = ds[0]
            self.assertEqual(x.shape, (1024,))
            self.assertEqual(y.shape, (1024,))

            # First doc at 0..200
            self.assertEqual(x[0].item(), BOS_ID)
            self.assertEqual(x[199].item(), EOS_ID)
            self.assertEqual(y[199].item(), IGNORE_INDEX)  # EOS must not predict next token!

            # Second doc at 200..500
            self.assertEqual(x[200].item(), BOS_ID)
            self.assertEqual(x[499].item(), EOS_ID)
            self.assertEqual(y[499].item(), IGNORE_INDEX)

            # Third doc at 500..900
            self.assertEqual(x[500].item(), BOS_ID)
            self.assertEqual(x[899].item(), EOS_ID)
            self.assertEqual(y[899].item(), IGNORE_INDEX)

            # Tail padding 900..1024
            self.assertTrue((x[900:] == PAD_ID).all())
            self.assertTrue((y[900:] == IGNORE_INDEX).all())
        finally:
            if ds is not None:
                ds.close()
            bin_path.unlink(missing_ok=True)

    def test_long_doc_chunking(self):
        """Test that a document > block_size is split into non-overlapping chunks."""
        doc_lens = [2500]  # 1024 + 1024 + 452
        bin_path, _ = create_synthetic_bin(doc_lens)
        ds = None
        try:
            ds = PackedBinaryTokenDataset(bin_path, block_size=1024, strategy="first_fit")
            self.assertEqual(len(ds), 3, f"Expected 3 samples (2 full + 1 tail), got {len(ds)}")

            # First chunk (full 1024, middle of doc, no EOS)
            x0, y0 = ds[0]
            self.assertEqual(x0[0].item(), BOS_ID)
            self.assertNotEqual(x0[-1].item(), EOS_ID)
            # y0 should have valid targets all the way to index 1023 (no IGNORE_INDEX)
            self.assertTrue((y0 != IGNORE_INDEX).all())

            # Second chunk (full 1024, middle of doc, no EOS)
            x1, y1 = ds[1]
            self.assertTrue((y1 != IGNORE_INDEX).all())

            # Third chunk (tail 452 tokens, ends with EOS, rest padded)
            x2, y2 = ds[2]
            self.assertEqual(x2[451].item(), EOS_ID)
            self.assertEqual(y2[451].item(), IGNORE_INDEX)
            self.assertTrue((x2[452:] == PAD_ID).all())
            self.assertTrue((y2[452:] == IGNORE_INDEX).all())
        finally:
            if ds is not None:
                ds.close()
            bin_path.unlink(missing_ok=True)

    def test_document_boundary_transition_mask(self):
        """Verify that EVERY position where input is EOS has target == IGNORE_INDEX."""
        doc_lens = [150, 250, 350, 600, 800, 1500]
        bin_path, _ = create_synthetic_bin(doc_lens)
        ds = None
        try:
            ds = PackedBinaryTokenDataset(bin_path, block_size=1024, strategy="first_fit")
            for i in range(len(ds)):
                x, y = ds[i]
                eos_indices = (x == EOS_ID).nonzero(as_tuple=True)[0]
                for eos_idx in eos_indices:
                    self.assertEqual(
                        y[eos_idx].item(),
                        IGNORE_INDEX,
                        f"Sample {i}, index {eos_idx}: Input is EOS but target is {y[eos_idx].item()}",
                    )
        finally:
            if ds is not None:
                ds.close()
            bin_path.unlink(missing_ok=True)

    def test_no_negative_input_indices(self):
        """Ensure input x contains only valid non-negative token IDs (PAD_ID=0)."""
        doc_lens = [100, 50, 80]
        bin_path, _ = create_synthetic_bin(doc_lens)
        ds = None
        try:
            ds = PackedBinaryTokenDataset(bin_path, block_size=1024)
            for i in range(len(ds)):
                x, y = ds[i]
                self.assertTrue((x >= 0).all(), f"Negative token ID found in input x: {x[x < 0]}")
        finally:
            if ds is not None:
                ds.close()
            bin_path.unlink(missing_ok=True)

    def test_packing_efficiency_on_expanded_dataset(self):
        """Verify that packing efficiency on the expanded dataset exceeds 92%."""
        bin_path = Path("d:/pytorch/gpt_pretraining/data/tokenized/train_expanded.bin")
        if not bin_path.exists():
            print("train_expanded.bin not found, skipping dataset efficiency test.")
            return

        ds = PackedBinaryTokenDataset(bin_path, block_size=1024, strategy="first_fit")
        try:
            print(f"\n[Test] Measured packing efficiency: {ds.packing_efficiency:.2f}%")
            self.assertGreater(
                ds.packing_efficiency,
                92.0,
                f"Packing efficiency {ds.packing_efficiency:.2f}% is below 92% requirement!",
            )
        finally:
            ds.close()


if __name__ == "__main__":
    unittest.main()
