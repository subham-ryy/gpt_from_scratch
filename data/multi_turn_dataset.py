"""Multi-Turn SFT Dataset with Component-Wise Tokenization & Exact Loss Masking (Phase 4).

Guarantees:
1. Piece-wise tokenization: Components (headers, user turns, assistant turns, EOS) are tokenized
   separately and joined as ID lists to eliminate BPE merge shifts across prompt/response boundaries.
2. Exact Loss Masking:
   - User headers ("### User:\n") and messages are masked (target = -100).
   - Assistant headers ("### Assistant:\n") are masked (target = -100).
   - Assistant message tokens AND the terminal <|eos|> (id 2) are UNMASKED (target = token_id).
3. Dynamic Micro-Batch Padding:
   - Batches are padded only up to max_len in that micro-batch (rounded up to multiple of 8, max 1024).
4. Length-Grouped Batch Sampler:
   - Minimizes padding token waste by grouping similarly-sized conversations.
"""
from __future__ import annotations

import json
from pathlib import Path
import random
import sys
from typing import Iterator, List, Tuple

import numpy as np
import torch
from torch.utils.data import BatchSampler, Dataset, Sampler
from tokenizers import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

IGNORE_INDEX = -100  # PyTorch default ignore_index
PAD_ID = 0
BOS_ID = 1
EOS_ID = 2


class MultiTurnSFTDataset(Dataset):
    """Memory-efficient JSONL dataset for multi-turn conversations."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer: Tokenizer,
        max_seq_len: int = 1024,
    ) -> None:
        self.jsonl_path = Path(jsonl_path)
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len

        # Pre-tokenize headers and separators for fast lookup
        self.user_header_ids = self.tokenizer.encode("### User:\n").ids
        self.asst_header_ids = self.tokenizer.encode("### Assistant:\n").ids
        self.sep_ids = self.tokenizer.encode("\n\n").ids

        # Load raw dialogues and pre-compute token lengths for length-grouped batching
        self.examples = []
        self.lengths = []

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                self.examples.append(d)
                self.lengths.append(d.get("token_length", 256))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        ex = self.examples[idx]
        messages = ex["messages"]

        input_ids = [BOS_ID]
        target_ids = [IGNORE_INDEX]

        for turn in messages:
            role = turn["role"]
            content = turn["content"].strip()

            if role == "user":
                m_tokens = self.tokenizer.encode(f"{content}\n\n").ids
                turn_inputs = self.user_header_ids + m_tokens
                input_ids.extend(turn_inputs)
                target_ids.extend([IGNORE_INDEX] * len(turn_inputs))

            elif role == "assistant":
                # Mask header
                input_ids.extend(self.asst_header_ids)
                target_ids.extend([IGNORE_INDEX] * len(self.asst_header_ids))

                # Unmasked assistant tokens + unmasked terminal EOS
                m_tokens = self.tokenizer.encode(content).ids + [EOS_ID]
                input_ids.extend(m_tokens)
                target_ids.extend(m_tokens)  # Target is the tokens themselves!

                # Separator between turns (masked)
                input_ids.extend(self.sep_ids)
                target_ids.extend([IGNORE_INDEX] * len(self.sep_ids))

        # Truncate if exceeding max_seq_len + 1 (for autoregressive shift)
        if len(input_ids) > self.max_seq_len + 1:
            input_ids = input_ids[: self.max_seq_len + 1]
            target_ids = target_ids[: self.max_seq_len + 1]

        # Autoregressive shift: x predicts y (next token)
        x = torch.tensor(input_ids[:-1], dtype=torch.long)
        y = torch.tensor(target_ids[1:], dtype=torch.long)

        return x, y


def multi_turn_collate_fn(batch: List[Tuple[torch.Tensor, torch.Tensor]]) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dynamic micro-batch padding (rounded up to multiple of 8, max 1024)."""
    max_len = max(len(x) for x, _ in batch)
    padded_len = min(1024, ((max_len + 7) // 8) * 8)

    batch_x = []
    batch_y = []

    for x, y in batch:
        cur_len = len(x)
        if cur_len > padded_len:
            cur_x = x[:padded_len]
            cur_y = y[:padded_len]
        else:
            pad_amount = padded_len - cur_len
            cur_x = torch.cat([x, torch.full((pad_amount,), PAD_ID, dtype=torch.long)])
            cur_y = torch.cat([y, torch.full((pad_amount,), IGNORE_INDEX, dtype=torch.long)])

        batch_x.append(cur_x)
        batch_y.append(cur_y)

    return torch.stack(batch_x, dim=0), torch.stack(batch_y, dim=0)


class LengthGroupedBatchSampler(BatchSampler):
    """Batches similar-length conversations together to eliminate padding waste."""

    def __init__(
        self,
        dataset: MultiTurnSFTDataset,
        batch_size: int,
        shuffle: bool = True,
        bucket_size: int = 16,
    ) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.bucket_size = bucket_size
        self.indices = list(range(len(dataset)))

    def __iter__(self) -> Iterator[List[int]]:
        if self.shuffle:
            # Sort by length + slight noise
            sorted_indices = sorted(self.indices, key=lambda i: self.dataset.lengths[i] + random.uniform(-4, 4))
            # Chunk into megabatches and shuffle
            megabatch_size = self.batch_size * self.bucket_size
            megabatches = [
                sorted_indices[i : i + megabatch_size]
                for i in range(0, len(sorted_indices), megabatch_size)
            ]
            random.shuffle(megabatches)

            all_batches = []
            for mb in megabatches:
                batches = [mb[j : j + self.batch_size] for j in range(0, len(mb), self.batch_size)]
                random.shuffle(batches)
                all_batches.extend(batches)
            return iter(all_batches)
        else:
            return iter([self.indices[i : i + self.batch_size] for i in range(0, len(self.indices), self.batch_size)])

    def __len__(self) -> int:
        return (len(self.dataset) + self.batch_size - 1) // self.batch_size
