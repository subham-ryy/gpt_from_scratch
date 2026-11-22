"""Token and Positional Embeddings for Decoder-Only GPT.

Combines learned token embeddings and learned positional embeddings,
followed by dropout regularization.
"""
from typing import Optional
import torch
import torch.nn as nn


class GPTEmbeddings(nn.Module):
    """Combines learned token embeddings with learned 1D positional embeddings.

    In a decoder-only Transformer, each token at index t is represented by:
        Embedding(t) = TokenEmbedding(token_id) + PositionEmbedding(t)

    This provides both semantic representation (what token is it?) and
    spatial representation (where in the sequence does it appear?).

    Args:
        vocab_size: Total number of tokens in the vocabulary (V).
        block_size: Maximum context length / sequence length supported (T_max).
        n_embd: Dimensionality of the embedding vector (d_model).
        dropout: Dropout probability applied to the summed embeddings.
    """

    def __init__(
        self,
        vocab_size: int,
        block_size: int,
        n_embd: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.n_embd = n_embd

        # Token embedding lookup table: [V, d_model]
        # Maps integer token IDs to continuous embedding vectors.
        self.token_embeddings = nn.Embedding(vocab_size, n_embd)

        # Learned positional embedding lookup table: [T_max, d_model]
        # Maps integer sequence positions [0, 1, ..., T-1] to position vectors.
        self.position_embeddings = nn.Embedding(block_size, n_embd)

        # Embedding dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, idx: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass for token and positional embeddings.

        Args:
            idx: Tensor of token indices with shape [batch_size, seq_len] (or [B, T]).
            position_ids: Optional custom position indices [seq_len] or [B, T].
                          If None, defaults to [0, 1, ..., seq_len - 1].

        Returns:
            Tensor of combined embeddings with shape [batch_size, seq_len, n_embd].

        Raises:
            ValueError: If input sequence length exceeds block_size context window.
        """
        device = idx.device
        batch_size, seq_len = idx.size()

        # Enforce context length boundary condition
        if seq_len > self.block_size:
            raise ValueError(
                f"Input sequence length {seq_len} exceeds maximum context window (block_size={self.block_size})."
            )

        # 1. Look up token embeddings: [B, T] -> [B, T, n_embd]
        tok_emb = self.token_embeddings(idx)

        # 2. Look up position embeddings:
        if position_ids is None:
            # Generate range [0, 1, ..., seq_len - 1] on the same device as the input
            position_ids = torch.arange(0, seq_len, dtype=torch.long, device=device)  # [T]
            # [T] -> [T, n_embd] -> broadcasted to [B, T, n_embd] upon addition
            pos_emb = self.position_embeddings(position_ids)
        else:
            # If explicit position_ids supplied: [B, T] -> [B, T, n_embd]
            pos_emb = self.position_embeddings(position_ids)

        # 3. Element-wise addition and dropout:
        # [B, T, n_embd] + [T, n_embd] -> [B, T, n_embd]
        x = self.dropout(tok_emb + pos_emb)

        return x
