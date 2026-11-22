"""Feed-Forward Network (MLP) for Decoder-Only GPT.

Implements the two-layer position-wise feed-forward block with GELU activation
and residual dropout.
"""
from typing import Optional
import torch
import torch.nn as nn


class MLP(nn.Module):
    """Position-wise Feed-Forward Network (FFN).

    Applies two linear transformations with a non-linear GELU activation
    and dropout in between:
        FFN(x) = Dropout(Linear_2(GELU(Linear_1(x))))

    The intermediate dimension is typically expanded by a factor of 4
    (e.g., 768 -> 3,072 -> 768), allowing the model to project token
    representations into a higher-dimensional space where non-linear
    relationships and factual patterns can be captured.

    Args:
        n_embd: Input and output embedding dimension (d_model).
        ffn_dim: Intermediate expansion dimension. Defaults to 4 * n_embd.
        dropout: Dropout probability applied to the output projection.
        bias: Whether linear layers include learnable bias parameters.
        approximate_gelu: Whether to use tanh approximation for GELU (GPT-2 style).
    """

    def __init__(
        self,
        n_embd: int = 768,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        bias: bool = True,
        approximate_gelu: bool = True,
    ) -> None:
        super().__init__()
        if ffn_dim is None:
            ffn_dim = 4 * n_embd

        self.n_embd = n_embd
        self.ffn_dim = ffn_dim

        # Up-projection: [n_embd] -> [ffn_dim] (e.g. 768 -> 3072)
        self.c_fc = nn.Linear(n_embd, ffn_dim, bias=bias)

        # Non-linear activation: Gaussian Error Linear Unit (GELU)
        # Using approximate="tanh" reproduces the standard GPT-2 formulation.
        if approximate_gelu:
            self.act = nn.GELU(approximate="tanh")
        else:
            self.act = nn.GELU()

        # Down-projection: [ffn_dim] -> [n_embd] (e.g. 3072 -> 768)
        self.c_proj = nn.Linear(ffn_dim, n_embd, bias=bias)

        # Regularization dropout
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass for the MLP block.

        Args:
            x: Input tensor of shape [batch_size, seq_len, n_embd] ([B, T, C]).

        Returns:
            Output tensor of shape [batch_size, seq_len, n_embd] ([B, T, C]).
        """
        # 1. Up-projection to 4x dimension: [B, T, C] -> [B, T, ffn_dim]
        x = self.c_fc(x)

        # 2. Non-linear activation: [B, T, ffn_dim]
        x = self.act(x)

        # 3. Down-projection back to embedding dimension: [B, T, ffn_dim] -> [B, T, C]
        x = self.c_proj(x)

        # 4. Regularization dropout: [B, T, C]
        x = self.dropout(x)

        return x
