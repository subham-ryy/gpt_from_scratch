"""Multi-Head Causal Self-Attention for Decoder-Only GPT.

Implements scaled dot-product attention with autoregressive causal masking,
multi-head projections, and output projection with residual dropout.
"""
import math
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    """Causal Multi-Head Self-Attention (MHA) module.

    Projects input embeddings into Query (Q), Key (K), and Value (V) representations,
    computes attention scores scaled by 1/sqrt(d_k), applies a lower-triangular causal
    mask so tokens cannot attend to future positions, and projects back to embedding dimension.

    Args:
        n_embd: Total embedding dimension (d_model).
        n_head: Number of parallel attention heads.
        block_size: Maximum context length (T_max) for the causal mask.
        dropout: Dropout probability for attention weights and output projection.
        bias: Whether linear projection layers include learnable bias vectors.
        use_flash: If True, uses PyTorch 2.0+ scaled_dot_product_attention (FlashAttention).
    """

    def __init__(
        self,
        n_embd: int = 768,
        n_head: int = 12,
        block_size: int = 1024,
        dropout: float = 0.1,
        bias: bool = True,
        use_flash: bool = True,
    ) -> None:
        super().__init__()
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) must be divisible by n_head ({n_head})."

        self.n_embd = n_embd
        self.n_head = n_head
        self.head_dim = n_embd // n_head
        self.block_size = block_size
        self.dropout = dropout
        self.use_flash = use_flash and hasattr(F, "scaled_dot_product_attention")

        # Fused linear projection for Query, Key, and Value: [n_embd] -> [3 * n_embd]
        # Computing Q, K, V in a single matrix multiplication is significantly faster
        # than three separate linear operations.
        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=bias)

        # Output linear projection: [n_embd] -> [n_embd]
        self.c_proj = nn.Linear(n_embd, n_embd, bias=bias)

        # Regularization dropouts
        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

        # Causal mask buffer: lower triangular matrix with 1s on and below diagonal.
        # Shape: [1, 1, block_size, block_size]
        # Registered as buffer so it moves to GPU with module but is not a trainable parameter.
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(block_size, block_size)).view(1, 1, block_size, block_size),
            persistent=False,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute causal self-attention.

        Args:
            x: Input activations of shape [batch_size, seq_len, n_embd] ([B, T, C]).
            attention_mask: Optional additional mask tensor [B, 1, T, T] or [B, T].

        Returns:
            Output activations of shape [batch_size, seq_len, n_embd] ([B, T, C]).
        """
        B, T, C = x.size()

        # 1. Project input to Query, Key, Value:
        # [B, T, C] -> [B, T, 3 * C]
        qkv = self.c_attn(x)

        # 2. Split into Q, K, V: each is [B, T, C]
        q, k, v = qkv.split(self.n_embd, dim=2)

        # 3. Reshape and transpose for multi-head attention:
        # [B, T, C] -> [B, T, n_head, head_dim] -> [B, n_head, T, head_dim]
        q = q.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        # 4. Scaled Dot-Product Attention:
        # Fast path: use PyTorch 2.0 scaled_dot_product_attention if enabled and no custom mask
        if self.use_flash and attention_mask is None:
            # When is_causal=True, PyTorch automatically applies lower-triangular causal masking
            y = F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True,
            )
        else:
            # Educational explicit path:
            # Scaled dot-product: [B, n_head, T, head_dim] x [B, n_head, head_dim, T] -> [B, n_head, T, T]
            scale = 1.0 / math.sqrt(self.head_dim)
            att = (q @ k.transpose(-2, -1)) * scale

            # Apply causal mask: positions with 0 in the mask are filled with -inf
            # att[b, h, i, j] is masked out if j > i (future attending)
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))

            # Optional custom mask (e.g. for padding tokens)
            if attention_mask is not None:
                att = att + attention_mask

            # Softmax across key sequence dimension (sum across columns = 1.0)
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)

            # Weight values: [B, n_head, T, T] x [B, n_head, T, head_dim] -> [B, n_head, T, head_dim]
            y = att @ v

        # 5. Concatenate heads and project back to embedding dimension:
        # [B, n_head, T, head_dim] -> [B, T, n_head, head_dim] -> [B, T, C]
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # 6. Final linear projection and residual dropout:
        # [B, T, C] -> [B, T, C]
        y = self.resid_dropout(self.c_proj(y))

        return y
