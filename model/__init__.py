"""GPT model architecture package.

Exports:
    GPT: Decoder-only Transformer model.
    GPTConfig: Model hyperparameter configuration dataclass.
    GPTEmbeddings: Token and positional embedding module.
    CausalSelfAttention: Causal multi-head self-attention module.
    MLP: Feed-forward network module with GELU.
    TransformerBlock: Pre-LayerNorm Transformer decoder layer.
"""
from model.embeddings import GPTEmbeddings
from model.attention import CausalSelfAttention
from model.mlp import MLP
from model.transformer import GPT, GPTConfig, TransformerBlock

__all__ = [
    "GPT",
    "GPTConfig",
    "GPTEmbeddings",
    "CausalSelfAttention",
    "MLP",
    "TransformerBlock",
]

