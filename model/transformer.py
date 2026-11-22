"""Decoder-Only GPT Architecture in PyTorch from Scratch.

Educational, modular implementation of a causal autoregressive Transformer language model,
featuring:
- Pre-LayerNorm residual architecture (GPT-2 / modern standard)
- Multi-Head Causal Self-Attention with scaled dot-product and causal masking
- Position-wise Feed-Forward Network (MLP) with GELU
- Learned token and positional embeddings
- Parameter tying between token embeddings and output LM head
- Standard GPT-2 weight initialization with residual projection scaling
- Detailed mathematical parameter breakdown and inspection utilities
- Autoregressive generation method with temperature and top-k sampling
"""
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.embeddings import GPTEmbeddings
from model.attention import CausalSelfAttention
from model.mlp import MLP


@dataclass
class GPTConfig:
    """Hyperparameter configuration for the GPT architecture."""

    vocab_size: int = 16384         # Number of tokens in vocabulary
    block_size: int = 1024          # Maximum context window length (T)
    n_layer: int = 12               # Number of Transformer layers (L)
    n_head: int = 12                # Number of attention heads (H)
    n_embd: int = 768               # Embedding dimension (d_model)
    ffn_dim: int = 3072             # Intermediate feed-forward dimension (4 * n_embd)
    dropout: float = 0.1            # Dropout rate for regularization
    bias: bool = True               # Learnable bias in Linear and LayerNorm layers
    tie_weights: bool = True        # Tie token embeddings and LM head weights
    use_flash: bool = True          # Use PyTorch 2.0 scaled_dot_product_attention

    @classmethod
    def from_tokenizer_config(cls, config_path: str or Path, **kwargs) -> "GPTConfig":
        """Instantiate GPTConfig reading vocab_size from tokenizer_config.json."""
        path = Path(config_path)
        vocab_size = cls.vocab_size             
        if path.exists():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                vocab_size = data.get("vocab_size", vocab_size)
        return cls(vocab_size=vocab_size, **kwargs)


class TransformerBlock(nn.Module):
    """Single Transformer Decoder Block with Pre-LayerNorm architecture.

    Pre-LN formulation:
        x = x + Attention(LayerNorm_1(x))
        x = x + MLP(LayerNorm_2(x))

    Pre-LayerNorm ensures that gradients pass through the residual stream
    unimpeded during backpropagation, making deep Transformers much more
    numerically stable to train than Post-LN architectures.
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        # Pre-attention LayerNorm
        self.ln_1 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)

        # Causal Self-Attention
        self.attn = CausalSelfAttention(
            n_embd=config.n_embd,
            n_head=config.n_head,
            block_size=config.block_size,
            dropout=config.dropout,
            bias=config.bias,
            use_flash=config.use_flash,
        )

        # Pre-MLP LayerNorm
        self.ln_2 = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)

        # Position-wise Feed-Forward Network
        self.mlp = MLP(
            n_embd=config.n_embd,
            ffn_dim=config.ffn_dim,
            dropout=config.dropout,
            bias=config.bias,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for a single Transformer block.

        Args:
            x: Input tensor [batch_size, seq_len, n_embd] ([B, T, C]).
            attention_mask: Optional attention mask tensor.

        Returns:
            Output tensor [batch_size, seq_len, n_embd] ([B, T, C]).
        """
        # Residual connection 1: Attention branch with Pre-LN
        x = x + self.attn(self.ln_1(x), attention_mask=attention_mask)

        # Residual connection 2: MLP branch with Pre-LN
        x = x + self.mlp(self.ln_2(x))

        return x


class GPT(nn.Module):
    """Full Decoder-Only GPT Language Model.

    Comprises:
    1. Token & learned 1D positional embeddings
    2. Stack of N identical Transformer decoder blocks
    3. Final LayerNorm
    4. Linear Language Model (LM) Head with weight tying
    """

    def __init__(self, config: GPTConfig) -> None:
        super().__init__()
        self.config = config

        # 1. Embeddings: Token embeddings + Positional embeddings
        self.embeddings = GPTEmbeddings(
            vocab_size=config.vocab_size,
            block_size=config.block_size,
            n_embd=config.n_embd,
            dropout=config.dropout,
        )

        # 2. Transformer Decoder Blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layer)])

        # 3. Final LayerNorm before projection to vocabulary
        self.ln_f = nn.LayerNorm(config.n_embd, elementwise_affine=config.bias)

        # 4. Output Language Model Head: projects [n_embd] -> [vocab_size]
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying: share weights between input token embeddings and output projection
        # This dramatically reduces parameter count and improves language representation.
        if config.tie_weights:
            self.lm_head.weight = self.embeddings.token_embeddings.weight

        # 5. Initialize model parameters
        self.apply(self._init_weights)

        # Apply special scaled initialization for residual projection layers (GPT-2 style)
        # Scaled by 1 / sqrt(2 * n_layer) to prevent activation variance growth across residual depth
        for pn, p in self.named_parameters():
            if pn.endswith("c_proj.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layer))

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights according to standard GPT-2 scheme:

        - Linear and Embedding weights ~ N(0, 0.02)
        - Biases = 0
        - LayerNorm weights = 1, biases = 0
        """
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            if module.bias is not None:
                nn.init.zeros_(module.bias)
            if module.weight is not None:
                nn.init.ones_(module.weight)

    def forward(
        self,
        idx: torch.Tensor,
        targets: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass for GPT language model.

        Args:
            idx: Input token indices tensor of shape [batch_size, seq_len] ([B, T]).
            targets: Optional ground-truth token indices of shape [batch_size, seq_len] ([B, T]).
                     When provided, Cross-Entropy loss is computed against next-token targets.
            attention_mask: Optional custom mask tensor.

        Returns:
            Tuple of:
            - logits: Next-token logits of shape [batch_size, seq_len, vocab_size] ([B, T, V]).
            - loss: Cross-Entropy loss scalar if targets provided, else None.

        Raises:
            ValueError: If input sequence length exceeds configured block_size.
        """
        device = idx.device
        B, T = idx.size()

        if T > self.config.block_size:
            raise ValueError(
                f"Sequence length {T} exceeds maximum context window (block_size={self.config.block_size})."
            )

        # 1. Embeddings: [B, T] -> [B, T, n_embd]
        x = self.embeddings(idx)

        # 2. Sequential Transformer decoder blocks: [B, T, n_embd] -> [B, T, n_embd]
        for block in self.blocks:
            x = block(x, attention_mask=attention_mask)

        # 3. Final LayerNorm: [B, T, n_embd] -> [B, T, n_embd]
        x = self.ln_f(x)

        # 4. LM Head projection: [B, T, n_embd] -> [B, T, vocab_size]
        if targets is not None:
            # Training mode: compute logits for all positions
            logits = self.lm_head(x)  # [B, T, V]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1,
            )
        else:
            # Inference mode: full sequence logits
            logits = self.lm_head(x)  # [B, T, V]
            loss = None

        return logits, loss

    def get_num_params(self, non_embedding: bool = False) -> int:
        """Return the number of parameters in the model.

        Args:
            non_embedding: If True, subtract positional embeddings.
        """
        # Count unique parameters (accounting for tied weights)
        unique_params = set(self.parameters())
        n_params = sum(p.numel() for p in unique_params)
        if non_embedding:
            n_params -= self.embeddings.position_embeddings.weight.numel()
        return n_params

    def parameter_breakdown(self) -> Dict[str, Any]:
        """Compute an exact mathematical and parameter count breakdown of the model.

        Returns:
            Dictionary with breakdown per module, counts, and percentages.
        """
        cfg = self.config
        V = cfg.vocab_size
        T = cfg.block_size
        d = cfg.n_embd
        L = cfg.n_layer
        bias = cfg.bias

        # 1. Embeddings
        wte_params = V * d
        wpe_params = T * d
        emb_total = wte_params + wpe_params

        # 2. Per-Block Attention
        # QKV linear projection
        qkv_w = d * (3 * d)
        qkv_b = 3 * d if bias else 0
        qkv_total = qkv_w + qkv_b

        # Output projection
        proj_w = d * d
        proj_b = d if bias else 0
        proj_total = proj_w + proj_b
        attn_per_block = qkv_total + proj_total

        # 3. Per-Block MLP
        fc_w = d * cfg.ffn_dim
        fc_b = cfg.ffn_dim if bias else 0
        mlp_proj_w = cfg.ffn_dim * d
        mlp_proj_b = d if bias else 0
        mlp_per_block = fc_w + fc_b + mlp_proj_w + mlp_proj_b

        # 4. LayerNorms
        # 2 LayerNorms per block: weight (d) + bias (d if bias else 0)
        ln_per_layer = 2 * (d + (d if bias else 0))
        block_total = ln_per_layer + attn_per_block + mlp_per_block
        all_blocks_total = L * block_total

        # Final LayerNorm
        ln_f_total = d + (d if bias else 0)

        # Total LayerNorms across the entire model: (2 * L + 1)
        total_ln_params = (2 * L + 1) * (d + (d if bias else 0))

        # LM Head: tied with token embeddings (0 additional unique parameters)
        lm_head_unique = 0 if cfg.tie_weights else (d * V)

        # Theoretical unique total
        theoretical_total = emb_total + all_blocks_total + ln_f_total + lm_head_unique

        # Actual PyTorch parameter count
        unique_pytorch_params = sum(p.numel() for p in set(self.parameters()))
        trainable_pytorch_params = sum(p.numel() for p in set(self.parameters()) if p.requires_grad)

        breakdown = {
            "config": {
                "vocab_size": V,
                "block_size": T,
                "n_layer": L,
                "n_head": cfg.n_head,
                "n_embd": d,
                "head_dim": d // cfg.n_head,
                "ffn_dim": cfg.ffn_dim,
                "bias": bias,
                "tie_weights": cfg.tie_weights,
            },
            "token_embeddings": {
                "count": wte_params,
                "shape": f"[{V}, {d}]",
                "percent": (wte_params / theoretical_total) * 100,
            },
            "position_embeddings": {
                "count": wpe_params,
                "shape": f"[{T}, {d}]",
                "percent": (wpe_params / theoretical_total) * 100,
            },
            "total_embeddings": {
                "count": emb_total,
                "percent": (emb_total / theoretical_total) * 100,
            },
            "per_block": {
                "attention": attn_per_block,
                "mlp": mlp_per_block,
                "layernorms": ln_per_layer,
                "total": block_total,
            },
            "all_blocks": {
                "count": all_blocks_total,
                "percent": (all_blocks_total / theoretical_total) * 100,
            },
            "total_layernorms": {
                "count": total_ln_params,
                "percent": (total_ln_params / theoretical_total) * 100,
            },
            "total_attention_all_layers": {
                "count": L * attn_per_block,
                "percent": ((L * attn_per_block) / theoretical_total) * 100,
            },
            "total_mlp_all_layers": {
                "count": L * mlp_per_block,
                "percent": ((L * mlp_per_block) / theoretical_total) * 100,
            },
            "final_layernorm": {
                "count": ln_f_total,
                "percent": (ln_f_total / theoretical_total) * 100,
            },
            "lm_head_unique": {
                "count": lm_head_unique,
                "tied": cfg.tie_weights,
            },
            "theoretical_total": theoretical_total,
            "actual_unique_params": unique_pytorch_params,
            "actual_trainable_params": trainable_pytorch_params,
            "match": theoretical_total == unique_pytorch_params,
        }

        return breakdown

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        min_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        no_repeat_ngram_size: Optional[int] = None,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """Autoregressive text generation with advanced sampling controls.

        Takes a conditioning sequence of indices idx [B, T] and generates
        max_new_tokens new tokens autoregressively.

        Args:
            idx: Conditioning prompt tensor of shape [batch_size, seq_len] ([B, T]).
            max_new_tokens: Number of new tokens to sample.
            temperature: Softmax temperature (> 0). Lower = more deterministic,
                         higher = more creative/random. (0.0 = greedy).
            top_k: If set, only sample from the top k highest probability tokens.
            top_p: If set, sample from cumulative probability nucleus top_p.
            min_p: If set, prune tokens with prob < min_p * max_prob.
            repetition_penalty: CTRL/HF repetition penalty (divide pos, multiply neg).
            no_repeat_ngram_size: Forbids repetition of n-grams of this length.
            eos_token_id: If set, generation stops when this token is sampled.

        Returns:
            Tensor of generated token indices with shape [B, T + max_new_tokens].
        """
        for _ in range(max_new_tokens):
            # Crop current context if it exceeds the maximum context window
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Forward pass to get logits for the conditioning sequence
            logits, _ = self(idx_cond)

            # Pluck logits at the final time step: [B, T, V] -> [B, V]
            logits = logits[:, -1, :].clone()

            batch_size = logits.size(0)

            # 1. Repetition penalty (CTRL / Hugging Face formulation)
            if repetition_penalty != 1.0 and repetition_penalty > 0:
                for b in range(batch_size):
                    seen_tokens = torch.unique(idx[b])
                    seen_logits = logits[b, seen_tokens]
                    logits[b, seen_tokens] = torch.where(
                        seen_logits > 0,
                        seen_logits / repetition_penalty,
                        seen_logits * repetition_penalty,
                    )

            # 2. No-repeat n-gram blocking
            if no_repeat_ngram_size is not None and no_repeat_ngram_size > 0:
                n = no_repeat_ngram_size
                for b in range(batch_size):
                    seq = idx[b].tolist()
                    seq_len = len(seq)
                    if seq_len >= n:
                        prefix = tuple(seq[-(n - 1):])
                        banned_tokens = set()
                        for i in range(seq_len - n + 1):
                            if tuple(seq[i : i + n - 1]) == prefix:
                                banned_tokens.add(seq[i + n - 1])

                        if banned_tokens:
                            banned_idx = list(banned_tokens)
                            backup_logits = logits[b].clone()
                            logits[b, banned_idx] = -float("Inf")
                            # Fallback if all tokens are masked
                            if torch.isneginf(logits[b]).all():
                                logits[b] = backup_logits

            # Greedy shortcut for temperature == 0
            if temperature == 0.0:
                idx_next = torch.argmax(logits, dim=-1, keepdim=True)
                idx = torch.cat((idx, idx_next), dim=1)
                if eos_token_id is not None and (idx_next == eos_token_id).all():
                    break
                continue

            # 3. Temperature scaling
            if temperature != 1.0 and temperature > 0:
                logits = logits / temperature

            # 4. Min-P filtering
            if min_p is not None and 0.0 < min_p < 1.0:
                probs_for_min_p = F.softmax(logits, dim=-1)
                p_max, _ = torch.max(probs_for_min_p, dim=-1, keepdim=True)
                min_p_threshold = p_max * min_p
                # Mask logits where probability falls below threshold
                min_p_mask = probs_for_min_p < min_p_threshold
                logits = torch.where(min_p_mask, torch.tensor(-float("Inf"), device=logits.device), logits)

            # 5. Top-k filtering: crop logits to top_k values
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            # 6. Top-p (nucleus) filtering: crop cumulative probability distribution
            if top_p is not None and 0.0 < top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Remove tokens with cumulative probability above threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = False

                indices_to_remove = sorted_indices_to_remove.scatter(
                    dim=1, index=sorted_indices, src=sorted_indices_to_remove
                )
                logits[indices_to_remove] = -float("Inf")

            # Fallback protection if all values became -Inf
            all_neginf = torch.isneginf(logits).all(dim=-1, keepdim=True)
            if all_neginf.any():
                for b in range(batch_size):
                    if torch.isneginf(logits[b]).all():
                        logits[b, 0] = 0.0

            # Convert logits to probabilities
            probs = F.softmax(logits, dim=-1)

            # Sample next token from categorical distribution: [B, 1]
            idx_next = torch.multinomial(probs, num_samples=1)

            # Append sampled token to running sequence: [B, T+1]
            idx = torch.cat((idx, idx_next), dim=1)

            # Stop early if all batch sequences produced eos_token_id
            if eos_token_id is not None and (idx_next == eos_token_id).all():
                break

        return idx
