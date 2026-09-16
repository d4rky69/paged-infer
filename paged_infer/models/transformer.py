"""Mini Transformer architecture with native PagedAttention execution."""

from dataclasses import dataclass
import math
from typing import List, Optional, Tuple
import numpy as np

from paged_infer.models.paged_attention import PagedKVCache, paged_attention_decode


@dataclass
class ModelConfig:
    """Hyperparameters defining the transformer architecture."""
    vocab_size: int = 4096
    hidden_size: int = 256
    num_layers: int = 4
    num_heads: int = 8
    block_size: int = 16
    max_seq_len: int = 2048

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads


class MiniTransformer:
    """Lightweight Transformer executing attention directly via PagedAttention.
    
    Provides complete end-to-end forward passes for both prompt prefill and
    token decode without external dependencies.
    """

    def __init__(self, config: Optional[ModelConfig] = None, seed: int = 42):
        self.config: ModelConfig = config or ModelConfig()
        np.random.seed(seed)

        c = self.config
        # Token Embeddings
        self.wte: np.ndarray = np.random.randn(c.vocab_size, c.hidden_size).astype(np.float32) * 0.02

        # Layer weights
        self.layers = []
        for _ in range(c.num_layers):
            layer_weights = {
                # Q, K, V projections
                "wq": np.random.randn(c.hidden_size, c.hidden_size).astype(np.float32) * 0.02,
                "wk": np.random.randn(c.hidden_size, c.hidden_size).astype(np.float32) * 0.02,
                "wv": np.random.randn(c.hidden_size, c.hidden_size).astype(np.float32) * 0.02,
                "wo": np.random.randn(c.hidden_size, c.hidden_size).astype(np.float32) * 0.02,
                # Feed-forward network (SwiGLU-style)
                "w1": np.random.randn(c.hidden_size, c.hidden_size * 2).astype(np.float32) * 0.02,
                "w2": np.random.randn(c.hidden_size * 2, c.hidden_size).astype(np.float32) * 0.02,
            }
            self.layers.append(layer_weights)

        # Output projection head
        self.lm_head: np.ndarray = self.wte.T  # Tied embeddings

    def _rms_norm(self, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
        return x / np.sqrt(np.mean(x ** 2, axis=-1, keepdims=True) + eps)

    def prefill(
        self,
        prompt_token_ids: List[int],
        kv_cache: PagedKVCache,
        block_table: List[int],
    ) -> np.ndarray:
        """Executes full prompt prefill and stores Key/Value cache into physical blocks.
        
        Args:
            prompt_token_ids: Prompt token IDs of shape (seq_len,).
            kv_cache: Global PagedKVCache instance.
            block_table: Physical block mapping for this sequence.
            
        Returns:
            Logits for the final prompt token (vocab_size,).
        """
        c = self.config
        seq_len = len(prompt_token_ids)
        h = self.wte[prompt_token_ids]  # (seq_len, hidden_size)

        for layer_idx, layer in enumerate(self.layers):
            normed_h = self._rms_norm(h)

            # Compute Q, K, V
            q = normed_h @ layer["wq"]  # (seq_len, hidden_size)
            k = normed_h @ layer["wk"]  # (seq_len, hidden_size)
            v = normed_h @ layer["wv"]  # (seq_len, hidden_size)

            # Reshape to (seq_len, num_heads, head_dim)
            q = q.reshape(seq_len, c.num_heads, c.head_dim)
            k = k.reshape(seq_len, c.num_heads, c.head_dim)
            v = v.reshape(seq_len, c.num_heads, c.head_dim)

            # Write K and V into physical blocks mapped by block_table
            kv_cache.write_batch_prefill(layer_idx, block_table, k, v)

            # Causal self-attention for prefill
            # Q: (num_heads, seq_len, head_dim)
            q_trans = np.transpose(q, (1, 0, 2))
            k_trans = np.transpose(k, (1, 0, 2))
            v_trans = np.transpose(v, (1, 0, 2))

            scale = 1.0 / math.sqrt(c.head_dim)
            scores = np.matmul(q_trans, np.transpose(k_trans, (0, 2, 1))) * scale

            # Apply causal mask (lower triangular)
            mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
            scores[:, mask] = -1e9

            # Softmax
            max_scores = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - max_scores)
            attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

            attn_out = np.matmul(attn_weights, v_trans)  # (num_heads, seq_len, head_dim)
            attn_out = np.transpose(attn_out, (1, 0, 2)).reshape(seq_len, c.hidden_size)

            # Residual + Output projection
            h = h + (attn_out @ layer["wo"])

            # Feed-forward network (GELU)
            normed_h2 = self._rms_norm(h)
            ffn_hidden = np.maximum(0, normed_h2 @ layer["w1"])  # ReLU / GELU approx
            h = h + (ffn_hidden @ layer["w2"])

        # Final norm and head projection for last token only
        final_h = self._rms_norm(h[-1:])  # (1, hidden_size)
        logits = (final_h @ self.lm_head).squeeze(0)  # (vocab_size,)
        return logits

    def decode(
        self,
        token_id: int,
        kv_cache: PagedKVCache,
        block_table: List[int],
        seq_len: int,
        new_block_num: int,
        offset: int,
    ) -> np.ndarray:
        """Executes one autoregressive decode step using PagedAttention.
        
        Args:
            token_id: The most recently generated token ID.
            kv_cache: Global PagedKVCache storage.
            block_table: Sequence's physical block table.
            seq_len: Total context length including the new token.
            new_block_num: Physical block ID to write the new KV pair to.
            offset: Slot offset within the physical block.
            
        Returns:
            Logits for the next token (vocab_size,).
        """
        c = self.config
        h = self.wte[token_id]  # (hidden_size,)

        for layer_idx, layer in enumerate(self.layers):
            normed_h = self._rms_norm(h)

            # Compute Q, K, V for this single token
            q = (normed_h @ layer["wq"]).reshape(c.num_heads, c.head_dim)
            k = (normed_h @ layer["wk"]).reshape(c.num_heads, c.head_dim)
            v = (normed_h @ layer["wv"]).reshape(c.num_heads, c.head_dim)

            # Write newly generated K, V into assigned physical slot
            kv_cache.write_single_token(layer_idx, new_block_num, offset, k, v)

            # PagedAttention Kernel execution
            attn_out = paged_attention_decode(
                query=q,
                kv_cache=kv_cache,
                layer_idx=layer_idx,
                block_table=block_table,
                seq_len=seq_len,
            )  # (num_heads, head_dim)

            attn_proj = attn_out.reshape(c.hidden_size) @ layer["wo"]
            h = h + attn_proj

            # Feed-forward network
            normed_h2 = self._rms_norm(h)
            ffn_hidden = np.maximum(0, normed_h2 @ layer["w1"])
            h = h + (ffn_hidden @ layer["w2"])

        # Final projection to vocabulary logits
        final_h = self._rms_norm(h)
        logits = final_h @ self.lm_head  # (vocab_size,)
        return logits
