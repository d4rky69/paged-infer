"""PagedAttention Kernel and Paged KV-Cache Storage Engine.

Implements non-contiguous Key-Value cache storage in physical blocks and
executes attention directly against physical memory addresses via page tables.
Eliminates internal and external KV-cache fragmentation.
"""

import math
from typing import List, Optional, Tuple
import numpy as np


class PagedKVCache:
    """Pre-allocated physical tensor storage for Key and Value vectors.
    
    Tensors are laid out as:
    (num_layers, num_blocks, block_size, num_heads, head_dim)
    """

    def __init__(
        self,
        num_layers: int,
        num_blocks: int,
        block_size: int,
        num_heads: int,
        head_dim: int,
        dtype: np.dtype = np.float32,
    ):
        self.num_layers: int = num_layers
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        self.num_heads: int = num_heads
        self.head_dim: int = head_dim
        self.dtype: np.dtype = dtype

        # Pre-allocate contiguous pool of physical memory frames
        self.k_cache: np.ndarray = np.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
        )
        self.v_cache: np.ndarray = np.zeros(
            (num_layers, num_blocks, block_size, num_heads, head_dim),
            dtype=dtype,
        )

    def write_single_token(
        self,
        layer_idx: int,
        block_number: int,
        offset: int,
        key: np.ndarray,
        value: np.ndarray,
    ) -> None:
        """Writes Key and Value vectors for a single token into a physical block slot.
        
        Args:
            layer_idx: Transformer layer index.
            block_number: Physical block frame ID.
            offset: Slot index within the physical block [0, block_size - 1].
            key: Array of shape (num_heads, head_dim).
            value: Array of shape (num_heads, head_dim).
        """
        self.k_cache[layer_idx, block_number, offset] = key
        self.v_cache[layer_idx, block_number, offset] = value

    def write_batch_prefill(
        self,
        layer_idx: int,
        block_numbers: List[int],
        keys: np.ndarray,
        values: np.ndarray,
    ) -> None:
        """Writes prompt prefill KV tokens across mapped physical blocks.
        
        Args:
            layer_idx: Transformer layer index.
            block_numbers: Ordered list of physical block numbers from page table.
            keys: Array of shape (seq_len, num_heads, head_dim).
            values: Array of shape (seq_len, num_heads, head_dim).
        """
        seq_len = keys.shape[0]
        curr_tok = 0
        for block_num in block_numbers:
            if curr_tok >= seq_len:
                break
            tokens_in_block = min(self.block_size, seq_len - curr_tok)
            self.k_cache[layer_idx, block_num, :tokens_in_block] = keys[curr_tok : curr_tok + tokens_in_block]
            self.v_cache[layer_idx, block_num, :tokens_in_block] = values[curr_tok : curr_tok + tokens_in_block]
            curr_tok += tokens_in_block


def paged_attention_decode(
    query: np.ndarray,
    kv_cache: PagedKVCache,
    layer_idx: int,
    block_table: List[int],
    seq_len: int,
    scale: Optional[float] = None,
) -> np.ndarray:
    """Executes PagedAttention for a single token decode step.
    
    Reads keys and values directly from non-contiguous physical blocks mapped
    by the sequence's block_table.
    
    Args:
        query: Query vector of shape (num_heads, head_dim).
        kv_cache: The pre-allocated physical PagedKVCache storage.
        layer_idx: Transformer layer index.
        block_table: Physical block numbers mapped to this sequence.
        seq_len: Total context length (prompt + output tokens up to current step).
        scale: Attention softmax scaling factor (1 / sqrt(head_dim)).
        
    Returns:
        Attention output vector of shape (num_heads, head_dim).
    """
    num_heads, head_dim = query.shape
    block_size = kv_cache.block_size
    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Gather K and V chunks from physical blocks up to seq_len
    k_blocks = []
    v_blocks = []
    tokens_gathered = 0

    for block_num in block_table:
        if tokens_gathered >= seq_len:
            break
        num_toks = min(block_size, seq_len - tokens_gathered)
        k_chunk = kv_cache.k_cache[layer_idx, block_num, :num_toks]  # (num_toks, num_heads, head_dim)
        v_chunk = kv_cache.v_cache[layer_idx, block_num, :num_toks]  # (num_toks, num_heads, head_dim)
        k_blocks.append(k_chunk)
        v_blocks.append(v_chunk)
        tokens_gathered += num_toks

    # Concatenate gathered tokens: shape (seq_len, num_heads, head_dim)
    all_k = np.concatenate(k_blocks, axis=0)
    all_v = np.concatenate(v_blocks, axis=0)

    # Attention calculation per head:
    # Q: (num_heads, head_dim) -> (num_heads, 1, head_dim)
    # K: (seq_len, num_heads, head_dim) -> transpose to (num_heads, seq_len, head_dim)
    q_expanded = query[:, np.newaxis, :]  # (num_heads, 1, head_dim)
    k_transposed = np.transpose(all_k, (1, 0, 2))  # (num_heads, seq_len, head_dim)
    v_transposed = np.transpose(all_v, (1, 0, 2))  # (num_heads, seq_len, head_dim)

    # Q @ K^T -> (num_heads, 1, seq_len)
    scores = np.matmul(q_expanded, np.transpose(k_transposed, (0, 2, 1))) * scale

    # Numerically stable Softmax over seq_len
    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    attn_weights = exp_scores / (np.sum(exp_scores, axis=-1, keepdims=True) + 1e-9)

    # Attn_weights @ V -> (num_heads, 1, head_dim)
    output = np.matmul(attn_weights, v_transposed).squeeze(1)  # (num_heads, head_dim)
    return output
