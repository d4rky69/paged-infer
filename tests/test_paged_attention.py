"""Unit tests for PagedAttention kernel, KV cache storage, and MiniTransformer."""

import numpy as np
import pytest
from paged_infer.memory.block_manager import BlockSpaceManager
from paged_infer.models.paged_attention import PagedKVCache, paged_attention_decode
from paged_infer.models.sampler import Sampler
from paged_infer.models.tokenizer import SimpleTokenizer
from paged_infer.models.transformer import MiniTransformer, ModelConfig


def test_paged_kv_cache_read_write():
    num_layers = 2
    num_blocks = 4
    block_size = 16
    num_heads = 4
    head_dim = 32

    cache = PagedKVCache(num_layers, num_blocks, block_size, num_heads, head_dim)

    # Write token at layer 0, block 2, offset 5
    dummy_k = np.ones((num_heads, head_dim), dtype=np.float32) * 3.14
    dummy_v = np.ones((num_heads, head_dim), dtype=np.float32) * 2.71

    cache.write_single_token(0, 2, 5, dummy_k, dummy_v)

    np.testing.assert_allclose(cache.k_cache[0, 2, 5], dummy_k)
    np.testing.assert_allclose(cache.v_cache[0, 2, 5], dummy_v)


def test_paged_attention_decode_output_shape():
    num_layers = 1
    num_blocks = 4
    block_size = 16
    num_heads = 4
    head_dim = 32

    cache = PagedKVCache(num_layers, num_blocks, block_size, num_heads, head_dim)

    # Fill 2 blocks (32 tokens) with random K and V
    cache.k_cache[0, 0] = np.random.randn(block_size, num_heads, head_dim)
    cache.v_cache[0, 0] = np.random.randn(block_size, num_heads, head_dim)
    cache.k_cache[0, 1] = np.random.randn(block_size, num_heads, head_dim)
    cache.v_cache[0, 1] = np.random.randn(block_size, num_heads, head_dim)

    query = np.random.randn(num_heads, head_dim).astype(np.float32)
    block_table = [0, 1]
    seq_len = 24  # 16 tokens in block 0, 8 tokens in block 1

    out = paged_attention_decode(
        query=query,
        kv_cache=cache,
        layer_idx=0,
        block_table=block_table,
        seq_len=seq_len,
    )

    assert out.shape == (num_heads, head_dim)
    assert not np.isnan(out).any()


def test_mini_transformer_prefill_and_decode():
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        block_size=8,
    )
    model = MiniTransformer(config, seed=42)
    cache = PagedKVCache(
        num_layers=config.num_layers,
        num_blocks=10,
        block_size=config.block_size,
        num_heads=config.num_heads,
        head_dim=config.head_dim,
    )

    # Sequence with prompt length 12 -> maps to 2 blocks of 8 tokens
    prompt_tokens = [10, 20, 30, 40, 50, 60, 70, 80, 15, 25, 35, 45]
    block_table = [3, 7]  # Arbitrary non-contiguous physical block IDs

    # 1. Prefill
    logits = model.prefill(prompt_tokens, cache, block_table)
    assert logits.shape == (config.vocab_size,)
    assert not np.isnan(logits).any()

    # Sample next token
    next_tok = Sampler.sample(logits, temperature=0.0)  # Greedy
    assert 0 <= next_tok < config.vocab_size

    # 2. Decode step (token 13 at block 7, offset 4)
    decode_logits = model.decode(
        token_id=next_tok,
        kv_cache=cache,
        block_table=block_table,
        seq_len=13,
        new_block_num=7,
        offset=4,
    )
    assert decode_logits.shape == (config.vocab_size,)
    assert not np.isnan(decode_logits).any()


def test_simple_tokenizer():
    tokenizer = SimpleTokenizer(vocab_size=1000)
    text = "paged attention continuous batching"
    tokens = tokenizer.encode(text)
    assert len(tokens) > 0

    decoded = tokenizer.decode(tokens)
    assert "paged" in decoded
    assert "attention" in decoded
