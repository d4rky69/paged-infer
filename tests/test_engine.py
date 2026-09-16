"""End-to-end integration tests for AsyncPagedInferEngine."""

import asyncio
import pytest
from paged_infer.engine import AsyncPagedInferEngine
from paged_infer.models.transformer import ModelConfig
from paged_infer.scheduler.sequence import SamplingParams


@pytest.mark.asyncio
async def test_engine_single_stream():
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        block_size=8,
    )
    engine = AsyncPagedInferEngine(
        model_config=config,
        num_blocks=16,
        block_size=8,
        max_batch_size=4,
    )
    await engine.start()

    prompt = "paged attention continuous batching"
    params = SamplingParams(max_tokens=5, temperature=0.0)

    tokens = []
    async for chunk in engine.add_request(prompt, params):
        tokens.append(chunk)

    assert len(tokens) == 5
    assert engine.total_requests_completed == 1
    assert engine.total_tokens_generated >= 5

    await engine.stop()


@pytest.mark.asyncio
async def test_engine_concurrent_continuous_batching():
    config = ModelConfig(
        vocab_size=100,
        hidden_size=64,
        num_layers=2,
        num_heads=4,
        block_size=8,
    )
    engine = AsyncPagedInferEngine(
        model_config=config,
        num_blocks=32,
        block_size=8,
        max_batch_size=8,
    )
    await engine.start()

    async def run_client(req_id: int, max_tokens: int):
        prompt = f"query {req_id} testing throughput"
        params = SamplingParams(max_tokens=max_tokens, temperature=0.7)
        collected = []
        async for chunk in engine.add_request(prompt, params):
            collected.append(chunk)
        return len(collected)

    # Launch 5 concurrent clients with staggered token lengths (Continuous batching test)
    tasks = [
        run_client(1, max_tokens=3),
        run_client(2, max_tokens=8),
        run_client(3, max_tokens=4),
        run_client(4, max_tokens=6),
        run_client(5, max_tokens=5),
    ]

    results = await asyncio.gather(*tasks)
    assert results == [3, 8, 4, 6, 5]
    assert engine.total_requests_completed == 5

    telemetry = engine.get_telemetry()
    assert telemetry["total_requests_completed"] == 5
    assert telemetry["total_tokens_generated"] == sum([3, 8, 4, 6, 5])
    assert telemetry["memory"]["used_blocks"] == 0  # All returned to pool!

    await engine.stop()
