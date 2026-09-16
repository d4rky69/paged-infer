"""Asynchronous serving engine for PagedInfer.

Orchestrates the continuous batching scheduler, physical KV-cache memory pool,
and transformer forward passes over an async event loop.
"""

import asyncio
from collections import deque
import time
import uuid
from typing import Any, AsyncGenerator, Dict, List, Optional

from paged_infer.memory.block_manager import BlockSpaceManager
from paged_infer.models.paged_attention import PagedKVCache
from paged_infer.models.sampler import Sampler
from paged_infer.models.tokenizer import SimpleTokenizer
from paged_infer.models.transformer import MiniTransformer, ModelConfig
from paged_infer.scheduler.scheduler import Scheduler
from paged_infer.scheduler.sequence import SamplingParams, Sequence, SequenceStatus


class AsyncPagedInferEngine:
    """Production-grade asynchronous serving engine for high-throughput LLM serving."""

    def __init__(
        self,
        model_config: Optional[ModelConfig] = None,
        num_blocks: int = 64,
        block_size: int = 16,
        max_batch_size: int = 16,
        max_num_batched_tokens: int = 2048,
        enable_prefix_caching: bool = True,
    ):
        self.model_config: ModelConfig = model_config or ModelConfig()
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size

        # 1. Initialize tokenizer and model
        self.tokenizer = SimpleTokenizer(vocab_size=self.model_config.vocab_size)
        self.model = MiniTransformer(self.model_config)

        # 2. Allocate physical KV-cache memory pool
        self.kv_cache = PagedKVCache(
            num_layers=self.model_config.num_layers,
            num_blocks=num_blocks,
            block_size=block_size,
            num_heads=self.model_config.num_heads,
            head_dim=self.model_config.head_dim,
        )

        # 3. Initialize Memory Manager (Page table allocator)
        self.block_manager = BlockSpaceManager(
            num_blocks=num_blocks,
            block_size=block_size,
            enable_prefix_caching=enable_prefix_caching,
        )

        # 4. Continuous batching scheduler
        self.scheduler = Scheduler(
            block_manager=self.block_manager,
            max_batch_size=max_batch_size,
            max_num_batched_tokens=max_num_batched_tokens,
        )

        # Stream output queues keyed by seq_id
        self.output_queues: Dict[str, asyncio.Queue[Optional[str]]] = {}
        self.sequence_registry: Dict[str, Sequence] = {}

        # Background step task
        self._step_task: Optional[asyncio.Task] = None
        self._running: bool = False
        self._new_work_event: asyncio.Event = asyncio.Event()

        # Engine throughput & latency telemetry
        self.total_tokens_generated: int = 0
        self.total_requests_completed: int = 0
        self.peak_used_blocks: int = 0
        self.start_time: float = time.time()
        self.latencies_ttft: List[float] = []
        self.latencies_e2e: List[float] = []
        self.completed_history: deque[Dict[str, Any]] = deque(maxlen=20)

    async def start(self) -> None:
        """Starts the background continuous batching execution loop."""
        if self._running:
            return
        self._running = True
        self._step_task = asyncio.create_task(self._engine_loop())

    async def stop(self) -> None:
        """Gracefully terminates the background step loop."""
        self._running = False
        if self._step_task:
            self._step_task.cancel()
            try:
                await self._step_task
            except asyncio.CancelledError:
                pass

    async def add_request(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> AsyncGenerator[str, None]:
        """Submits a prompt for generation and returns an async generator of streamed tokens."""
        if not self._running:
            await self.start()

        seq_id = f"req-{uuid.uuid4().hex[:8]}"
        prompt_token_ids = self.tokenizer.encode(prompt)
        sampling_params = sampling_params or SamplingParams()

        seq = Sequence(
            seq_id=seq_id,
            prompt=prompt,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )

        queue: asyncio.Queue[Optional[str]] = asyncio.Queue()
        self.output_queues[seq_id] = queue
        self.sequence_registry[seq_id] = seq

        # Enqueue to scheduler
        self.scheduler.add_sequence(seq)
        self._new_work_event.set()

        try:
            while True:
                token_chunk = await queue.get()
                if token_chunk is None:
                    # Generation completed
                    break
                yield token_chunk
        finally:
            if seq_id in self.output_queues:
                del self.output_queues[seq_id]

    async def generate_full(
        self,
        prompt: str,
        sampling_params: Optional[SamplingParams] = None,
    ) -> Dict[str, Any]:
        """Non-streaming convenience execution."""
        tokens: List[str] = []
        seq_id: Optional[str] = None
        async for chunk in self.add_request(prompt, sampling_params):
            tokens.append(chunk)

        # Retrieve sequence object for metadata
        full_text = "".join(tokens)
        return {
            "text": full_text,
            "num_tokens": len(tokens),
        }

    async def _engine_loop(self) -> None:
        """The heart of the inference engine: continuous iteration-level stepping."""
        while self._running:
            if not self.scheduler.has_unfinished_sequences():
                self._new_work_event.clear()
                try:
                    await asyncio.wait_for(self._new_work_event.wait(), timeout=0.1)
                except asyncio.TimeoutError:
                    continue

            # Execute single iteration step
            self._step()
            # Yield control back to event loop for low-latency streaming
            await asyncio.sleep(0.001)

    def _step(self) -> None:
        """Executes one step of either prefill or decode across scheduled sequences."""
        outputs = self.scheduler.schedule()
        if outputs.is_empty:
            return

        if outputs.is_prefill:
            # Prefill Phase
            for seq in outputs.scheduled_seqs:
                block_table = self.block_manager.get_block_table(seq.seq_id)
                logits = self.model.prefill(
                    prompt_token_ids=seq.prompt_token_ids,
                    kv_cache=self.kv_cache,
                    block_table=block_table,
                )
                # Sample first token
                next_tok = Sampler.sample(
                    logits=logits,
                    temperature=seq.sampling_params.temperature,
                    top_p=seq.sampling_params.top_p,
                    top_k=seq.sampling_params.top_k,
                )
                seq.append_token_id(next_tok)
                self.total_tokens_generated += 1

                # Send to output stream
                tok_text = self.tokenizer.decode([next_tok])
                if seq.seq_id in self.output_queues:
                    self.output_queues[seq.seq_id].put_nowait(tok_text)

                if seq.is_finished:
                    self._complete_sequence(seq)
        else:
            # Decode Phase
            for seq in outputs.scheduled_seqs:
                if seq.seq_id not in outputs.blocks_to_append:
                    continue
                block_num, offset = outputs.blocks_to_append[seq.seq_id]
                block_table = self.block_manager.get_block_table(seq.seq_id)

                logits = self.model.decode(
                    token_id=seq.get_last_token_id(),
                    kv_cache=self.kv_cache,
                    block_table=block_table,
                    seq_len=seq.total_tokens,
                    new_block_num=block_num,
                    offset=offset,
                )

                next_tok = Sampler.sample(
                    logits=logits,
                    temperature=seq.sampling_params.temperature,
                    top_p=seq.sampling_params.top_p,
                    top_k=seq.sampling_params.top_k,
                )
                seq.append_token_id(next_tok)
                self.total_tokens_generated += 1

                tok_text = self.tokenizer.decode([next_tok])
                if seq.seq_id in self.output_queues:
                    self.output_queues[seq.seq_id].put_nowait(tok_text)

                if seq.is_finished:
                    self._complete_sequence(seq)

        # Track peak blocks allocated
        self.peak_used_blocks = max(self.peak_used_blocks, self.block_manager.num_used_blocks)

    def _complete_sequence(self, seq: Sequence) -> None:
        """Handles sequence termination and telemetry logging."""
        if seq.time_to_first_token:
            self.latencies_ttft.append(seq.time_to_first_token)
        if seq.end_to_end_latency:
            self.latencies_e2e.append(seq.end_to_end_latency)

        self.completed_history.appendleft({
            "seq_id": seq.seq_id,
            "prompt_preview": seq.prompt[:32] + ("..." if len(seq.prompt) > 32 else ""),
            "prompt_tokens": seq.num_prompt_tokens,
            "output_tokens": seq.num_output_tokens,
            "ttft_ms": round(seq.time_to_first_token * 1000, 1) if seq.time_to_first_token else 0.0,
            "total_latency_s": round(seq.end_to_end_latency, 2) if seq.end_to_end_latency else 0.0,
            "tokens_per_sec": round(seq.num_output_tokens / max(seq.end_to_end_latency, 0.001), 1) if seq.end_to_end_latency else 0.0,
            "finish_reason": seq.finish_reason or "completed",
            "completed_at": time.strftime("%H:%M:%S"),
        })

        self.total_requests_completed += 1
        self.scheduler.free_sequence(seq)

        if seq.seq_id in self.output_queues:
            self.output_queues[seq.seq_id].put_nowait(None)  # Signal EOF

        if seq.seq_id in self.sequence_registry:
            del self.sequence_registry[seq.seq_id]

    def get_telemetry(self) -> Dict[str, Any]:
        """Provides real-time system metrics for the Web UI HUD and Prometheus endpoints."""
        elapsed = max(time.time() - self.start_time, 0.001)
        mem_stats = self.block_manager.get_stats()

        avg_ttft = (sum(self.latencies_ttft) / len(self.latencies_ttft)) if self.latencies_ttft else 0.0
        avg_e2e = (sum(self.latencies_e2e) / len(self.latencies_e2e)) if self.latencies_e2e else 0.0
        tokens_per_sec = self.total_tokens_generated / elapsed

        # Snapshot of currently running sequences
        active_seqs = [
            {
                "seq_id": s.seq_id,
                "status": s.status.value,
                "prompt_tokens": s.num_prompt_tokens,
                "output_tokens": s.num_output_tokens,
                "blocks_allocated": len(self.block_manager.get_block_table(s.seq_id)),
            }
            for s in self.scheduler.running
        ]

        # Block state map for visual grid
        block_states = []
        for b_id in range(self.num_blocks):
            blk = self.block_manager.all_blocks[b_id]
            if blk.ref_count == 0:
                state = "free"
            elif blk.ref_count > 1:
                state = "shared_prefix"
            else:
                state = "allocated"
            block_states.append({
                "id": b_id,
                "state": state,
                "tokens": blk.num_tokens,
                "ref_count": blk.ref_count,
            })

        return {
            "uptime_seconds": round(elapsed, 1),
            "total_tokens_generated": self.total_tokens_generated,
            "total_requests_completed": self.total_requests_completed,
            "tokens_per_sec": round(tokens_per_sec, 2),
            "avg_ttft_ms": round(avg_ttft * 1000, 2),
            "avg_e2e_latency_ms": round(avg_e2e * 1000, 2),
            "waiting_requests": len(self.scheduler.waiting),
            "running_requests": len(self.scheduler.running),
            "preemptions": self.scheduler.total_preemptions,
            "memory": mem_stats,
            "peak_used_blocks": self.peak_used_blocks,
            "active_sequences": active_seqs,
            "recent_completed": list(self.completed_history),
            "block_grid": block_states,
        }
