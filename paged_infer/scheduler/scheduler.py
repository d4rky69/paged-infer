"""Continuous / Dynamic Batching Scheduler for PagedInfer.

Implements token-level scheduling, memory-aware admission control, and
preemption recovery under KV-cache memory pressure.
"""

from collections import deque
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from paged_infer.memory.block_manager import BlockSpaceManager
from paged_infer.scheduler.sequence import Sequence, SequenceStatus


@dataclass
class SchedulerOutputs:
    """Carries the batch of sequences to be executed in the current engine step."""
    scheduled_seqs: List[Sequence] = field(default_factory=list)
    is_prefill: bool = False
    blocks_to_append: Dict[str, tuple[int, int]] = field(default_factory=dict)
    preempted_seq_ids: List[str] = field(default_factory=list)

    @property
    def is_empty(self) -> bool:
        return len(self.scheduled_seqs) == 0


class Scheduler:
    """Coordinates token-level dynamic batching and KV-cache lifecycle.
    
    Adheres to the continuous batching paradigm:
    - Existing decode requests are advanced iteratively.
    - New requests (prefill) are interleaved dynamically whenever memory allows.
    - Sequences that finish early vacate their memory slots immediately without
      blocking the remaining batch (eliminating the static batching bubble).
    """

    def __init__(
        self,
        block_manager: BlockSpaceManager,
        max_batch_size: int = 16,
        max_num_batched_tokens: int = 2048,
    ):
        self.block_manager: BlockSpaceManager = block_manager
        self.max_batch_size: int = max_batch_size
        self.max_num_batched_tokens: int = max_num_batched_tokens

        # Queues
        self.waiting: deque[Sequence] = deque()
        self.running: List[Sequence] = []

        # Telemetry
        self.total_preemptions: int = 0
        self.total_completed: int = 0

    def add_sequence(self, sequence: Sequence) -> None:
        """Enqueues a newly submitted request into the waiting queue."""
        sequence.status = SequenceStatus.WAITING
        self.waiting.append(sequence)

    def abort_sequence(self, seq_id: str) -> None:
        """Aborts an in-flight or waiting sequence and releases its memory."""
        # Check waiting queue
        self.waiting = deque([s for s in self.waiting if s.seq_id != seq_id])
        # Check running
        for seq in list(self.running):
            if seq.seq_id == seq_id:
                self.free_sequence(seq)

    def free_sequence(self, sequence: Sequence) -> None:
        """Frees the sequence from the running list and reclaims its KV cache."""
        if sequence in self.running:
            self.running.remove(sequence)
        self.block_manager.free(sequence.seq_id)
        self.total_completed += 1

    def has_unfinished_sequences(self) -> bool:
        """Returns True if there are requests waiting or actively generating."""
        return bool(self.waiting or self.running)

    def schedule(self) -> SchedulerOutputs:
        """Computes the execution plan for the next forward pass.
        
        Algorithm:
        1. If there are running sequences, attempt to schedule their decode step.
           If memory is insufficient, preempt the lowest priority (most recently added) sequence.
        2. If capacity remains, admit waiting requests for prefill.
        """
        # --- 1. Decode Phase for Running Sequences ---
        if self.running:
            scheduled_decodes: List[Sequence] = []
            blocks_to_append: Dict[str, tuple[int, int]] = {}
            preempted_seq_ids: List[str] = []

            # Prioritize older running sequences
            for seq in list(self.running):
                if self.block_manager.can_append_slot(seq.seq_id):
                    # Sequence can advance safely
                    block_num, offset = self.block_manager.append_slot(seq.seq_id)
                    blocks_to_append[seq.seq_id] = (block_num, offset)
                    scheduled_decodes.append(seq)
                else:
                    # Preemption required: free KV cache and move back to head of waiting
                    self.block_manager.free(seq.seq_id)
                    seq.status = SequenceStatus.PREEMPTED
                    self.running.remove(seq)
                    self.waiting.appendleft(seq)
                    preempted_seq_ids.append(seq.seq_id)
                    self.total_preemptions += 1

            if scheduled_decodes:
                return SchedulerOutputs(
                    scheduled_seqs=scheduled_decodes,
                    is_prefill=False,
                    blocks_to_append=blocks_to_append,
                    preempted_seq_ids=preempted_seq_ids,
                )

        # --- 2. Prefill Phase for Waiting Sequences ---
        scheduled_prefills: List[Sequence] = []
        num_batched_tokens = 0

        while self.waiting:
            if len(self.running) >= self.max_batch_size:
                break

            seq = self.waiting[0]
            tokens_needed = seq.num_prompt_tokens

            if num_batched_tokens + tokens_needed > self.max_num_batched_tokens:
                break

            if not self.block_manager.can_allocate(tokens_needed):
                # Cannot allocate blocks for this sequence yet
                break

            # Admit sequence
            seq = self.waiting.popleft()
            self.block_manager.allocate(
                seq_id=seq.seq_id,
                num_tokens=tokens_needed,
                token_ids=seq.prompt_token_ids,
            )
            seq.status = SequenceStatus.RUNNING
            scheduled_prefills.append(seq)
            self.running.append(seq)
            num_batched_tokens += tokens_needed

        if scheduled_prefills:
            return SchedulerOutputs(
                scheduled_seqs=scheduled_prefills,
                is_prefill=True,
            )

        return SchedulerOutputs()
