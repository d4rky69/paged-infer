"""Unit tests for the Continuous Batching Scheduler."""

import pytest
from paged_infer.memory.block_manager import BlockSpaceManager
from paged_infer.scheduler.sequence import Sequence, SequenceStatus, SamplingParams
from paged_infer.scheduler.scheduler import Scheduler


def test_scheduler_prefill_and_decode_lifecycle():
    # 10 blocks of 16 tokens
    block_manager = BlockSpaceManager(num_blocks=10, block_size=16)
    scheduler = Scheduler(block_manager, max_batch_size=4)

    seq1 = Sequence(seq_id="seq-1", prompt="hello", prompt_token_ids=[1, 2, 3, 4])
    seq2 = Sequence(seq_id="seq-2", prompt="world", prompt_token_ids=[5, 6, 7])

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    assert len(scheduler.waiting) == 2
    assert len(scheduler.running) == 0

    # Step 1: Should schedule Prefill
    out1 = scheduler.schedule()
    assert out1.is_prefill
    assert len(out1.scheduled_seqs) == 2
    assert seq1.status == SequenceStatus.RUNNING
    assert seq2.status == SequenceStatus.RUNNING
    assert len(scheduler.running) == 2
    assert len(scheduler.waiting) == 0

    # Step 2: Should schedule Decode
    out2 = scheduler.schedule()
    assert not out2.is_prefill
    assert len(out2.scheduled_seqs) == 2
    assert "seq-1" in out2.blocks_to_append
    assert "seq-2" in out2.blocks_to_append

    # Complete seq1
    seq1.status = SequenceStatus.FINISHED
    scheduler.free_sequence(seq1)
    assert len(scheduler.running) == 1

    # Step 3: Only seq2 scheduled
    out3 = scheduler.schedule()
    assert not out3.is_prefill
    assert len(out3.scheduled_seqs) == 1
    assert out3.scheduled_seqs[0].seq_id == "seq-2"


def test_scheduler_preemption_under_memory_pressure():
    # Only 2 blocks total!
    block_manager = BlockSpaceManager(num_blocks=2, block_size=16)
    scheduler = Scheduler(block_manager, max_batch_size=2)

    # seq1 takes 1 block (16 tokens)
    seq1 = Sequence(seq_id="seq-1", prompt="a", prompt_token_ids=list(range(16)))
    # seq2 takes 1 block (16 tokens with distinct IDs so no prefix sharing)
    seq2 = Sequence(seq_id="seq-2", prompt="b", prompt_token_ids=list(range(16, 32)))

    scheduler.add_sequence(seq1)
    scheduler.add_sequence(seq2)

    # Prefill: both admitted (using all 2 blocks)
    out1 = scheduler.schedule()
    assert len(out1.scheduled_seqs) == 2
    assert block_manager.num_free_blocks == 0

    # Next step is Decode. Both seq1 and seq2 have full blocks (16 tokens),
    # so they need new blocks to decode!
    # With 0 free blocks, one sequence must be preempted!
    out2 = scheduler.schedule()
    assert len(out2.preempted_seq_ids) == 1
    preempted_id = out2.preempted_seq_ids[0]
    assert preempted_id in ["seq-1", "seq-2"]
    # The preempted sequence is back in waiting queue
    assert len(scheduler.waiting) == 1
    assert scheduler.waiting[0].status == SequenceStatus.PREEMPTED
