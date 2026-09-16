"""Unit tests for PagedInfer physical memory management and block table mapping."""

import pytest
from paged_infer.memory.block import BlockTable, PhysicalBlock
from paged_infer.memory.block_manager import BlockSpaceManager


def test_physical_block_lifecycle():
    block = PhysicalBlock(block_number=0, block_size=16)
    assert block.num_tokens == 0
    assert not block.is_full
    assert block.free_slots == 16

    added = block.append_tokens(10)
    assert added == 10
    assert block.num_tokens == 10
    assert not block.is_full
    assert block.free_slots == 6

    added2 = block.append_tokens(10)
    assert added2 == 6
    assert block.num_tokens == 16
    assert block.is_full
    assert block.free_slots == 0


def test_block_table_token_location():
    table = BlockTable(block_size=16)
    table.append_block(105)
    table.append_block(207)

    # Token 0 -> block 105, offset 0
    b, off = table.locate_token(0)
    assert b == 105 and off == 0

    # Token 15 -> block 105, offset 15
    b, off = table.locate_token(15)
    assert b == 105 and off == 15

    # Token 16 -> block 207, offset 0
    b, off = table.locate_token(16)
    assert b == 207 and off == 0

    # Token 31 -> block 207, offset 15
    b, off = table.locate_token(31)
    assert b == 207 and off == 15

    with pytest.raises(IndexError):
        table.locate_token(32)


def test_block_space_manager_allocation_and_free():
    # 4 blocks of 16 tokens = 64 tokens capacity
    manager = BlockSpaceManager(num_blocks=4, block_size=16)
    assert manager.num_free_blocks == 4
    assert manager.num_used_blocks == 0

    # Allocate sequence with 20 tokens -> requires 2 blocks
    table1 = manager.allocate(seq_id="seq-1", num_tokens=20)
    assert len(table1) == 2
    assert manager.num_used_blocks == 2
    assert manager.num_free_blocks == 2

    # Allocate sequence with 32 tokens -> requires 2 blocks
    table2 = manager.allocate(seq_id="seq-2", num_tokens=32)
    assert len(table2) == 2
    assert manager.num_used_blocks == 4
    assert manager.num_free_blocks == 0

    # Next allocation should fail with OOM
    assert not manager.can_allocate(1)
    with pytest.raises(MemoryError):
        manager.allocate(seq_id="seq-3", num_tokens=1)

    # Free seq-1 -> returns 2 blocks
    manager.free("seq-1")
    assert manager.num_used_blocks == 2
    assert manager.num_free_blocks == 2

    # Free seq-2 -> all returned
    manager.free("seq-2")
    assert manager.num_used_blocks == 0
    assert manager.num_free_blocks == 4


def test_prefix_caching_deduplication():
    manager = BlockSpaceManager(num_blocks=4, block_size=16, enable_prefix_caching=True)
    
    # Common system prompt prefix (32 tokens: exactly 2 blocks of 16)
    common_prefix = list(range(100, 132))
    
    # Sequence 1 allocates the 2 blocks
    t1 = manager.allocate("seq-1", num_tokens=32, token_ids=common_prefix)
    assert len(t1) == 2
    assert manager.num_used_blocks == 2
    
    # Sequence 2 shares identical prefix
    t2 = manager.allocate("seq-2", num_tokens=32, token_ids=common_prefix)
    assert len(t2) == 2
    # Memory pool utilized blocks should remain 2 due to prefix sharing!
    assert manager.num_used_blocks == 2
    assert manager.prefix_cache_hits == 2
    assert t1.get_physical_block_numbers() == t2.get_physical_block_numbers()


def test_copy_on_write_trigger():
    manager = BlockSpaceManager(num_blocks=5, block_size=16)
    
    # Allocate seq-1 with 8 tokens (half a block)
    t1 = manager.allocate("seq-1", num_tokens=8)
    block_num = t1[0]
    
    # Simulate a fork where another sequence shares this block
    manager.all_blocks[block_num].ref_count = 2
    
    initial_cow = manager.cow_events
    # When seq-1 appends a token into this shared block, it must trigger Copy-on-Write
    new_block_num, offset = manager.append_slot("seq-1")
    assert manager.cow_events == initial_cow + 1
    assert new_block_num != block_num
    assert offset == 8  # 8 existing tokens copied + 1 appended at offset 8


def test_fragmentation_metrics():
    manager = BlockSpaceManager(num_blocks=10, block_size=16)
    # Allocate 18 tokens (1 full block of 16, 1 partial block of 2)
    # Total slots allocated = 32, internal frag = 14 tokens
    manager.allocate("seq-1", num_tokens=18)
    
    stats = manager.get_stats()
    assert stats["used_blocks"] == 2
    assert stats["free_blocks"] == 8
    assert stats["internal_frag_tokens"] == 14
    assert stats["external_frag_pct"] == 0.0
