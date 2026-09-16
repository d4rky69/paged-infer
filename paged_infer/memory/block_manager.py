"""Block Space Manager (Memory Allocator) for PagedAttention.

Implements OS-inspired physical block allocation, reference-counted prefix caching
(Copy-on-Write), and fragmentation accounting.
"""

from collections import deque
from typing import Any, Dict, List, Optional, Set, Tuple

from paged_infer.memory.block import BlockTable, PhysicalBlock


class BlockSpaceManager:
    """Manages the physical block memory pool for KV cache storage.
    
    Acts as the OS Memory Management Unit (MMU) and page frame allocator.
    Guarantees zero external memory fragmentation by allocating fixed-size blocks.
    Supports prefix caching with Copy-On-Write (CoW) semantics.
    """

    def __init__(self, num_blocks: int, block_size: int = 16, enable_prefix_caching: bool = True):
        """
        Args:
            num_blocks: Total number of physical blocks in the pool.
            block_size: Number of token slots per block.
            enable_prefix_caching: Whether to enable prompt prefix deduplication.
        """
        self.num_blocks: int = num_blocks
        self.block_size: int = block_size
        self.enable_prefix_caching: bool = enable_prefix_caching

        # Pool of all physical blocks indexed by physical block number
        self.all_blocks: Dict[int, PhysicalBlock] = {
            i: PhysicalBlock(block_number=i, block_size=block_size)
            for i in range(num_blocks)
        }

        # Free list allocator: O(1) allocation and deallocation
        self.free_blocks: deque[int] = deque(range(num_blocks))

        # Sequence ID -> BlockTable mapping (Page tables)
        self.block_tables: Dict[str, BlockTable] = {}

        # Prefix hash -> physical block number for prefix caching
        self.prefix_hash_to_block: Dict[int, int] = {}

        # Telemetry
        self.prefix_cache_hits: int = 0
        self.total_allocations: int = 0
        self.cow_events: int = 0

    @property
    def num_free_blocks(self) -> int:
        return len(self.free_blocks)

    @property
    def num_used_blocks(self) -> int:
        return self.num_blocks - len(self.free_blocks)

    def can_allocate(self, num_tokens: int) -> bool:
        """Determines if there are enough free blocks to allocate for a sequence."""
        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        return self.num_free_blocks >= needed_blocks

    def allocate(
        self,
        seq_id: str,
        num_tokens: int,
        token_ids: Optional[List[int]] = None,
    ) -> BlockTable:
        """Allocates physical blocks for a new sequence (Prefill phase).
        
        If prefix caching is enabled and matching prefix hashes exist,
        re-uses existing blocks and increments their reference count.
        
        Args:
            seq_id: Unique identifier for the sequence.
            num_tokens: Total tokens to allocate for.
            token_ids: Optional token ID list used for prefix cache hash computation.
            
        Returns:
            The populated BlockTable for the sequence.
        """
        if seq_id in self.block_tables:
            raise ValueError(f"Sequence {seq_id} already has an active BlockTable.")

        needed_blocks = (num_tokens + self.block_size - 1) // self.block_size
        if self.num_free_blocks < needed_blocks:
            raise MemoryError(
                f"Out of Memory: required {needed_blocks} blocks, but only {self.num_free_blocks} available."
            )

        block_table = BlockTable(block_size=self.block_size)
        tokens_remaining = num_tokens

        for block_idx in range(needed_blocks):
            tokens_in_this_block = min(tokens_remaining, self.block_size)
            is_full_block = (tokens_in_this_block == self.block_size)

            physical_block_num: Optional[int] = None

            # Attempt prefix cache match if tokens provided and block is full
            if self.enable_prefix_caching and token_ids is not None and is_full_block:
                start_tok = block_idx * self.block_size
                end_tok = start_tok + self.block_size
                chunk = tuple(token_ids[start_tok:end_tok])
                chunk_hash = hash(chunk)

                if chunk_hash in self.prefix_hash_to_block:
                    cached_block_num = self.prefix_hash_to_block[chunk_hash]
                    cached_block = self.all_blocks[cached_block_num]
                    # Verify validity
                    if cached_block.prefix_hash == chunk_hash:
                        cached_block.ref_count += 1
                        physical_block_num = cached_block_num
                        self.prefix_cache_hits += 1

            if physical_block_num is None:
                # Allocate a fresh physical block
                physical_block_num = self.free_blocks.popleft()
                block = self.all_blocks[physical_block_num]
                block.ref_count = 1
                block.num_tokens = tokens_in_this_block

                # Register prefix hash if full
                if self.enable_prefix_caching and token_ids is not None and is_full_block:
                    start_tok = block_idx * self.block_size
                    end_tok = start_tok + self.block_size
                    chunk_hash = hash(tuple(token_ids[start_tok:end_tok]))
                    block.prefix_hash = chunk_hash
                    self.prefix_hash_to_block[chunk_hash] = physical_block_num

            block_table.append_block(physical_block_num)
            tokens_remaining -= tokens_in_this_block
            self.total_allocations += 1

        self.block_tables[seq_id] = block_table
        return block_table

    def can_append_slot(self, seq_id: str) -> bool:
        """Checks if a sequence can append one more token in the decode phase."""
        if seq_id not in self.block_tables:
            return False

        block_table = self.block_tables[seq_id]
        if len(block_table) == 0:
            return self.num_free_blocks >= 1

        last_block_num = block_table[-1]
        last_block = self.all_blocks[last_block_num]

        if not last_block.is_full:
            # Fits in current block without allocating new block
            return True
        return self.num_free_blocks >= 1

    def append_slot(self, seq_id: str) -> Tuple[int, int]:
        """Appends one token slot for an ongoing autoregressive decoding step.
        
        Handles:
        1. Appending to the current block if space is available.
        2. Copy-On-Write (CoW) if the block is shared (ref_count > 1).
        3. Allocating a new physical block when the current block is full.
        
        Returns:
            Tuple of (physical_block_number, offset_in_block).
        """
        if seq_id not in self.block_tables:
            raise KeyError(f"Sequence {seq_id} not registered.")

        block_table = self.block_tables[seq_id]

        if len(block_table) == 0:
            if self.num_free_blocks == 0:
                raise MemoryError("KV Cache exhausted during sequence decode.")
            new_block_num = self.free_blocks.popleft()
            block = self.all_blocks[new_block_num]
            block.ref_count = 1
            block.num_tokens = 1
            block_table.append_block(new_block_num)
            return new_block_num, 0

        last_block_num = block_table[-1]
        last_block = self.all_blocks[last_block_num]

        # Case A: Current block has room
        if not last_block.is_full:
            # Check for Copy-On-Write if shared
            if last_block.ref_count > 1:
                # Trigger Copy-on-Write: Allocate new unshared block
                if self.num_free_blocks == 0:
                    raise MemoryError("OOM during Copy-On-Write block duplication.")
                new_block_num = self.free_blocks.popleft()
                new_block = self.all_blocks[new_block_num]
                new_block.ref_count = 1
                new_block.num_tokens = last_block.num_tokens + 1
                
                # Replace the shared block in this sequence's table
                block_table.pop_block()
                block_table.append_block(new_block_num)
                
                # Decrement reference on old shared block
                last_block.ref_count -= 1
                self.cow_events += 1
                return new_block_num, new_block.num_tokens - 1
            else:
                offset = last_block.num_tokens
                last_block.num_tokens += 1
                return last_block_num, offset

        # Case B: Current block is full -> allocate new physical block
        if self.num_free_blocks == 0:
            raise MemoryError(f"OOM: Cannot allocate new block for sequence {seq_id}.")

        new_block_num = self.free_blocks.popleft()
        new_block = self.all_blocks[new_block_num]
        new_block.ref_count = 1
        new_block.num_tokens = 1
        block_table.append_block(new_block_num)
        return new_block_num, 0

    def free(self, seq_id: str) -> None:
        """Releases all physical memory blocks owned by a sequence.
        
        Decrements ref_count on each block; only returns blocks with
        ref_count == 0 to the free list.
        """
        if seq_id not in self.block_tables:
            return

        block_table = self.block_tables[seq_id]
        for block_num in block_table.get_physical_block_numbers():
            block = self.all_blocks[block_num]
            block.ref_count -= 1
            if block.ref_count <= 0:
                # Remove from prefix cache if present
                if block.prefix_hash is not None and block.prefix_hash in self.prefix_hash_to_block:
                    if self.prefix_hash_to_block[block.prefix_hash] == block_num:
                        del self.prefix_hash_to_block[block.prefix_hash]
                block.reset()
                self.free_blocks.append(block_num)

        del self.block_tables[seq_id]

    def get_block_table(self, seq_id: str) -> List[int]:
        """Returns physical block IDs mapped to a sequence."""
        if seq_id not in self.block_tables:
            return []
        return self.block_tables[seq_id].get_physical_block_numbers()

    def get_stats(self) -> Dict[str, Any]:
        """Calculates memory pool telemetry, utilization, and fragmentation."""
        used = self.num_used_blocks
        total = self.num_blocks
        utilization = (used / total) if total > 0 else 0.0

        # Internal fragmentation: unused token slots inside allocated blocks
        internal_frag_tokens = 0
        allocated_slots = 0
        for block_num, block in self.all_blocks.items():
            if block.ref_count > 0:
                allocated_slots += self.block_size
                internal_frag_tokens += (self.block_size - block.num_tokens)

        internal_frag_pct = (internal_frag_tokens / allocated_slots * 100) if allocated_slots > 0 else 0.0

        return {
            "total_blocks": total,
            "used_blocks": used,
            "free_blocks": self.num_free_blocks,
            "block_size": self.block_size,
            "total_token_capacity": total * self.block_size,
            "utilization_pct": round(utilization * 100, 2),
            "internal_frag_tokens": internal_frag_tokens,
            "internal_frag_pct": round(internal_frag_pct, 2),
            "external_frag_pct": 0.0,  # Zero by design with paging!
            "prefix_cache_hits": self.prefix_cache_hits,
            "cow_events": self.cow_events,
            "active_sequences": len(self.block_tables),
        }
