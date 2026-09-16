"""Physical and Logical Memory Block representations for PagedAttention KV-Cache.

Analogous to OS Virtual Memory:
- PhysicalBlock corresponds to a physical frame in memory.
- LogicalTokenBlock corresponds to a virtual page within a sequence.
- BlockTable maps virtual pages to physical frames.
"""

from typing import List, Optional, Tuple


class PhysicalBlock:
    """Represents an allocated physical block in the KV cache memory pool.
    
    A physical block holds key and value vectors for a fixed number of tokens
    (e.g., 16 tokens). Supports reference counting for Copy-On-Write (CoW)
    prompt prefix caching.
    """

    def __init__(self, block_number: int, block_size: int = 16):
        self.block_number: int = block_number
        self.block_size: int = block_size
        self.ref_count: int = 0
        self.num_tokens: int = 0
        self.prefix_hash: Optional[int] = None

    @property
    def is_full(self) -> bool:
        """Returns True if the block holds the maximum capacity of tokens."""
        return self.num_tokens >= self.block_size

    @property
    def free_slots(self) -> int:
        """Returns the number of unfilled token slots remaining in this block."""
        return self.block_size - self.num_tokens

    def append_tokens(self, count: int) -> int:
        """Appends tokens to this physical block up to capacity.
        
        Returns the number of tokens actually accommodated.
        """
        can_add = min(count, self.free_slots)
        self.num_tokens += can_add
        return can_add

    def reset(self) -> None:
        """Resets the block state for return to the free list."""
        self.ref_count = 0
        self.num_tokens = 0
        self.prefix_hash = None

    def __repr__(self) -> str:
        return (
            f"PhysicalBlock(id={self.block_number}, "
            f"tokens={self.num_tokens}/{self.block_size}, ref_count={self.ref_count})"
        )


class LogicalTokenBlock:
    """Represents a virtual block within a single request's token sequence."""

    def __init__(self, logical_block_idx: int, block_size: int = 16):
        self.logical_block_idx: int = logical_block_idx
        self.block_size: int = block_size
        self.num_tokens: int = 0

    @property
    def is_full(self) -> bool:
        return self.num_tokens >= self.block_size

    def __repr__(self) -> str:
        return f"LogicalTokenBlock(idx={self.logical_block_idx}, tokens={self.num_tokens}/{self.block_size})"


class BlockTable:
    """Maps a sequence's logical token blocks to physical memory blocks.
    
    This acts as the per-sequence Page Table in an operating system.
    """

    def __init__(self, block_size: int = 16):
        self.block_size: int = block_size
        self.physical_block_numbers: List[int] = []

    def append_block(self, physical_block_number: int) -> None:
        """Appends a new physical block mapping to the end of the page table."""
        self.physical_block_numbers.append(physical_block_number)

    def pop_block(self) -> int:
        """Removes and returns the last physical block mapping."""
        return self.physical_block_numbers.pop()

    def get_physical_block_numbers(self) -> List[int]:
        """Returns the list of physical block numbers mapped to this sequence."""
        return list(self.physical_block_numbers)

    def locate_token(self, token_seq_index: int) -> Tuple[int, int]:
        """Locates the physical block and offset for a given sequence token index.
        
        Args:
            token_seq_index: 0-based token index in the logical sequence.
            
        Returns:
            Tuple of (physical_block_number, offset_within_block).
        """
        logical_block_idx = token_seq_index // self.block_size
        offset = token_seq_index % self.block_size
        if logical_block_idx >= len(self.physical_block_numbers):
            raise IndexError(
                f"Token index {token_seq_index} mapped to logical block {logical_block_idx}, "
                f"but page table only has {len(self.physical_block_numbers)} blocks."
            )
        physical_block_number = self.physical_block_numbers[logical_block_idx]
        return physical_block_number, offset

    def __len__(self) -> int:
        return len(self.physical_block_numbers)

    def __getitem__(self, idx: int) -> int:
        return self.physical_block_numbers[idx]

    def __repr__(self) -> str:
        return f"BlockTable(blocks={self.physical_block_numbers})"
