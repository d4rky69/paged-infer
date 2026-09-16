"""Sequence and Sequence Group abstractions for managing inference lifecycles."""

from enum import Enum
import time
from typing import List, Optional, Set


class SequenceStatus(str, Enum):
    """Execution status of a sequence in the inference scheduler."""
    WAITING = "WAITING"        # Enqueued, waiting for prefill allocation
    RUNNING = "RUNNING"        # Currently in decode phase
    PREEMPTED = "PREEMPTED"    # Evicted back to waiting queue due to memory pressure
    FINISHED = "FINISHED"      # Completed generation


class SamplingParams:
    """Hyperparameters governing autoregressive token sampling."""

    def __init__(
        self,
        max_tokens: int = 64,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        stop_token_ids: Optional[Set[int]] = None,
        ignore_eos: bool = False,
    ):
        self.max_tokens: int = max_tokens
        self.temperature: float = max(temperature, 1e-5)
        self.top_p: float = max(min(top_p, 1.0), 0.0)
        self.top_k: int = top_k
        self.stop_token_ids: Set[int] = stop_token_ids or set()
        self.ignore_eos: bool = ignore_eos


class Sequence:
    """Represents a single generation stream within the serving engine."""

    def __init__(
        self,
        seq_id: str,
        prompt: str,
        prompt_token_ids: List[int],
        sampling_params: Optional[SamplingParams] = None,
    ):
        self.seq_id: str = seq_id
        self.prompt: str = prompt
        self.prompt_token_ids: List[int] = list(prompt_token_ids)
        self.output_token_ids: List[int] = []
        self.status: SequenceStatus = SequenceStatus.WAITING
        self.sampling_params: SamplingParams = sampling_params or SamplingParams()

        # Telemetry timestamps
        self.arrival_time: float = time.time()
        self.first_token_time: Optional[float] = None
        self.finish_time: Optional[float] = None

        # Reason for finishing
        self.finish_reason: Optional[str] = None

    @property
    def num_prompt_tokens(self) -> int:
        return len(self.prompt_token_ids)

    @property
    def num_output_tokens(self) -> int:
        return len(self.output_token_ids)

    @property
    def total_tokens(self) -> int:
        return self.num_prompt_tokens + self.num_output_tokens

    def get_all_token_ids(self) -> List[int]:
        return self.prompt_token_ids + self.output_token_ids

    def get_last_token_id(self) -> int:
        if self.output_token_ids:
            return self.output_token_ids[-1]
        return self.prompt_token_ids[-1]

    def append_token_id(self, token_id: int) -> None:
        """Appends a newly sampled token and checks completion criteria."""
        now = time.time()
        if self.first_token_time is None:
            self.first_token_time = now

        self.output_token_ids.append(token_id)

        # Check stopping criteria
        if not self.sampling_params.ignore_eos and token_id in self.sampling_params.stop_token_ids:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "stop"
            self.finish_time = now
        elif len(self.output_token_ids) >= self.sampling_params.max_tokens:
            self.status = SequenceStatus.FINISHED
            self.finish_reason = "length"
            self.finish_time = now

    @property
    def is_finished(self) -> bool:
        return self.status == SequenceStatus.FINISHED

    @property
    def time_to_first_token(self) -> Optional[float]:
        """Calculates TTFT (latency from arrival to first generated token)."""
        if self.first_token_time is not None:
            return self.first_token_time - self.arrival_time
        return None

    @property
    def end_to_end_latency(self) -> Optional[float]:
        """Total execution latency in seconds."""
        if self.finish_time is not None:
            return self.finish_time - self.arrival_time
        return None

    def __repr__(self) -> str:
        return (
            f"Sequence(id={self.seq_id}, status={self.status.value}, "
            f"prompt_toks={self.num_prompt_tokens}, out_toks={self.num_output_tokens})"
        )
