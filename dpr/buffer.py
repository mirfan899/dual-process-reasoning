"""Hippocampal working buffer for experiential trace storage."""

from dataclasses import dataclass, field
import time
from typing import List, Tuple
import numpy as np


@dataclass
class ReasoningTrace:
    trace_id: int
    prompt: str
    solution_trajectory: str
    tokens: List[int]
    target_token: int
    surprise_score: float
    is_consolidated: bool = False


class HippocampalBuffer:
    def __init__(self, capacity: int = 500):
        self.capacity = capacity
        self.traces: List[ReasoningTrace] = []
        self._counter = 1

    def add_trace(self, prompt: str, solution_trajectory: str, tokens: List[int], target_token: int, surprise_score: float) -> ReasoningTrace:
        trace = ReasoningTrace(
            trace_id=self._counter,
            prompt=prompt,
            solution_trajectory=solution_trajectory,
            tokens=tokens,
            target_token=target_token,
            surprise_score=surprise_score
        )
        self._counter += 1
        if len(self.traces) >= self.capacity:
            self.traces.pop(0)
        self.traces.append(trace)
        return trace

    def sample_unconsolidated(self) -> List[ReasoningTrace]:
        return [t for t in self.traces if not t.is_consolidated]

    def mark_consolidated(self, ids: List[int]) -> None:
        id_set = set(ids)
        for t in self.traces:
            if t.trace_id in id_set:
                t.is_consolidated = True
