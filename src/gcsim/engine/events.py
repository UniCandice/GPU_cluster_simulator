"""Event queue and trace.

The queue exists for a real reason rather than as ceremony: **telemetry sampling
runs on its own clock**. Sample ticks are scheduled at a fixed 1 Hz while phase
boundaries land wherever the physics puts them, and popping a single
time-ordered queue is what interleaves the two correctly. A tick that falls in
the middle of a timestep sees a partial timestep, exactly as a real exporter
would.

The trace is the audit trail. Every row in every telemetry table is produced
during some event's handling, so `events.parquet` is what lets a reviewer walk
back from a number on the dashboard to the occurrence that caused it.
"""

from __future__ import annotations

import heapq
import itertools
import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import pandas as pd


class EventType(str, Enum):
    SIM_START = "SIM_START"
    SIM_END = "SIM_END"
    DATA_LOAD_START = "DATA_LOAD_START"
    DATA_LOAD_END = "DATA_LOAD_END"
    ITERATION_START = "ITERATION_START"
    ITERATION_END = "ITERATION_END"
    HALO_EXCHANGE_END = "HALO_EXCHANGE_END"
    COMPUTE_END = "COMPUTE_END"
    ALLREDUCE_END = "ALLREDUCE_END"
    OUTPUT_START = "OUTPUT_START"
    OUTPUT_END = "OUTPUT_END"
    SAMPLE_TICK = "SAMPLE_TICK"
    INJECTION_APPLIED = "INJECTION_APPLIED"
    STRAGGLER_DETECTED = "STRAGGLER_DETECTED"
    THROTTLE_ENGAGED = "THROTTLE_ENGAGED"
    THROTTLE_RELEASED = "THROTTLE_RELEASED"
    CONGESTION_ONSET = "CONGESTION_ONSET"
    CONGESTION_CLEARED = "CONGESTION_CLEARED"


@dataclass(order=True)
class _Queued:
    time_s: float
    sequence: int
    event: "Event" = field(compare=False)


@dataclass
class Event:
    time_s: float
    event_type: EventType
    rank_id: int | None = None
    gpu_id: str | None = None
    payload: dict[str, Any] = field(default_factory=dict)


class EventQueue:
    """A time-ordered queue. Ties break on insertion order, so it is stable."""

    def __init__(self) -> None:
        self._heap: list[_Queued] = []
        self._counter = itertools.count()

    def push(self, event: Event) -> None:
        heapq.heappush(self._heap, _Queued(event.time_s, next(self._counter), event))

    def pop(self) -> Event:
        return heapq.heappop(self._heap).event

    def peek_time(self) -> float | None:
        return self._heap[0].time_s if self._heap else None

    def __len__(self) -> int:
        return len(self._heap)

    def __bool__(self) -> bool:
        return bool(self._heap)


class EventTrace:
    """Records events for the `events` table.

    `SAMPLE_TICK` is excluded by default: it fires thousands of times and its
    occurrence is already implied by every telemetry row's timestamp. Everything
    that changes the simulation is always recorded.
    """

    SUPPRESSED = frozenset({EventType.SAMPLE_TICK})

    def __init__(self, scenario: str, seed: int, record_ticks: bool = False):
        self.scenario = scenario
        self.seed = seed
        self.record_ticks = record_ticks
        self._rows: list[dict[str, Any]] = []
        self._last_time = -1.0

    def record(self, event: Event) -> None:
        if event.event_type in self.SUPPRESSED and not self.record_ticks:
            return
        #  Monotonicity is asserted here rather than in a test so that a bug in
        #  the engine surfaces at the point it happens.
        if event.time_s < self._last_time - 1e-9:
            raise AssertionError(
                f"event trace went backwards: {event.event_type} at {event.time_s} "
                f"after {self._last_time}"
            )
        self._last_time = max(self._last_time, event.time_s)
        self._rows.append({
            "scenario": self.scenario,
            "seed": self.seed,
            "timestamp": event.time_s,
            "event_type": event.event_type.value,
            "rank_id": event.rank_id,
            "gpu_id": event.gpu_id,
            "payload": json.dumps(event.payload, sort_keys=True, default=str),
        })

    def to_frame(self) -> pd.DataFrame:
        if not self._rows:
            return pd.DataFrame(columns=["scenario", "seed", "timestamp", "event_type",
                                         "rank_id", "gpu_id", "payload"])
        return pd.DataFrame(self._rows)

    def __len__(self) -> int:
        return len(self._rows)
