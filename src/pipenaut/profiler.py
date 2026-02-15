"""
Lightweight profiler for pipeline stages.

Tracks compute, communication, and idle (bubble) time per step
to calculate pipeline utilization metrics.
"""

import time
from collections import defaultdict


class PipelineProfiler:
    """
    Records timing events during pipeline execution.

    Usage:
        profiler = PipelineProfiler()
        profiler.start("compute_fwd")
        ... do work ...
        profiler.stop("compute_fwd")

        stats = profiler.get_stats()
    """

    def __init__(self):
        self._timings: dict[str, list[float]] = defaultdict(list)
        self._active: dict[str, float] = {}
        self.step_start: float | None = None
        self.step_times: list[float] = []

    def start(self, name: str) -> None:
        """Mark the start of a named event."""
        self._active[name] = time.perf_counter()

    def stop(self, name: str) -> None:
        """Mark the end of a named event and record its duration."""
        if name in self._active:
            elapsed = time.perf_counter() - self._active.pop(name)
            self._timings[name].append(elapsed)

    def start_step(self) -> None:
        """Mark the start of one full training step."""
        self.step_start = time.perf_counter()

    def stop_step(self) -> None:
        """Mark the end of one full training step."""
        if self.step_start is not None:
            self.step_times.append(time.perf_counter() - self.step_start)
            self.step_start = None

    def get_stats(self) -> dict:
        """
        Compute aggregated statistics.

        Returns:
            dict with keys: compute_time, comm_time, total_time, bubble_time,
                           bubble_pct, compute_pct, comm_pct
        """
        compute_time = sum(
            sum(v) for k, v in self._timings.items() if "compute" in k
        )
        comm_time = sum(
            sum(v) for k, v in self._timings.items()
            if "send" in k or "recv" in k
        )
        total_time = sum(self.step_times) if self.step_times else (compute_time + comm_time)

        bubble_time = max(0, total_time - compute_time - comm_time)

        return {
            "compute_time": compute_time,
            "comm_time": comm_time,
            "bubble_time": bubble_time,
            "total_time": total_time,
            "compute_pct": (compute_time / total_time * 100) if total_time > 0 else 0,
            "comm_pct": (comm_time / total_time * 100) if total_time > 0 else 0,
            "bubble_pct": (bubble_time / total_time * 100) if total_time > 0 else 0,
            "num_steps": len(self.step_times),
        }

    def reset(self) -> None:
        """Clear all recorded timings."""
        self._timings.clear()
        self._active.clear()
        self.step_times.clear()
        self.step_start = None
