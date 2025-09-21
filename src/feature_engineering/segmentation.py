"""Signal segmentation utilities."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Sequence

import numpy as np


@dataclass
class Segment:
    """Container for a single signal segment."""

    index: int
    start: int
    end: int
    data: np.ndarray

    @property
    def length(self) -> int:
        return int(self.end - self.start)

    @property
    def duration(self) -> int:
        return self.length


def sliding_window(
    signal: Sequence[float] | np.ndarray,
    window_size: int,
    step_size: int,
    drop_last: bool = True,
) -> Iterator[Segment]:
    """Iterate over ``signal`` with a fixed window and step size."""

    if window_size <= 0:
        raise ValueError("window_size must be positive")
    if step_size <= 0:
        raise ValueError("step_size must be positive")

    array = np.asarray(signal)
    total = array.shape[0]
    index = 0
    start = 0

    while start + window_size <= total:
        end = start + window_size
        segment = array[start:end]
        yield Segment(index=index, start=start, end=end, data=segment)
        index += 1
        start += step_size

    if not drop_last and start < total:
        end = total
        pad = window_size - (end - start)
        if pad > 0:
            padded = np.zeros(window_size, dtype=array.dtype)
            padded[: end - start] = array[start:end]
            segment_data = padded
        else:
            segment_data = array[start:end]
        yield Segment(index=index, start=start, end=end, data=segment_data)


def seconds_to_samples(duration_seconds: float, sampling_rate: float) -> int:
    """Convert a duration in seconds into a sample count."""

    return int(round(float(duration_seconds) * float(sampling_rate)))


def segment_signal(
    signal: Sequence[float] | np.ndarray,
    sampling_rate: float,
    window_seconds: float,
    overlap: float = 0.0,
    drop_last: bool = True,
) -> Iterable[Segment]:
    """Return fixed length segments for downstream feature extraction."""

    if not 0 <= overlap < 1:
        raise ValueError("overlap must be in [0, 1)")
    window_size = max(1, seconds_to_samples(window_seconds, sampling_rate))
    step_size = max(1, int(round(window_size * (1.0 - overlap))))
    return sliding_window(signal, window_size=window_size, step_size=step_size, drop_last=drop_last)


__all__ = ["Segment", "sliding_window", "segment_signal", "seconds_to_samples"]
