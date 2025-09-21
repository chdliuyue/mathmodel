"""Utilities to recover raw signal segments referenced in feature tables."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import logging

import numpy as np
from ...data_io.mat_loader import extract_signal_channels, load_mat_variables

LOGGER = logging.getLogger(__name__)


@dataclass
class SegmentFetcher:
    """Lazy loader that caches vibration signals for repeated segment extraction."""

    cache: Dict[Path, Dict[str, np.ndarray]] = field(default_factory=dict)

    def _load_channels(self, path: Path) -> Dict[str, np.ndarray]:
        cached = self.cache.get(path)
        if cached is not None:
            return cached
        if not path.exists():
            LOGGER.warning("Signal file %s not found when extracting segments", path)
            self.cache[path] = {}
            return {}
        variables = load_mat_variables(path)
        channels = extract_signal_channels(variables)
        normalised = {key.upper(): np.asarray(signal, dtype=float).ravel() for key, signal in channels.items()}
        self.cache[path] = normalised
        return normalised

    def get_segment(self, row: Mapping[str, Any]) -> Optional[np.ndarray]:
        """Return the time-series samples corresponding to a feature table row."""

        file_path = Path(row.get("file_path", ""))
        if not file_path:
            return None
        channels = self._load_channels(file_path)
        if not channels:
            return None
        channel = str(row.get("channel", "")).upper()
        signal = channels.get(channel)
        if signal is None:
            # Fall back to the first available channel to avoid dropping samples entirely.
            signal = next(iter(channels.values()))

        start_value = row.get("start_sample", 0)
        end_value = row.get("end_sample")
        length_value = row.get("segment_length")

        try:
            start = int(start_value)
        except Exception:
            start = 0
        start = max(start, 0)

        if end_value is None or (isinstance(end_value, float) and np.isnan(end_value)):
            try:
                length = int(length_value)
            except Exception:
                length = signal.shape[0]
            end = start + length
        else:
            try:
                end = int(end_value)
            except Exception:
                end = start

        end = min(end, signal.shape[0])
        if end <= start:
            return None

        return signal[start:end]


__all__ = ["SegmentFetcher"]
