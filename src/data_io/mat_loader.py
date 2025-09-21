"""MAT-file loading utilities for the bearing diagnosis project."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import logging
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional

import numpy as np
from scipy.io import loadmat


LOGGER = logging.getLogger(__name__)


@dataclass
class LabelInfo:
    """Metadata parsed from Case Western style file names."""

    code: Optional[str] = None
    label: Optional[str] = None
    fault_size_inch: Optional[float] = None
    fault_size_mm: Optional[float] = None
    load_hp: Optional[int] = None


@dataclass
class SignalRecord:
    """Single vibration signal channel loaded from a MAT file."""

    file_path: Path
    file_id: str
    channel: str
    signal: np.ndarray
    sampling_rate: float
    rpm: Optional[float] = None
    label: Optional[str] = None
    label_code: Optional[str] = None
    fault_size_inch: Optional[float] = None
    fault_size_mm: Optional[float] = None
    load_hp: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FileSummary:
    """Container bundling together all channels for a single MAT file."""

    file_path: Path
    file_id: str
    sampling_rate: float
    rpm: Optional[float]
    label_info: LabelInfo
    records: List[SignalRecord]
    metadata: Dict[str, Any] = field(default_factory=dict)


CHANNEL_HINTS: Mapping[str, tuple[str, ...]] = {
    "DE": ("DE", "DRIVE"),
    "FE": ("FE", "FAN"),
    "BA": ("BA", "BASE"),
}

LABEL_MAP = {
    "IR": "inner_race_fault",
    "OR": "outer_race_fault",
    "B": "ball_fault",
    "N": "normal",
    "NORMAL": "normal",
}

SIZE_PATTERN = re.compile(r"(IR|OR|B)(\d{3})", re.IGNORECASE)
LOAD_PATTERN = re.compile(r"_(?P<load>[0-3])(?!\d)|(?P<hp>[0-3])HP", re.IGNORECASE)
LABEL_PATTERN = re.compile(r"\b(IR|OR|B|NORMAL|N)\b", re.IGNORECASE)


def _clean_value(value: Any) -> Any:
    """Recursively convert MATLAB structures into Python primitives."""

    if isinstance(value, np.ndarray):
        if value.dtype == object:
            if value.size == 1:
                return _clean_value(value.item())
            return np.array([_clean_value(v) for v in value.flat]).reshape(value.shape)
        return np.asarray(value)
    if hasattr(value, "_fieldnames"):
        return {name: _clean_value(getattr(value, name)) for name in value._fieldnames}
    return value


def load_mat_variables(path: Path) -> Dict[str, Any]:
    raw = loadmat(path, squeeze_me=True, struct_as_record=False)
    return {key: _clean_value(value) for key, value in raw.items() if not key.startswith("__")}


def _flatten_signal(value: Any) -> Optional[np.ndarray]:
    if isinstance(value, np.ndarray):
        array = np.asarray(value, dtype=float).squeeze()
        if array.ndim == 0:
            return None
        return array.ravel()
    return None


def extract_signal_channels(variables: Mapping[str, Any]) -> Dict[str, np.ndarray]:
    channels: Dict[str, np.ndarray] = {}
    for key, value in variables.items():
        signal_array = _flatten_signal(value)
        if signal_array is None:
            continue
        key_upper = key.upper()
        matched = False
        for channel, hints in CHANNEL_HINTS.items():
            if any(hint in key_upper for hint in hints):
                channels[channel] = signal_array
                matched = True
                break
        if not matched and key_upper.isalpha():
            # Target domain files store the signal in a variable named after the
            # file (single letter).  We keep only one entry.
            channels.setdefault("SENSOR", signal_array)
    return channels


def extract_rpm(variables: Mapping[str, Any]) -> Optional[float]:
    for key, value in variables.items():
        if "RPM" in key.upper():
            array = _flatten_signal(value)
            if array is not None and array.size:
                return float(np.mean(array))
    return None


def infer_sampling_rate(variables: Mapping[str, Any], path: Path, default: float) -> float:
    # Attempt to infer from a monotonic time vector if present.
    for key, value in variables.items():
        array = _flatten_signal(value)
        if array is None or array.size < 2:
            continue
        key_lower = key.lower()
        if "time" in key_lower and np.all(np.diff(array) > 0):
            diffs = np.diff(array)
            if np.mean(diffs) > 0:
                return float(1.0 / np.mean(diffs))
    text = str(path).lower()
    if "48k" in text:
        return 48000.0
    if "24k" in text:
        return 24000.0
    if "12k" in text:
        return 12000.0
    return float(default)


def infer_label_from_path(path: Path) -> LabelInfo:
    text = str(path)
    upper = text.upper()
    code = None
    size = None

    size_match = SIZE_PATTERN.search(upper)
    if size_match:
        code = size_match.group(1).upper()
        size = size_match.group(2)
    else:
        label_match = LABEL_PATTERN.search(upper)
        if label_match:
            code = label_match.group(1).upper()
            if code == "N":
                code = "NORMAL"

    label = LABEL_MAP.get(code) if code else None

    load_hp = None
    load_match = LOAD_PATTERN.search(upper)
    if load_match:
        load_str = load_match.group("load") or load_match.group("hp")
        if load_str is not None:
            load_hp = int(load_str)

    fault_size_inch = None
    fault_size_mm = None
    if size is not None:
        fault_size_inch = float(int(size)) / 1000.0
        fault_size_mm = fault_size_inch * 25.4

    return LabelInfo(code=code, label=label, fault_size_inch=fault_size_inch, fault_size_mm=fault_size_mm, load_hp=load_hp)


def load_source_file(path: Path, default_sampling_rate: float = 12000.0) -> Optional[FileSummary]:
    variables = load_mat_variables(path)
    channels = extract_signal_channels(variables)
    if not channels:
        LOGGER.warning("No vibration channels found in %s", path)
        return None

    rpm = extract_rpm(variables)
    sampling_rate = infer_sampling_rate(variables, path, default_sampling_rate)
    label_info = infer_label_from_path(path)

    records: List[SignalRecord] = []
    for channel, signal_array in channels.items():
        records.append(
            SignalRecord(
                file_path=path,
                file_id=path.stem,
                channel=channel,
                signal=signal_array,
                sampling_rate=sampling_rate,
                rpm=rpm,
                label=label_info.label,
                label_code=label_info.code,
                fault_size_inch=label_info.fault_size_inch,
                fault_size_mm=label_info.fault_size_mm,
                load_hp=label_info.load_hp,
                metadata={"channel_key": channel},
            )
        )

    return FileSummary(
        file_path=path,
        file_id=path.stem,
        sampling_rate=sampling_rate,
        rpm=rpm,
        label_info=label_info,
        records=records,
        metadata={"channel_count": len(records)},
    )


def load_source_directory(root: Path, pattern: str = "**/*.mat", default_sampling_rate: float = 12000.0) -> List[FileSummary]:
    if not root.exists():
        LOGGER.warning("Source directory %s does not exist", root)
        return []

    files: List[FileSummary] = []
    for path in sorted(root.glob(pattern)):
        summary = load_source_file(path, default_sampling_rate=default_sampling_rate)
        if summary is not None:
            files.append(summary)
    return files


def load_target_file(path: Path, sampling_rate: float, rpm: Optional[float]) -> Optional[FileSummary]:
    variables = load_mat_variables(path)
    channels = extract_signal_channels(variables)
    if not channels:
        LOGGER.warning("Target file %s did not yield a signal channel", path)
        return None

    # Target domain MAT files contain only a single channel, we keep the first.
    channel_name, signal_array = next(iter(channels.items()))
    record = SignalRecord(
        file_path=path,
        file_id=path.stem,
        channel=channel_name,
        signal=signal_array,
        sampling_rate=sampling_rate,
        rpm=rpm,
        label=None,
        label_code=None,
        metadata={"channel_key": channel_name},
    )

    return FileSummary(
        file_path=path,
        file_id=path.stem,
        sampling_rate=sampling_rate,
        rpm=rpm,
        label_info=LabelInfo(),
        records=[record],
        metadata={"channel_count": 1},
    )


def load_target_directory(root: Path, sampling_rate: float, rpm: Optional[float], pattern: str = "*.mat") -> List[FileSummary]:
    if not root.exists():
        LOGGER.warning("Target directory %s does not exist", root)
        return []

    files: List[FileSummary] = []
    for path in sorted(root.glob(pattern)):
        summary = load_target_file(path, sampling_rate=sampling_rate, rpm=rpm)
        if summary is not None:
            files.append(summary)
    return files


__all__ = [
    "LabelInfo",
    "SignalRecord",
    "FileSummary",
    "load_source_directory",
    "load_source_file",
    "load_target_directory",
    "load_target_file",
]
