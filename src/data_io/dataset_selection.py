"""Selection logic for representative source domain files."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .mat_loader import FileSummary


@dataclass
class SelectionConfig:
    """Parameters controlling how source files are ranked for similarity."""

    rpm_target: float
    sampling_rate_target: float
    top_k_per_label: Optional[int] = 5
    rpm_weight: float = 0.6
    sampling_rate_weight: float = 0.3
    load_weight: float = 0.05
    fault_size_weight: float = 0.05
    prefer_load: Optional[int] = None
    prefer_fault_sizes: Optional[Sequence[float]] = None


def _normalise_difference(value: Optional[float], target: float) -> float:
    if value is None or target == 0:
        return 1.0
    return abs(value - target) / max(abs(target), 1e-6)


def _fault_size_penalty(summary: FileSummary, preferred: Optional[Sequence[float]]) -> float:
    if not preferred:
        return 0.0
    size = summary.label_info.fault_size_inch
    if size is None:
        return 1.0
    return min(abs(size - ref) for ref in preferred)


def _load_penalty(summary: FileSummary, preferred: Optional[int]) -> float:
    if preferred is None:
        return 0.0
    load = summary.label_info.load_hp
    if load is None:
        return 1.0
    return abs(load - preferred) / max(preferred, 1)


def score_summary(summary: FileSummary, config: SelectionConfig) -> float:
    rpm_term = _normalise_difference(summary.rpm, config.rpm_target)
    sr_term = _normalise_difference(summary.sampling_rate, config.sampling_rate_target)
    load_term = _load_penalty(summary, config.prefer_load)
    size_term = _fault_size_penalty(summary, config.prefer_fault_sizes)
    return (
        config.rpm_weight * rpm_term
        + config.sampling_rate_weight * sr_term
        + config.load_weight * load_term
        + config.fault_size_weight * size_term
    )


def select_representative_files(
    summaries: Iterable[FileSummary], config: SelectionConfig
) -> List[Tuple[FileSummary, float]]:
    grouped: Dict[str, List[FileSummary]] = {}
    for summary in summaries:
        label = summary.label_info.label or summary.label_info.code or "unlabelled"
        grouped.setdefault(label, []).append(summary)

    selected: List[Tuple[FileSummary, float]] = []
    for label, group in grouped.items():
        scored = [(summary, score_summary(summary, config)) for summary in group]
        scored.sort(key=lambda item: item[1])
        limit = config.top_k_per_label
        if limit is None or limit <= 0:
            selected.extend(scored)
        else:
            selected.extend(scored[: limit])
    return selected


__all__ = ["SelectionConfig", "select_representative_files", "score_summary"]
