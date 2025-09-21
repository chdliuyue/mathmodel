"""Pipeline utilities to assemble feature tables from MAT files."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd

from ..data_io.dataset_selection import SelectionConfig, select_representative_files
from ..data_io.mat_loader import FileSummary, SignalRecord, load_source_directory, load_target_directory
from ..feature_engineering.bearing import DEFAULT_BEARINGS, BearingSpec
from ..feature_engineering.feature_extractor import FeatureExtractor, FeatureExtractorConfig
from ..feature_engineering.segmentation import Segment, segment_signal


@dataclass
class SegmentationConfig:
    window_seconds: float = 1.0
    overlap: float = 0.5
    drop_last: bool = True


def resolve_bearing(channel: str, channel_map: Mapping[str, str | BearingSpec]) -> Optional[BearingSpec]:
    if channel_map is None:
        return None
    key = channel_map.get(channel.upper())
    if key is None:
        return None
    if isinstance(key, BearingSpec):
        return key
    return DEFAULT_BEARINGS.get(key)


def _segment_record(record: SignalRecord, segmentation: SegmentationConfig) -> Iterable[Segment]:
    return segment_signal(
        record.signal,
        sampling_rate=record.sampling_rate,
        window_seconds=segmentation.window_seconds,
        overlap=segmentation.overlap,
        drop_last=segmentation.drop_last,
    )


@dataclass
class FeatureDatasetBuilder:
    dataset_name: str
    feature_extractor: FeatureExtractor
    segmentation: SegmentationConfig
    channel_bearings: Mapping[str, str | BearingSpec] = field(default_factory=dict)

    def build(self, summaries: Iterable[FileSummary], selection_scores: Optional[Mapping[str, float]] = None) -> pd.DataFrame:
        rows: List[Dict[str, float]] = []
        for summary in summaries:
            file_score = selection_scores.get(summary.file_id) if selection_scores else None
            for record in summary.records:
                bearing = resolve_bearing(record.channel, self.channel_bearings)
                rpm = record.rpm if record.rpm is not None else summary.rpm
                for segment in _segment_record(record, self.segmentation):
                    duration_seconds = segment.length / record.sampling_rate
                    features = self.feature_extractor.extract(segment.data, record.sampling_rate, rpm=rpm, bearing=bearing)
                    row: Dict[str, float] = {
                        "dataset": self.dataset_name,
                        "file_id": summary.file_id,
                        "file_path": str(summary.file_path),
                        "channel": record.channel,
                        "segment_index": segment.index,
                        "start_sample": segment.start,
                        "end_sample": segment.end,
                        "segment_length": segment.length,
                        "segment_duration": duration_seconds,
                        "sampling_rate": record.sampling_rate,
                        "rpm": rpm,
                        "label": record.label,
                        "label_code": record.label_code,
                        "load_hp": record.load_hp,
                        "fault_size_inch": record.fault_size_inch,
                        "fault_size_mm": record.fault_size_mm,
                        "selection_score": file_score,
                    }
                    row.update(features)
                    rows.append(row)
        return pd.DataFrame(rows)


def build_source_feature_table(
    root: Path,
    segmentation: SegmentationConfig,
    selection_config: SelectionConfig,
    channel_bearings: Optional[Mapping[str, str | BearingSpec]] = None,
    feature_config: Optional[FeatureExtractorConfig] = None,
    pattern: str = "**/*.mat",
    default_sampling_rate: float = 12000.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = load_source_directory(root, pattern=pattern, default_sampling_rate=default_sampling_rate)
    selection = select_representative_files(summaries, selection_config)
    selected_summaries = [summary for summary, _ in selection]
    scores = {summary.file_id: score for summary, score in selection}

    extractor = FeatureExtractor(feature_config)
    builder = FeatureDatasetBuilder(
        dataset_name="source",
        feature_extractor=extractor,
        segmentation=segmentation,
        channel_bearings=channel_bearings or {"DE": "SKF6205", "FE": "SKF6203", "BA": "SKF6205"},
    )
    feature_table = builder.build(selected_summaries, selection_scores=scores)

    metadata_rows = []
    for summary, score in selection:
        metadata_rows.append(
            {
                "file_id": summary.file_id,
                "file_path": str(summary.file_path),
                "label": summary.label_info.label,
                "label_code": summary.label_info.code,
                "rpm": summary.rpm,
                "sampling_rate": summary.sampling_rate,
                "fault_size_inch": summary.label_info.fault_size_inch,
                "fault_size_mm": summary.label_info.fault_size_mm,
                "load_hp": summary.label_info.load_hp,
                "selection_score": score,
            }
        )
    return feature_table, pd.DataFrame(metadata_rows)


def build_target_feature_table(
    root: Path,
    segmentation: SegmentationConfig,
    sampling_rate: float,
    rpm: Optional[float],
    channel_bearings: Optional[Mapping[str, str | BearingSpec]] = None,
    feature_config: Optional[FeatureExtractorConfig] = None,
    pattern: str = "*.mat",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = load_target_directory(root, sampling_rate=sampling_rate, rpm=rpm, pattern=pattern)
    extractor = FeatureExtractor(feature_config)
    builder = FeatureDatasetBuilder(
        dataset_name="target",
        feature_extractor=extractor,
        segmentation=segmentation,
        channel_bearings=channel_bearings or {"SENSOR": "SKF6205"},
    )
    feature_table = builder.build(summaries)
    metadata_rows = [
        {
            "file_id": summary.file_id,
            "file_path": str(summary.file_path),
            "rpm": summary.rpm,
            "sampling_rate": summary.sampling_rate,
            "channel_count": len(summary.records),
        }
        for summary in summaries
    ]
    return feature_table, pd.DataFrame(metadata_rows)


__all__ = [
    "SegmentationConfig",
    "FeatureDatasetBuilder",
    "build_source_feature_table",
    "build_target_feature_table",
]
