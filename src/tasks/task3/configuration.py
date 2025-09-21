"""Configuration parsing helpers for task 3 pipelines."""
from __future__ import annotations

from typing import Any, Dict, Sequence

from ...modeling import SourceDiagnosisConfig
from ..task2 import resolve_task2_config
from .features import TimeFrequencyConfig
from .transfer import PseudoLabelConfig, TransferConfig


def parse_time_frequency(config: Dict[str, Any]) -> TimeFrequencyConfig:
    stft_cfg = config.get("stft", {})
    cwt_cfg = config.get("cwt", {})
    return TimeFrequencyConfig(
        stft_nperseg=int(stft_cfg.get("nperseg", 512)),
        stft_noverlap=int(stft_cfg.get("noverlap", 256)),
        stft_window=str(stft_cfg.get("window", "hann")),
        stft_nfft=int(stft_cfg.get("nfft", 1024)),
        cwt_wavelet=str(cwt_cfg.get("wavelet", "ricker")),
        cwt_min_scale=float(cwt_cfg.get("min_scale", 1.0)),
        cwt_max_scale=float(cwt_cfg.get("max_scale", 128.0)),
        cwt_num_scales=int(cwt_cfg.get("num_scales", 64)),
    )


def parse_pseudo_label(config: Dict[str, Any]) -> PseudoLabelConfig:
    return PseudoLabelConfig(
        enabled=bool(config.get("enabled", True)),
        confidence_threshold=float(config.get("confidence_threshold", 0.95)),
        max_iterations=int(config.get("max_iterations", 2)),
        max_ratio=float(config.get("max_ratio", 0.4)),
    )


def parse_transfer_config(raw: Dict[str, Any]) -> TransferConfig:
    modeling_cfg = resolve_task2_config(raw.get("modeling", {}))
    features_cfg = modeling_cfg.features_config
    label_column = str(features_cfg.get("label_column", "label"))
    feature_columns = features_cfg.get("feature_columns")
    if feature_columns is not None:
        feature_columns = [str(column) for column in feature_columns]
    exclude_columns = features_cfg.get("exclude_columns")
    if exclude_columns is not None:
        exclude_columns = [str(column) for column in exclude_columns]

    diagnosis = SourceDiagnosisConfig(
        label_column=label_column,
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
        alignment=modeling_cfg.alignment,
        split=modeling_cfg.split,
        model=modeling_cfg.model,
        cross_validation=modeling_cfg.cross_validation,
        permutation_importance=modeling_cfg.permutation,
    )

    time_frequency = parse_time_frequency(raw.get("time_frequency", {}))
    pseudo_label = parse_pseudo_label(raw.get("pseudo_label", {}))
    metadata_columns = raw.get("features", {}).get(
        "metadata_columns",
        ["file_id", "channel", "segment_index", "start_sample", "end_sample", "rpm"],
    )
    metadata_columns = [str(column) for column in metadata_columns]

    return TransferConfig(
        diagnosis=diagnosis,
        time_frequency=time_frequency,
        pseudo_label=pseudo_label,
        metadata_columns=metadata_columns,
    )


__all__ = ["parse_transfer_config", "parse_time_frequency", "parse_pseudo_label"]
