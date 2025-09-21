"""Generate visualisations and statistics for extracted feature tables."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# Ensure the project root is on the import path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.feature_analysis import (
    SignalPlotConfig,
    compute_feature_statistics,
    load_feature_table,
    plot_embedding,
    plot_signal_diagnostics,
    plot_time_series_grid,
    prepare_combined_features,
    run_tsne,
    run_umap,
)
from src.data_io.mat_loader import load_source_directory, load_target_directory
from src.pipelines.build_feature_dataset import SegmentationConfig

LOGGER = logging.getLogger("feature_analysis")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_segmentation(config: Dict[str, Any]) -> SegmentationConfig:
    return SegmentationConfig(
        window_seconds=float(config.get("window_seconds", 1.0)),
        overlap=float(config.get("overlap", 0.5)),
        drop_last=bool(config.get("drop_last", True)),
    )


def _load_target_summaries(config: Dict[str, Any]):
    root = Path(config.get("root", "targetData"))
    pattern = config.get("pattern", "*.mat")
    sampling_rate = float(config.get("sampling_rate", 32000))
    rpm_value = config.get("rpm")
    rpm = float(rpm_value) if rpm_value is not None else None
    LOGGER.info("Loading target signals from %s", root)
    return load_target_directory(root, sampling_rate=sampling_rate, rpm=rpm, pattern=pattern)


def _load_source_summaries(config: Dict[str, Any]):
    root = Path(config.get("root", "sourceData"))
    pattern = config.get("pattern", "**/*.mat")
    default_sampling_rate = float(config.get("default_sampling_rate", 12000))
    LOGGER.info("Loading source signals from %s", root)
    return load_source_directory(root, pattern=pattern, default_sampling_rate=default_sampling_rate)


def analyse_features(
    config_path: Path,
    output_dir: Optional[Path] = None,
    analysis_dir: Optional[Path] = None,
    max_records: int = 4,
    preview_seconds: float = 2.0,
) -> None:
    config = _load_yaml(config_path)
    outputs = config.get("outputs", {})

    feature_root = output_dir or Path(outputs.get("directory", "artifacts"))
    feature_root.mkdir(parents=True, exist_ok=True)

    analysis_root = analysis_dir or (feature_root / "analysis")
    analysis_root.mkdir(parents=True, exist_ok=True)

    source_path = feature_root / outputs.get("source_feature_table", "source_features.csv")
    target_path = feature_root / outputs.get("target_feature_table", "target_features.csv")

    source_features = load_feature_table(source_path, dataset_name="source")
    target_features = load_feature_table(target_path, dataset_name="target")

    combined = prepare_combined_features([source_features, target_features])
    if combined.empty:
        LOGGER.warning("No feature tables available for analysis")
    else:
        stats = compute_feature_statistics(combined)
        if not stats.empty:
            stats_path = analysis_root / "feature_statistics.csv"
            LOGGER.info("Writing feature statistics to %s", stats_path)
            stats.to_csv(stats_path, index=False)
        else:
            LOGGER.warning("Feature statistics could not be computed")

        tsne_result = run_tsne(combined)
        if tsne_result:
            tsne_path = analysis_root / "tsne_embedding.png"
            plot_embedding(tsne_result, tsne_path, title="t-SNE embedding of source/target features")
            LOGGER.info("Saved t-SNE visualisation to %s", tsne_path)

        umap_result = run_umap(combined)
        if umap_result:
            umap_path = analysis_root / "umap_embedding.png"
            plot_embedding(umap_result, umap_path, title="UMAP embedding of source/target features")
            LOGGER.info("Saved UMAP visualisation to %s", umap_path)

    signal_config = SignalPlotConfig(preview_seconds=preview_seconds)

    target_config = config.get("target")
    if target_config:
        segmentation = _parse_segmentation(target_config.get("segmentation", {}))
        signal_config.window_seconds = segmentation.window_seconds
        signal_config.overlap = segmentation.overlap
        signal_config.drop_last = segmentation.drop_last

        target_summaries = _load_target_summaries(target_config)
        if target_summaries:
            grid_path = analysis_root / "target_time_series_overview.png"
            plot_time_series_grid(target_summaries, grid_path, signal_config, max_records=max_records)
            LOGGER.info("Saved target time-series overview to %s", grid_path)

            first_summary = next((summary for summary in target_summaries if summary.records), None)
            if first_summary is not None:
                first_record = first_summary.records[0]
                diag_path = analysis_root / f"target_{first_summary.file_id}_diagnostics.png"
                plot_signal_diagnostics(first_summary, first_record, diag_path, signal_config)
                LOGGER.info("Saved target diagnostic plot to %s", diag_path)
        else:
            LOGGER.warning("No target signals found for time-series visualisation")

    source_config = config.get("source")
    if source_config:
        segmentation = _parse_segmentation(source_config.get("segmentation", {}))
        signal_config.window_seconds = segmentation.window_seconds
        signal_config.overlap = segmentation.overlap
        signal_config.drop_last = segmentation.drop_last

        source_summaries = _load_source_summaries(source_config)
        if source_summaries:
            grid_path = analysis_root / "source_time_series_overview.png"
            plot_time_series_grid(source_summaries, grid_path, signal_config, max_records=max_records)
            LOGGER.info("Saved source time-series overview to %s", grid_path)

            first_summary = next((summary for summary in source_summaries if summary.records), None)
            if first_summary is not None:
                first_record = first_summary.records[0]
                diag_path = analysis_root / f"source_{first_summary.file_id}_diagnostics.png"
                plot_signal_diagnostics(first_summary, first_record, diag_path, signal_config)
                LOGGER.info("Saved source diagnostic plot to %s", diag_path)
        else:
            LOGGER.warning("No source signals found for time-series visualisation")



def main() -> None:
    parser = argparse.ArgumentParser(description="Create analysis artefacts for extracted bearing features.")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Override the directory that stores feature CSV files.")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="Optional directory in which to store visualisations and reports.",
    )
    parser.add_argument("--max-records", type=int, default=4, help="How many signals to include in grid visualisations.")
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=2.0,
        help="Duration of each signal preview (seconds) in the time-domain plots.",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    analyse_features(
        args.config,
        output_dir=args.output_dir,
        analysis_dir=args.analysis_dir,
        max_records=args.max_records,
        preview_seconds=args.preview_seconds,
    )


if __name__ == "__main__":
    main()
