"""Command line interface to run the feature engineering pipeline."""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml

# Allow running the script directly without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.data_io.dataset_selection import SelectionConfig
from src.feature_engineering.feature_extractor import FeatureExtractorConfig
from src.pipelines.build_feature_dataset import (
    SegmentationConfig,
    build_source_feature_table,
    build_target_feature_table,
)


LOGGER = logging.getLogger("feature_pipeline")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _parse_segmentation(config: Mapping[str, Any]) -> SegmentationConfig:
    return SegmentationConfig(
        window_seconds=float(config.get("window_seconds", 1.0)),
        overlap=float(config.get("overlap", 0.5)),
        drop_last=bool(config.get("drop_last", True)),
    )


def _parse_feature_config(config: Optional[Mapping[str, Any]]) -> FeatureExtractorConfig:
    if not config:
        return FeatureExtractorConfig()
    return FeatureExtractorConfig(
        include_frequency_domain=bool(config.get("include_frequency_domain", True)),
        include_envelope_domain=bool(config.get("include_envelope_domain", True)),
        include_fault_bands=bool(config.get("include_fault_bands", True)),
        fault_bandwidth=float(config.get("fault_bandwidth", 5.0)),
    )


def _parse_selection_config(config: Mapping[str, Any]) -> SelectionConfig:
    return SelectionConfig(
        rpm_target=float(config.get("rpm_target", 600.0)),
        sampling_rate_target=float(config.get("sampling_rate_target", 32000.0)),
        top_k_per_label=int(config.get("top_k_per_label", 10)),
        rpm_weight=float(config.get("rpm_weight", 0.6)),
        sampling_rate_weight=float(config.get("sampling_rate_weight", 0.3)),
        load_weight=float(config.get("load_weight", 0.05)),
        fault_size_weight=float(config.get("fault_size_weight", 0.05)),
        prefer_load=(int(config["prefer_load"]) if config.get("prefer_load") is not None else None),
        prefer_fault_sizes=config.get("prefer_fault_sizes"),
    )


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_pipeline(config_path: Path, output_dir: Optional[Path] = None) -> None:
    config = _load_yaml(config_path)
    outputs = config.get("outputs", {})
    output_root = output_dir or Path(outputs.get("directory", "artifacts"))
    _ensure_directory(output_root)

    source_config = config.get("source")
    if source_config:
        LOGGER.info("Loading source dataset from %s", source_config.get("root"))
        segmentation = _parse_segmentation(source_config.get("segmentation", {}))
        selection = _parse_selection_config(source_config.get("selection", {}))
        feature_config = _parse_feature_config(source_config.get("feature"))
        root = Path(source_config.get("root", "sourceData"))
        channel_bearings = source_config.get("channel_bearings", {})
        pattern = source_config.get("pattern", "**/*.mat")
        default_sampling_rate = float(source_config.get("default_sampling_rate", 12000))

        source_features, source_metadata = build_source_feature_table(
            root=root,
            segmentation=segmentation,
            selection_config=selection,
            channel_bearings=channel_bearings,
            feature_config=feature_config,
            pattern=pattern,
            default_sampling_rate=default_sampling_rate,
        )
        feature_path = output_root / outputs.get("source_feature_table", "source_features.csv")
        metadata_path = output_root / outputs.get("source_metadata", "source_metadata.csv")
        if not source_features.empty:
            LOGGER.info("Writing %s", feature_path)
            source_features.to_csv(feature_path, index=False)
            LOGGER.info("Writing %s", metadata_path)
            source_metadata.to_csv(metadata_path, index=False)
        else:
            LOGGER.warning("No source features were extracted; check whether the dataset is available.")

    target_config = config.get("target")
    if target_config:
        LOGGER.info("Loading target dataset from %s", target_config.get("root"))
        segmentation = _parse_segmentation(target_config.get("segmentation", {}))
        feature_config = _parse_feature_config(target_config.get("feature"))
        root = Path(target_config.get("root", "targetData"))
        channel_bearings = target_config.get("channel_bearings", {})
        pattern = target_config.get("pattern", "*.mat")
        sampling_rate = float(target_config.get("sampling_rate", 32000))
        rpm_value = target_config.get("rpm")
        rpm = float(rpm_value) if rpm_value is not None else None

        target_features, target_metadata = build_target_feature_table(
            root=root,
            segmentation=segmentation,
            sampling_rate=sampling_rate,
            rpm=rpm,
            channel_bearings=channel_bearings,
            feature_config=feature_config,
            pattern=pattern,
        )
        feature_path = output_root / outputs.get("target_feature_table", "target_features.csv")
        metadata_path = output_root / outputs.get("target_metadata", "target_metadata.csv")
        if not target_features.empty:
            LOGGER.info("Writing %s", feature_path)
            target_features.to_csv(feature_path, index=False)
            LOGGER.info("Writing %s", metadata_path)
            target_metadata.to_csv(metadata_path, index=False)
        else:
            LOGGER.warning("No target features were extracted; verify the target dataset path.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract diagnostic features from source and target domains.")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Optional directory override for generated feature tables."
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_pipeline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
