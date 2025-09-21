"""Run the transfer diagnosis pipeline (task 3)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
from joblib import dump
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.feature_analysis import plot_embedding, run_tsne
from src.tasks.task3 import TransferConfig, parse_transfer_config, run_transfer_learning

LOGGER = logging.getLogger("task3_transfer")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_transfer_config(raw: Dict[str, Any]) -> TransferConfig:
    return parse_transfer_config(raw)


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def run_pipeline(
    config_path: Path,
    source_override: Optional[Path] = None,
    target_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> None:
    raw_config = _load_yaml(config_path)
    transfer_config = _build_transfer_config(raw_config)

    features_cfg = raw_config.get("features", {})
    source_path = Path(features_cfg.get("source_table", "artifacts/source_features.csv"))
    target_path = Path(features_cfg.get("target_table", "artifacts/target_features.csv"))
    if source_override is not None:
        source_path = source_override
    if target_override is not None:
        target_path = target_override
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError("Feature tables not found. Ensure task 1 outputs are available.")

    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)

    result = run_transfer_learning(source_df, target_df, transfer_config)

    outputs_cfg = raw_config.get("outputs", {})
    output_dir = output_override or Path(outputs_cfg.get("directory", "artifacts/task3"))
    _ensure_directory(output_dir)

    initial_predictions_path = output_dir / outputs_cfg.get("predictions_initial", "target_predictions_initial.csv")
    final_predictions_path = output_dir / outputs_cfg.get("predictions_final", "target_predictions.csv")
    pseudo_labels_path = output_dir / outputs_cfg.get("pseudo_labels", "pseudo_labels.csv")
    pseudo_history_path = output_dir / outputs_cfg.get("pseudo_history", "pseudo_history.csv")
    alignment_before_path = output_dir / outputs_cfg.get("alignment_before", "alignment_before.csv")
    alignment_after_path = output_dir / outputs_cfg.get("alignment_after", "alignment_after.csv")
    combined_before_path = output_dir / outputs_cfg.get("combined_before", "combined_features_before.csv")
    combined_aligned_path = output_dir / outputs_cfg.get("combined_aligned", "combined_features_aligned.csv")
    tf_feature_path = output_dir / outputs_cfg.get("time_frequency_features", "time_frequency_features.txt")
    metrics_path = output_dir / outputs_cfg.get("metrics", "metrics.json")
    embedding_before_path = output_dir / outputs_cfg.get("embedding_before", "tsne_before.png")
    embedding_after_path = output_dir / outputs_cfg.get("embedding_after", "tsne_after.png")
    model_path = output_dir / outputs_cfg.get("model_path", "transfer_model.joblib")

    _write_dataframe(result.initial_predictions, initial_predictions_path)
    _write_dataframe(result.final_predictions, final_predictions_path)
    _write_dataframe(result.pseudo_labels, pseudo_labels_path)
    _write_dataframe(pd.DataFrame(result.pseudo_label_history), pseudo_history_path)
    _write_dataframe(result.alignment_before, alignment_before_path)
    _write_dataframe(result.alignment_after, alignment_after_path)
    _write_dataframe(result.combined_before, combined_before_path)
    _write_dataframe(result.combined_aligned, combined_aligned_path)

    if result.time_frequency_features:
        with tf_feature_path.open("w", encoding="utf-8") as handle:
            for feature in result.time_frequency_features:
                handle.write(f"{feature}\n")

    metrics_payload = {
        "base_model": result.base_result.metrics,
        "feature_columns": result.feature_columns,
        "time_frequency_feature_count": len(result.time_frequency_features),
        "pseudo_label_count": int(len(result.pseudo_labels)),
        "pseudo_iterations": int(len(result.pseudo_label_history)),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    dump(result.final_pipeline, model_path)

    tsne_before = run_tsne(result.combined_before)
    if tsne_before is not None:
        plot_embedding(tsne_before, embedding_before_path, title="t-SNE (before alignment)")
    tsne_after = run_tsne(result.combined_aligned)
    if tsne_after is not None:
        plot_embedding(tsne_after, embedding_after_path, title="t-SNE (after alignment)")

    LOGGER.info("Transfer learning pipeline completed. Outputs saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the task 3 transfer diagnosis pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config/task3_config.yaml"), help="Path to the YAML configuration file.")
    parser.add_argument("--source-features", type=Path, default=None, help="Optional override for the source feature CSV path.")
    parser.add_argument("--target-features", type=Path, default=None, help="Optional override for the target feature CSV path.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Optional override for the output directory.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    run_pipeline(
        args.config,
        source_override=args.source_features,
        target_override=args.target_features,
        output_override=args.output_dir,
    )


if __name__ == "__main__":
    main()
