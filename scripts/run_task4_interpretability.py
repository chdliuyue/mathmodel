"""Generate interpretability artefacts for task 4."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tasks.task3 import parse_transfer_config, run_transfer_learning
from src.tasks.task4 import (
    GlobalInterpretabilityResult,
    compute_domain_shift_contributions,
    compute_global_feature_effects,
    explain_instance,
    plot_domain_shift,
    plot_global_importance,
    plot_local_explanation,
)

LOGGER = logging.getLogger("task4_interpretability")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _load_features(config: Dict[str, Any], source_override: Optional[Path], target_override: Optional[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    features_cfg = config.get("features", {})
    source_path = Path(features_cfg.get("source_table", "artifacts/task1/source_features.csv"))
    target_path = Path(features_cfg.get("target_table", "artifacts/task1/target_features.csv"))
    if source_override is not None:
        source_path = source_override
    if target_override is not None:
        target_path = target_override
    if not source_path.exists() or not target_path.exists():
        raise FileNotFoundError("Feature tables not found. Ensure task 1 outputs are available.")
    source_df = pd.read_csv(source_path)
    target_df = pd.read_csv(target_path)
    return source_df, target_df


def run_pipeline(
    config_path: Path,
    source_override: Optional[Path] = None,
    target_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> None:
    raw_config = _load_yaml(config_path)
    transfer_config = parse_transfer_config(raw_config)
    source_df, target_df = _load_features(raw_config, source_override, target_override)

    result = run_transfer_learning(source_df, target_df, transfer_config)
    outputs_cfg = raw_config.get("outputs", {})
    output_dir = output_override or Path(outputs_cfg.get("directory", "artifacts/task4"))
    _ensure_directory(output_dir)

    global_feature_path = output_dir / outputs_cfg.get("global_feature_table", "global_feature_effects.csv")
    global_category_path = output_dir / outputs_cfg.get("global_category_table", "global_feature_categories.csv")
    domain_shift_path = output_dir / outputs_cfg.get("domain_shift_table", "domain_shift_contributions.csv")
    local_table_path = output_dir / outputs_cfg.get("local_explanations", "local_explanation.csv")
    global_plot_path = output_dir / outputs_cfg.get("global_plot", "global_importance.png")
    domain_plot_path = output_dir / outputs_cfg.get("domain_shift_plot", "domain_shift.png")
    local_plot_path = output_dir / outputs_cfg.get("local_plot", "local_explanation.png")
    metrics_path = output_dir / outputs_cfg.get("metrics", "interpretability_summary.json")

    global_result = compute_global_feature_effects(result)
    _write_dataframe(global_result.per_feature, global_feature_path)
    _write_dataframe(global_result.per_category, global_category_path)

    domain_df = compute_domain_shift_contributions(result)
    _write_dataframe(domain_df, domain_shift_path)

    local_cfg = raw_config.get("local_explanation", {})
    dataset = str(local_cfg.get("dataset", "target")).lower()
    sample_index = int(local_cfg.get("index", 0))
    target_class = local_cfg.get("class")
    top_n = int(local_cfg.get("top_n", 10))

    if dataset == "source":
        frame = result.source_features.reset_index(drop=True)
    else:
        frame = result.target_features.reset_index(drop=True)
    if sample_index < 0 or sample_index >= len(frame):
        raise IndexError("Sample index out of bounds for local explanation.")
    sample = frame.iloc[sample_index]
    explanation_df = explain_instance(result, sample, target_class=target_class)
    _write_dataframe(explanation_df, local_table_path)

    plot_global_importance(global_result, global_plot_path, top_n=min(top_n * 2, len(global_result.per_feature)))
    plot_domain_shift(domain_df, domain_plot_path, top_n=min(top_n * 2, len(domain_df)))
    plot_local_explanation(explanation_df.head(top_n), local_plot_path, top_n=top_n)

    metrics_payload = {
        "feature_columns": result.feature_columns,
        "time_frequency_feature_count": len(result.time_frequency_features),
        "pseudo_label_count": int(len(result.pseudo_labels)),
        "pseudo_iterations": int(len(result.pseudo_label_history)),
        "local_explanation_dataset": dataset,
        "local_explanation_index": sample_index,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    LOGGER.info("Interpretability artefacts saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the task 4 interpretability workflow.")
    parser.add_argument("--config", type=Path, default=Path("config/task4_config.yaml"), help="Path to the YAML configuration file.")
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
