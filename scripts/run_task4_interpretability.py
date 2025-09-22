"""Generate interpretability artefacts for task 4."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import pandas as pd
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.tasks.task3 import parse_transfer_config, run_transfer_learning
from src.tasks.task4 import (
    GlobalInterpretabilityResult,
    cluster_local_explanations,
    compute_domain_shift_contributions,
    compute_global_feature_effects,
    explain_samples,
    plot_domain_shift,
    plot_global_importance,
    plot_local_cluster_heatmap,
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


def _load_features(
    config: Dict[str, Any], source_override: Optional[Path], target_override: Optional[Path]
) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    global_cfg = raw_config.get("global_importance", {})
    global_method = str(global_cfg.get("method", "auto"))
    global_shap_background = int(global_cfg.get("shap_background", 200))
    global_shap_nsamples = int(global_cfg.get("shap_nsamples", 200))
    global_top_n = int(global_cfg.get("top_n", 20))

    global_result = compute_global_feature_effects(
        result,
        method=global_method,
        shap_background=global_shap_background,
        shap_nsamples=global_shap_nsamples,
    )
    _write_dataframe(global_result.per_feature, global_feature_path)
    _write_dataframe(global_result.per_category, global_category_path)

    domain_cfg = raw_config.get("domain_shift", {})
    domain_top_n = int(domain_cfg.get("top_n", 20))
    domain_df = compute_domain_shift_contributions(result)
    _write_dataframe(domain_df, domain_shift_path)

    local_cfg = raw_config.get("local_explanation", {})
    dataset = str(local_cfg.get("dataset", "target")).lower()
    indices_cfg = local_cfg.get("indices")
    if indices_cfg is None:
        indices: List[int] = [int(local_cfg.get("index", 0))]
    else:
        indices = [int(value) for value in indices_cfg]
    method = str(local_cfg.get("method", "auto"))
    top_n = int(local_cfg.get("top_n", 10))

    shap_cfg = local_cfg.get("shap", {})
    shap_background = int(shap_cfg.get("background", 200))
    shap_nsamples = int(shap_cfg.get("nsamples", 200))

    lime_cfg = local_cfg.get("lime", {})
    lime_num_samples = int(lime_cfg.get("num_samples", 5000))
    lime_random_state = int(lime_cfg.get("random_state", 42))

    target_class_cfg = local_cfg.get("class")
    target_classes: Optional[Sequence[Optional[str]]]
    if isinstance(target_class_cfg, list):
        target_classes = [None if value is None else str(value) for value in target_class_cfg]
    elif target_class_cfg is None:
        target_classes = None
    else:
        target_classes = [str(target_class_cfg)]

    explanations = explain_samples(
        result,
        dataset=dataset,
        indices=indices,
        target_classes=target_classes,
        method=method,
        shap_background=shap_background,
        shap_nsamples=shap_nsamples,
        lime_num_samples=lime_num_samples,
        random_state=lime_random_state,
    )
    _write_dataframe(explanations, local_table_path)

    if not explanations.empty:
        first_index = indices[0]
        first_sample = explanations[explanations["样本索引"] == first_index].copy()
        first_sample.sort_values("贡献绝对值", ascending=False, inplace=True)
        plot_local_explanation(first_sample, local_plot_path, top_n=top_n)
    else:
        LOGGER.warning("未生成局部解释结果，跳过单样本绘图。")

    plot_global_importance(global_result, global_plot_path, top_n=min(global_top_n, len(global_result.per_feature)))

    cluster_cfg = local_cfg.get("cluster", {})
    cluster_enabled = bool(cluster_cfg.get("enabled", len(indices) > 1))
    cluster_n = int(cluster_cfg.get("n_clusters", max(1, min(3, len(set(indices))))))
    cluster_random_state = int(cluster_cfg.get("random_state", 42))
    cluster_top_n = int(cluster_cfg.get("top_n", 12))

    cluster_assign_path = output_dir / outputs_cfg.get("local_cluster_assignments", "local_cluster_assignments.csv")
    cluster_profile_path = output_dir / outputs_cfg.get("local_cluster_profiles", "local_cluster_profiles.csv")
    cluster_plot_path = output_dir / outputs_cfg.get("local_cluster_plot", "local_cluster_heatmap.png")

    cluster_assign_df = pd.DataFrame()
    cluster_profile_df = pd.DataFrame()
    if cluster_enabled and not explanations.empty and explanations["样本索引"].nunique() > 1:
        cluster_assign_df, cluster_profile_df = cluster_local_explanations(
            explanations,
            n_clusters=cluster_n,
            random_state=cluster_random_state,
        )
        _write_dataframe(cluster_assign_df, cluster_assign_path)
        _write_dataframe(cluster_profile_df, cluster_profile_path)
        plot_local_cluster_heatmap(cluster_profile_df, cluster_plot_path, top_n=cluster_top_n)

    plot_domain_shift(domain_df, domain_plot_path, top_n=min(domain_top_n, len(domain_df)))

    metrics_payload = {
        "feature_columns": result.feature_columns,
        "time_frequency_feature_count": len(result.time_frequency_features),
        "pseudo_label_count": int(len(result.pseudo_labels)),
        "pseudo_iterations": int(len(result.pseudo_label_history)),
        "local_explanation_dataset": dataset,
        "local_explanation_indices": indices,
        "local_explanation_method": method,
        "cluster_enabled": bool(cluster_enabled and not explanations.empty and explanations["样本索引"].nunique() > 1),
        "cluster_count": int(cluster_profile_df["聚类标签"].nunique()) if not cluster_profile_df.empty else 0,
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
