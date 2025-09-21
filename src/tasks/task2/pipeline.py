"""Orchestration helpers for task 2 source-domain modelling."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import json
import logging

import numpy as np
import pandas as pd

from ...modeling import (
    AlignmentConfig,
    CrossValidationConfig,
    LogisticModelConfig,
    PermutationImportanceConfig,
    SourceDiagnosisConfig,
    TrainTestSplitConfig,
    TrainingResult,
    train_source_domain_model,
)

LOGGER = logging.getLogger(__name__)


@dataclass
class Task2Config:
    """Lightweight container storing resolved configuration objects."""

    features_config: Dict[str, Any]
    alignment: AlignmentConfig
    split: TrainTestSplitConfig
    model: LogisticModelConfig
    cross_validation: CrossValidationConfig
    permutation: PermutationImportanceConfig
    outputs: Dict[str, Any]


def _parse_split_config(config: Optional[Dict[str, Any]]) -> TrainTestSplitConfig:
    if not config:
        return TrainTestSplitConfig()
    return TrainTestSplitConfig(
        test_size=float(config.get("test_size", 0.25)),
        random_state=int(config.get("random_state", 42)),
        stratify=bool(config.get("stratify", True)),
    )


def _parse_alignment_config(config: Optional[Dict[str, Any]]) -> AlignmentConfig:
    if not config:
        return AlignmentConfig()
    return AlignmentConfig(
        enabled=bool(config.get("enabled", True)),
        epsilon=float(config.get("epsilon", 1e-6)),
    )


def _parse_model_config(config: Optional[Dict[str, Any]]) -> LogisticModelConfig:
    if not config:
        return LogisticModelConfig()
    class_weight = config.get("class_weight")
    if class_weight is not None:
        class_weight = str(class_weight)
    return LogisticModelConfig(
        penalty=str(config.get("penalty", "l2")),
        C=float(config.get("C", 1.0)),
        solver=str(config.get("solver", "lbfgs")),
        max_iter=int(config.get("max_iter", 400)),
        multi_class=str(config.get("multi_class", "auto")),
        class_weight=class_weight,
    )


def _parse_cross_validation_config(config: Optional[Dict[str, Any]]) -> CrossValidationConfig:
    if not config:
        return CrossValidationConfig()
    return CrossValidationConfig(
        enabled=bool(config.get("enabled", True)),
        folds=int(config.get("folds", 5)),
        shuffle=bool(config.get("shuffle", True)),
    )


def _parse_permutation_config(config: Optional[Dict[str, Any]]) -> PermutationImportanceConfig:
    if not config:
        return PermutationImportanceConfig()
    scoring = config.get("scoring")
    if scoring is not None:
        scoring = str(scoring)
    return PermutationImportanceConfig(
        enabled=bool(config.get("enabled", True)),
        n_repeats=int(config.get("n_repeats", 10)),
        random_state=(int(config["random_state"]) if config.get("random_state") is not None else None),
        scoring=scoring,
    )


def resolve_task2_config(raw_config: Dict[str, Any]) -> Task2Config:
    features_config = raw_config.get("features", {})
    alignment = _parse_alignment_config(raw_config.get("alignment"))
    split = _parse_split_config(raw_config.get("split"))
    model = _parse_model_config(raw_config.get("model"))
    cross_validation = _parse_cross_validation_config(raw_config.get("cross_validation"))
    interpretability_config = raw_config.get("interpretability", {}) or {}
    permutation = _parse_permutation_config(interpretability_config.get("permutation_importance"))
    outputs = raw_config.get("outputs", {})
    return Task2Config(
        features_config=features_config,
        alignment=alignment,
        split=split,
        model=model,
        cross_validation=cross_validation,
        permutation=permutation,
        outputs=outputs,
    )


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_outputs(
    result: TrainingResult,
    output_dir: Path,
    outputs_config: Dict[str, Any],
    feature_table: pd.DataFrame,
) -> None:
    metrics_path = output_dir / outputs_config.get("metrics_file", "metrics.json")
    report_path = output_dir / outputs_config.get("classification_report", "classification_report.csv")
    confusion_path = output_dir / outputs_config.get("confusion_matrix", "confusion_matrix.csv")
    coefficients_path = output_dir / outputs_config.get("coefficient_importance", "coefficient_importance.csv")
    permutation_path = output_dir / outputs_config.get("permutation_importance", "permutation_importance.csv")
    predictions_path = output_dir / outputs_config.get("predictions", "predictions.csv")
    model_path = output_dir / outputs_config.get("model_path", "source_domain_model.joblib")
    summary_path = output_dir / outputs_config.get("feature_summary", "feature_summary.csv")
    features_used_path = output_dir / outputs_config.get("features_used", "features_used.txt")

    metrics_payload: Dict[str, Any] = {
        "train_accuracy": result.metrics.get("train_accuracy"),
        "test_accuracy": result.metrics.get("test_accuracy"),
        "macro_f1": result.metrics.get("macro_f1"),
        "n_classes": len(result.classes),
        "n_features": len(result.feature_columns),
        "test_samples": int(len(result.y_test)),
    }
    if result.cv_scores is not None:
        metrics_payload.update(
            {
                "cv_mean_accuracy": float(np.mean(result.cv_scores)),
                "cv_std_accuracy": float(np.std(result.cv_scores)),
                "cv_scores": [float(x) for x in result.cv_scores],
            }
        )
    if result.alignment_diagnostics:
        metrics_payload["alignment"] = result.alignment_diagnostics

    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    result.classification_report.to_csv(report_path, encoding="utf-8-sig")
    result.confusion_matrix.to_csv(confusion_path, encoding="utf-8-sig")
    result.coefficient_importance.to_csv(coefficients_path, index=False, encoding="utf-8-sig")
    if result.permutation_importance is not None:
        result.permutation_importance.to_csv(permutation_path, index=False, encoding="utf-8-sig")
    result.predictions.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    from joblib import dump

    dump(result.pipeline, model_path)

    feature_summary = feature_table[result.feature_columns].describe().transpose()
    feature_summary.to_csv(summary_path, encoding="utf-8-sig")
    with features_used_path.open("w", encoding="utf-8") as handle:
        for feature in result.feature_columns:
            handle.write(f"{feature}\n")


def run_training(
    raw_config: Dict[str, Any],
    feature_table_override: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Optional[TrainingResult]:
    config = resolve_task2_config(raw_config)

    table_path = Path(config.features_config.get("table_path", "artifacts/source_features.csv"))
    if feature_table_override is not None:
        table_path = feature_table_override

    if not table_path.exists():
        LOGGER.error("Feature table %s not found. Run the feature extraction pipeline first.", table_path)
        return None

    feature_table = pd.read_csv(table_path)
    label_column = config.features_config.get("label_column", "label")
    feature_columns = config.features_config.get("feature_columns")
    if feature_columns is not None:
        feature_columns = [str(column) for column in feature_columns]
    exclude_columns = config.features_config.get("exclude_columns")
    if exclude_columns is not None:
        exclude_columns = [str(column) for column in exclude_columns]

    diagnosis_config = SourceDiagnosisConfig(
        label_column=str(label_column),
        feature_columns=feature_columns,
        exclude_columns=exclude_columns,
        alignment=config.alignment,
        split=config.split,
        model=config.model,
        cross_validation=config.cross_validation,
        permutation_importance=config.permutation,
    )

    result = train_source_domain_model(feature_table, diagnosis_config)

    outputs_config = config.outputs
    output_root = output_dir_override or Path(outputs_config.get("directory", "artifacts/task2"))
    _ensure_directory(output_root)
    _write_outputs(result, output_root, outputs_config, feature_table)

    LOGGER.info("Training complete. Metrics saved to %s", output_root)
    return result


__all__ = [
    "Task2Config",
    "resolve_task2_config",
    "run_training",
]
