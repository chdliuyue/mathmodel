"""Interpretability utilities for analysing the transfer diagnosis pipeline."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import logging

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from ...analysis.feature_analysis import configure_chinese_font
from ...analysis.feature_dictionary import build_feature_dictionary
from ..task3.transfer import TransferResult

LOGGER = logging.getLogger(__name__)


@dataclass
class GlobalInterpretabilityResult:
    """Global feature attribution results."""

    per_feature: pd.DataFrame
    per_category: pd.DataFrame


def _extract_classifier(pipeline: Pipeline):
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("Pipeline does not contain a classifier step.")
    return classifier


def _transform_for_classifier(pipeline: Pipeline, features: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    data = features[feature_columns].astype(float)
    if len(pipeline.steps) <= 1:
        return data.values
    feature_pipeline = Pipeline(pipeline.steps[:-1])
    return feature_pipeline.transform(data)


def compute_global_feature_effects(result: TransferResult) -> GlobalInterpretabilityResult:
    classifier = _extract_classifier(result.final_pipeline)
    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier does not expose coefficients for interpretation.")

    feature_columns = result.feature_columns
    classes = getattr(classifier, "classes_", [])
    records: List[Dict[str, float]] = []
    for class_index, class_name in enumerate(classes):
        for feature_index, feature_name in enumerate(feature_columns):
            coefficient = float(classifier.coef_[class_index, feature_index])
            records.append(
                {
                    "class": str(class_name),
                    "feature": feature_name,
                    "coefficient": coefficient,
                    "abs_coefficient": abs(coefficient),
                    "odds_ratio": float(np.exp(coefficient)),
                }
            )
        if hasattr(classifier, "intercept_"):
            intercept = float(classifier.intercept_[class_index])
            records.append(
                {
                    "class": str(class_name),
                    "feature": "__intercept__",
                    "coefficient": intercept,
                    "abs_coefficient": abs(intercept),
                    "odds_ratio": float(np.exp(intercept)),
                }
            )
    per_feature = pd.DataFrame(records)

    dictionary = build_feature_dictionary(feature_columns)
    per_feature = per_feature.merge(
        dictionary[["feature", "category", "chinese_name"]],
        on="feature",
        how="left",
    )
    per_feature.rename(columns={"chinese_name": "feature_chinese"}, inplace=True)

    per_category = per_feature.groupby(["class", "category"], as_index=False)["abs_coefficient"].sum()
    per_category.rename(columns={"abs_coefficient": "importance"}, inplace=True)

    return GlobalInterpretabilityResult(per_feature=per_feature, per_category=per_category)


def compute_domain_shift_contributions(result: TransferResult) -> pd.DataFrame:
    classifier = _extract_classifier(result.final_pipeline)
    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier does not expose coefficients for interpretation.")

    feature_columns = result.feature_columns
    source_mean = result.source_features[feature_columns].mean(axis=0)
    target_mean = result.target_features[feature_columns].mean(axis=0)
    delta = target_mean - source_mean

    dictionary = build_feature_dictionary(feature_columns)

    records: List[Dict[str, float]] = []
    classes = getattr(classifier, "classes_", [])
    for class_index, class_name in enumerate(classes):
        coefficients = classifier.coef_[class_index]
        contributions = coefficients * delta.values
        for feature, coeff, contrib in zip(feature_columns, coefficients, contributions):
            record = {
                "class": str(class_name),
                "feature": feature,
                "coefficient": float(coeff),
                "delta": float(delta[feature]),
                "logit_contribution": float(contrib),
            }
            records.append(record)
    contribution_df = pd.DataFrame(records)
    contribution_df = contribution_df.merge(
        dictionary[["feature", "category", "chinese_name"]],
        on="feature",
        how="left",
    )
    contribution_df.rename(columns={"chinese_name": "feature_chinese"}, inplace=True)
    contribution_df["abs_contribution"] = contribution_df["logit_contribution"].abs()
    return contribution_df


def explain_instance(
    result: TransferResult,
    sample: pd.Series,
    target_class: Optional[str] = None,
    include_intercept: bool = True,
) -> pd.DataFrame:
    classifier = _extract_classifier(result.final_pipeline)
    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier does not expose coefficients for interpretation.")

    feature_columns = result.feature_columns
    transformed = _transform_for_classifier(result.final_pipeline, sample.to_frame().T, feature_columns)
    coefficients = classifier.coef_
    intercepts = classifier.intercept_ if hasattr(classifier, "intercept_") else np.zeros(coefficients.shape[0])
    classes = getattr(classifier, "classes_", [])

    if target_class is None:
        if hasattr(classifier, "predict_proba"):
            scores = classifier.predict_proba(transformed)
            target_index = int(np.argmax(scores))
        else:
            scores = classifier.decision_function(transformed)
            target_index = int(np.argmax(scores))
    else:
        try:
            target_index = list(classes).index(target_class)
        except ValueError as exc:
            raise ValueError(f"Class {target_class!r} not present in classifier.") from exc

    feature_values = transformed[0]
    contributions = coefficients[target_index] * feature_values
    records = []
    for feature_name, contribution, coefficient in zip(feature_columns, contributions, coefficients[target_index]):
        records.append(
            {
                "feature": feature_name,
                "value": float(sample[feature_name]),
                "transformed_value": float(feature_values[result.feature_columns.index(feature_name)]),
                "coefficient": float(coefficient),
                "logit_contribution": float(contribution),
            }
        )
    if include_intercept:
        intercept = float(intercepts[target_index])
        records.append(
            {
                "feature": "__intercept__",
                "value": 1.0,
                "transformed_value": 1.0,
                "coefficient": intercept,
                "logit_contribution": intercept,
            }
        )

    explanation = pd.DataFrame(records)
    dictionary = build_feature_dictionary(feature_columns)
    explanation = explanation.merge(
        dictionary[["feature", "category", "chinese_name"]],
        on="feature",
        how="left",
    )
    explanation.rename(columns={"chinese_name": "feature_chinese"}, inplace=True)
    explanation.sort_values("logit_contribution", key=lambda x: x.abs(), ascending=False, inplace=True)
    return explanation


def plot_global_importance(result: GlobalInterpretabilityResult, output_path: Path, top_n: int = 20) -> None:
    configure_chinese_font()
    import matplotlib.pyplot as plt

    summary = result.per_feature.sort_values("abs_coefficient", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("Global importance summary is empty; skipping plot")
        return
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    colours = summary["category"].astype(str)
    bars = plt.barh(summary["feature"], summary["abs_coefficient"], color="steelblue")
    plt.xlabel("|Coefficient|")
    plt.ylabel("Feature")
    plt.title("Global feature importance (absolute coefficients)")
    plt.gca().invert_yaxis()
    for bar, cat in zip(bars, colours):
        bar.set_alpha(0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_domain_shift(contributions: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    configure_chinese_font()
    import matplotlib.pyplot as plt

    summary = contributions.sort_values("abs_contribution", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("Domain shift contribution table is empty; skipping plot")
        return
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    colours = summary["class"].astype(str)
    bars = plt.barh(summary["feature"], summary["logit_contribution"], color="darkorange")
    plt.xlabel("Logit shift")
    plt.ylabel("Feature")
    plt.title("Feature-level contributions to domain shift")
    plt.gca().invert_yaxis()
    for bar, cls in zip(bars, colours):
        bar.set_alpha(0.8)
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_local_explanation(explanation: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    configure_chinese_font()
    import matplotlib.pyplot as plt

    summary = explanation.head(top_n)
    if summary.empty:
        LOGGER.warning("Local explanation table is empty; skipping plot")
        return
    plt.figure(figsize=(10, max(4, top_n * 0.4)))
    bars = plt.barh(summary["feature"], summary["logit_contribution"], color="seagreen")
    plt.xlabel("Logit contribution")
    plt.ylabel("Feature")
    plt.title("Top feature contributions for the analysed sample")
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


__all__ = [
    "GlobalInterpretabilityResult",
    "compute_global_feature_effects",
    "compute_domain_shift_contributions",
    "explain_instance",
    "plot_global_importance",
    "plot_domain_shift",
    "plot_local_explanation",
]
