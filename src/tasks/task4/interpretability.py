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


def _add_display_names(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty or "feature" not in frame.columns:
        return frame
    display = frame.get("feature_chinese")
    if display is None:
        display = frame["feature"].copy()
    else:
        display = display.fillna(frame["feature"])
    frame = frame.copy()
    frame["feature_display"] = display
    frame.loc[frame["feature"] == "__intercept__", "feature_display"] = "截距"
    return frame


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
    per_feature = _add_display_names(per_feature)

    per_category = per_feature.groupby(["class", "category"], as_index=False)["abs_coefficient"].sum()
    per_category.rename(columns={"abs_coefficient": "importance"}, inplace=True)

    return GlobalInterpretabilityResult(per_feature=per_feature, per_category=per_category)


def compute_domain_shift_contributions(result: TransferResult) -> pd.DataFrame:
    classifier = _extract_classifier(result.final_pipeline)
    if not hasattr(classifier, "coef_"):
        raise ValueError("Classifier does not expose coefficients for interpretation.")

    feature_columns = result.feature_columns
    if not feature_columns:
        return pd.DataFrame()

    try:
        source_transformed = _transform_for_classifier(result.final_pipeline, result.source_features, feature_columns)
        target_transformed = _transform_for_classifier(result.final_pipeline, result.target_features, feature_columns)
    except KeyError:
        return pd.DataFrame()

    if source_transformed.size == 0 or target_transformed.size == 0:
        return pd.DataFrame()

    source_mean = pd.Series(np.mean(source_transformed, axis=0), index=feature_columns)
    target_mean = pd.Series(np.mean(target_transformed, axis=0), index=feature_columns)
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
    contribution_df = _add_display_names(contribution_df)
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
    explanation = _add_display_names(explanation)
    return explanation


def plot_global_importance(result: GlobalInterpretabilityResult, output_path: Path, top_n: int = 20) -> None:
    configure_chinese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary = result.per_feature.sort_values("abs_coefficient", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("Global importance summary is empty; skipping plot")
        return
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    summary = summary.copy()
    summary["feature_display"] = summary.get("feature_display", summary["feature"]).fillna(summary["feature"])
    categories = summary["category"].fillna("未分类").astype(str)
    unique_categories = list(categories.unique())
    palette = plt.cm.get_cmap("tab20", max(len(unique_categories), 1))
    colour_map = {cat: palette(index) for index, cat in enumerate(unique_categories)}
    bar_colours = [colour_map[cat] for cat in categories]
    bars = plt.barh(summary["feature_display"], summary["abs_coefficient"], color=bar_colours)
    plt.xlabel("|系数|")
    plt.ylabel("特征")
    plt.title("全局特征重要度（绝对系数）")
    plt.gca().invert_yaxis()
    legend_handles = [Patch(facecolor=colour_map[cat], edgecolor="none", label=cat) for cat in unique_categories]
    if legend_handles:
        plt.legend(handles=legend_handles, title="特征类别", fontsize="small")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_domain_shift(contributions: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    configure_chinese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    summary = contributions.sort_values("abs_contribution", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("Domain shift contribution table is empty; skipping plot")
        return
    plt.figure(figsize=(10, max(4, top_n * 0.3)))
    summary = summary.copy()
    summary["feature_display"] = summary.get("feature_display", summary["feature"]).fillna(summary["feature"])
    classes = summary["class"].astype(str)
    unique_classes = list(classes.unique())
    palette = plt.cm.get_cmap("Set2", max(len(unique_classes), 1))
    colour_map = {cls: palette(index) for index, cls in enumerate(unique_classes)}
    bar_colours = [colour_map[cls] for cls in classes]
    bars = plt.barh(summary["feature_display"], summary["logit_contribution"], color=bar_colours)
    plt.xlabel("Logit 变化量")
    plt.ylabel("特征")
    plt.title("域偏移特征贡献排名")
    plt.gca().invert_yaxis()
    legend_handles = [Patch(facecolor=colour_map[cls], edgecolor="none", label=cls) for cls in unique_classes]
    if legend_handles:
        plt.legend(handles=legend_handles, title="类别", fontsize="small", loc="lower right")
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
    summary = summary.copy()
    summary["feature_display"] = summary.get("feature_display", summary["feature"]).fillna(summary["feature"])
    contributions = summary["logit_contribution"].to_numpy()
    colours = ["#d62728" if value >= 0 else "#1f77b4" for value in contributions]
    plt.barh(summary["feature_display"], contributions, color=colours)
    plt.xlabel("对Logit的贡献")
    plt.ylabel("特征")
    plt.title("样本特征贡献排序")
    plt.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
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
