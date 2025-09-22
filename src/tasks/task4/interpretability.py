"""解释性分析工具，支持多模态伪标签阶段的贡献度分析与可视化。"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from ...analysis.feature_analysis import configure_chinese_font
from ...analysis.feature_dictionary import build_feature_dictionary
from ..task3.transfer import TransferResult

LOGGER = logging.getLogger(__name__)


@dataclass
class GlobalInterpretabilityResult:
    """全局特征重要度分析结果。"""

    per_feature: pd.DataFrame
    per_category: pd.DataFrame


def _extract_classifier(pipeline: Pipeline):
    classifier = pipeline.named_steps.get("classifier")
    if classifier is None:
        raise ValueError("模型流水线缺少 classifier 步骤，无法解释。")
    return classifier


def _transform_for_classifier(pipeline: Pipeline, features: pd.DataFrame, feature_columns: Sequence[str]) -> np.ndarray:
    data = features[feature_columns].astype(float)
    if len(pipeline.steps) <= 1:
        return data.to_numpy(copy=True)
    feature_pipeline = Pipeline(pipeline.steps[:-1])
    return feature_pipeline.transform(data)


def _build_feature_metadata(feature_columns: Sequence[str]) -> pd.DataFrame:
    if not feature_columns:
        return pd.DataFrame(columns=["特征编码", "特征中文名", "特征类别", "特征显示名"])
    dictionary = build_feature_dictionary(feature_columns)
    dictionary = dictionary.rename(
        columns={"feature": "特征编码", "chinese_name": "特征中文名", "category": "特征类别"}
    )
    dictionary["特征显示名"] = dictionary["特征中文名"].fillna(dictionary["特征编码"])
    return dictionary


def _compute_global_shap_summary(
    result: TransferResult,
    background_size: int,
    nsamples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    import shap

    feature_columns = result.feature_columns
    if not feature_columns:
        return np.zeros((0, 0)), np.zeros((0, 0)), np.zeros((0, 1)), []

    background = result.source_features[feature_columns].astype(float)
    if background.empty:
        background = result.target_features[feature_columns].astype(float)
    if background.empty:
        return np.zeros((0, len(feature_columns))), np.zeros((0, len(feature_columns))), np.zeros((0, 1)), []

    if len(background) > background_size:
        background_sample = background.sample(background_size, random_state=42)
    else:
        background_sample = background

    target = result.target_features[feature_columns].astype(float)
    if target.empty:
        target = background_sample
    if len(target) > background_size:
        target_sample = target.sample(background_size, random_state=42)
    else:
        target_sample = target

    explainer = shap.KernelExplainer(result.final_pipeline.predict_proba, background_sample.to_numpy())
    shap_values = explainer.shap_values(target_sample.to_numpy(), nsamples=nsamples)
    expected_value = explainer.expected_value

    if isinstance(shap_values, list):
        shap_array = np.stack([np.asarray(values, dtype=float) for values in shap_values])
    else:
        raw = np.asarray(shap_values, dtype=float)
        shap_array = raw.reshape(1, raw.shape[0], raw.shape[1])

    mean_contribution = shap_array.mean(axis=1)
    mean_absolute = np.abs(shap_array).mean(axis=1)

    if isinstance(expected_value, (list, np.ndarray)):
        baseline = np.asarray(expected_value, dtype=float).reshape(mean_contribution.shape[0], 1)
    else:
        baseline = np.full((mean_contribution.shape[0], 1), float(expected_value))

    classifier = _extract_classifier(result.final_pipeline)
    classes = getattr(classifier, "classes_", [])
    if not len(classes):
        classes = [f"类别{idx}" for idx in range(mean_contribution.shape[0])]
    class_labels = [str(label) for label in classes]

    return mean_contribution, mean_absolute, baseline, class_labels


def compute_global_feature_effects(
    result: TransferResult,
    method: str = "auto",
    shap_background: int = 200,
    shap_nsamples: int = 200,
) -> GlobalInterpretabilityResult:
    classifier = _extract_classifier(result.final_pipeline)
    feature_columns = result.feature_columns
    dictionary = _build_feature_metadata(feature_columns)

    has_coef = hasattr(classifier, "coef_")
    chosen = method.lower()
    if chosen not in {"auto", "linear", "shap"}:
        raise ValueError(f"不支持的全局解释方法: {method}")
    if chosen == "auto":
        chosen = "linear" if has_coef else "shap"

    records: List[Dict[str, float]] = []
    intercept_label = "截距"

    if chosen == "linear":
        if not has_coef:
            raise ValueError("模型未提供线性系数，无法使用线性解释。")
        classes = getattr(classifier, "classes_", [])
        for class_index, class_name in enumerate(classes):
            coefficients = classifier.coef_[class_index]
            for feature_name, coefficient in zip(feature_columns, coefficients):
                records.append(
                    {
                        "类别": str(class_name),
                        "特征编码": feature_name,
                        "影响值": float(coefficient),
                        "重要性": float(abs(coefficient)),
                        "优势比": float(np.exp(coefficient)),
                    }
                )
            if hasattr(classifier, "intercept_"):
                intercept = float(classifier.intercept_[class_index])
                records.append(
                    {
                        "类别": str(class_name),
                        "特征编码": "__intercept__",
                        "影响值": intercept,
                        "重要性": abs(intercept),
                        "优势比": float(np.exp(intercept)),
                    }
                )
    else:  # SHAP 全局解释
        intercept_label = "基线贡献"
        mean_contribution, mean_absolute, baseline, class_labels = _compute_global_shap_summary(
            result,
            background_size=shap_background,
            nsamples=shap_nsamples,
        )
        for class_index, class_name in enumerate(class_labels):
            if mean_contribution.size:
                for feature_index, feature_name in enumerate(feature_columns):
                    records.append(
                        {
                            "类别": str(class_name),
                            "特征编码": feature_name,
                            "影响值": float(mean_contribution[class_index, feature_index]),
                            "重要性": float(mean_absolute[class_index, feature_index]),
                            "优势比": np.nan,
                        }
                    )
            if baseline.size:
                baseline_value = float(baseline[class_index, 0])
                records.append(
                    {
                        "类别": str(class_name),
                        "特征编码": "__intercept__",
                        "影响值": baseline_value,
                        "重要性": abs(baseline_value),
                        "优势比": np.nan,
                    }
                )

    per_feature = pd.DataFrame(records)
    if per_feature.empty:
        per_feature = pd.DataFrame(
            columns=["类别", "特征编码", "影响值", "重要性", "优势比", "特征中文名", "特征类别", "特征显示名"]
        )
    else:
        per_feature = per_feature.merge(dictionary, on="特征编码", how="left")
        per_feature["特征显示名"] = per_feature["特征中文名"].fillna(per_feature["特征编码"])
        intercept_mask = per_feature["特征编码"] == "__intercept__"
        if intercept_mask.any():
            per_feature.loc[intercept_mask, "特征显示名"] = intercept_label
            per_feature.loc[intercept_mask, "特征类别"] = per_feature.loc[intercept_mask, "特征类别"].fillna("偏置")

    non_intercept = per_feature[per_feature["特征编码"] != "__intercept__"]
    if non_intercept.empty:
        per_category = pd.DataFrame(columns=["类别", "特征类别", "重要性"])
    else:
        per_category = (
            non_intercept.groupby(["类别", "特征类别"], as_index=False)["重要性"].sum()
        )

    return GlobalInterpretabilityResult(per_feature=per_feature, per_category=per_category)


def compute_domain_shift_contributions(result: TransferResult) -> pd.DataFrame:
    classifier = _extract_classifier(result.final_pipeline)
    if not hasattr(classifier, "coef_"):
        LOGGER.warning("模型未提供线性系数，域偏移贡献无法计算。")
        return pd.DataFrame()

    feature_columns = result.feature_columns
    if not feature_columns:
        return pd.DataFrame()

    try:
        source_transformed = _transform_for_classifier(result.final_pipeline, result.source_features, feature_columns)
        target_transformed = _transform_for_classifier(result.final_pipeline, result.target_features, feature_columns)
    except KeyError:
        LOGGER.warning("特征矩阵不完整，跳过域偏移分析。")
        return pd.DataFrame()

    if source_transformed.size == 0 or target_transformed.size == 0:
        return pd.DataFrame()

    source_mean = pd.Series(np.mean(source_transformed, axis=0), index=feature_columns)
    target_mean = pd.Series(np.mean(target_transformed, axis=0), index=feature_columns)
    delta = target_mean - source_mean

    dictionary = _build_feature_metadata(feature_columns)

    records: List[Dict[str, float]] = []
    classes = getattr(classifier, "classes_", [])
    for class_index, class_name in enumerate(classes):
        coefficients = classifier.coef_[class_index]
        contributions = coefficients * delta.values
        for feature, coeff, contrib in zip(feature_columns, coefficients, contributions):
            records.append(
                {
                    "类别": str(class_name),
                    "特征编码": feature,
                    "系数": float(coeff),
                    "均值差异": float(delta[feature]),
                    "Logit贡献": float(contrib),
                }
            )

    contribution_df = pd.DataFrame(records)
    if contribution_df.empty:
        return contribution_df

    contribution_df = contribution_df.merge(dictionary, on="特征编码", how="left")
    contribution_df["特征显示名"] = contribution_df["特征中文名"].fillna(contribution_df["特征编码"])
    contribution_df["绝对贡献"] = contribution_df["Logit贡献"].abs()
    return contribution_df


def _compute_shap_values(
    result: TransferResult,
    sample_frame: pd.DataFrame,
    target_index: int,
    background_size: int = 200,
    nsamples: int = 200,
) -> Tuple[np.ndarray, float]:
    import shap

    feature_columns = result.feature_columns
    background = result.source_features[feature_columns].astype(float)
    if background.empty:
        background = result.target_features[feature_columns].astype(float)
    if background.empty:
        raise ValueError("缺少可用于 SHAP 的背景样本。")
    if len(background) > background_size:
        background = background.sample(background_size, random_state=42)

    explainer = shap.KernelExplainer(result.final_pipeline.predict_proba, background.to_numpy())
    shap_values = explainer.shap_values(sample_frame[feature_columns].astype(float).to_numpy(), nsamples=nsamples)
    if isinstance(shap_values, list):
        contributions = np.asarray(shap_values[target_index][0], dtype=float)
        base_value = explainer.expected_value[target_index]
    else:
        contributions = np.asarray(shap_values[0], dtype=float)
        base_value = explainer.expected_value
    return contributions, float(base_value)


def _compute_lime_values(
    result: TransferResult,
    sample_frame: pd.DataFrame,
    target_index: int,
    num_samples: int = 5000,
    random_state: int = 42,
) -> Tuple[np.ndarray, float]:
    from lime.lime_tabular import LimeTabularExplainer

    feature_columns = result.feature_columns
    background = result.source_features[feature_columns].astype(float)
    if background.empty:
        background = result.target_features[feature_columns].astype(float)
    if background.empty:
        raise ValueError("缺少可用于 LIME 的背景样本。")

    class_names = [str(label) for label in _extract_classifier(result.final_pipeline).classes_]
    explainer = LimeTabularExplainer(
        background.to_numpy(),
        feature_names=feature_columns,
        class_names=class_names,
        mode="classification",
        discretize_continuous=True,
        sample_around_instance=True,
        random_state=random_state,
    )
    explanation = explainer.explain_instance(
        sample_frame[feature_columns].astype(float).to_numpy()[0],
        result.final_pipeline.predict_proba,
        num_features=len(feature_columns),
        num_samples=num_samples,
    )
    local_exp = explanation.local_exp.get(target_index, [])
    contributions = np.zeros(len(feature_columns), dtype=float)
    for feature_idx, weight in local_exp:
        contributions[int(feature_idx)] = float(weight)
    intercept = explanation.intercept[target_index] if isinstance(explanation.intercept, (list, np.ndarray)) else explanation.intercept
    return contributions, float(intercept)


def explain_instance(
    result: TransferResult,
    sample: pd.Series,
    target_class: Optional[str] = None,
    include_intercept: bool = True,
    method: str = "auto",
    shap_background: int = 200,
    shap_nsamples: int = 200,
    lime_num_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    pipeline = result.final_pipeline
    classifier = _extract_classifier(pipeline)
    feature_columns = result.feature_columns
    if not feature_columns:
        raise ValueError("缺少可解释的特征列。")

    sample_frame = sample.to_frame().T
    sample_frame = sample_frame[feature_columns]
    numeric_sample = sample_frame.astype(float)

    classes = list(getattr(classifier, "classes_", []))
    chosen_method = method.lower()
    if chosen_method == "auto":
        chosen_method = "linear" if hasattr(classifier, "coef_") else "shap"

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(numeric_sample)
        if target_class is None:
            target_index = int(np.argmax(probabilities[0]))
        else:
            try:
                target_index = classes.index(str(target_class)) if classes else 0
            except ValueError as exc:
                raise ValueError(f"指定的类别 {target_class!r} 不存在于模型中。") from exc
        target_probability = float(probabilities[0, target_index])
    else:
        decision = pipeline.decision_function(numeric_sample)
        if decision.ndim == 1:
            decision = decision.reshape(1, -1)
        if target_class is None:
            target_index = int(np.argmax(decision[0]))
        else:
            try:
                target_index = classes.index(str(target_class)) if classes else 0
            except ValueError as exc:
                raise ValueError(f"指定的类别 {target_class!r} 不存在于模型中。") from exc
        target_probability = None

    transformed = _transform_for_classifier(pipeline, sample_frame, feature_columns)

    if chosen_method == "linear":
        if not hasattr(classifier, "coef_"):
            raise ValueError("模型未提供线性系数，无法使用线性解释。")
        coefficients = np.asarray(classifier.coef_[target_index], dtype=float)
        contributions = coefficients * transformed[0]
        intercept_value = (
            float(classifier.intercept_[target_index])
            if include_intercept and hasattr(classifier, "intercept_")
            else None
        )
    elif chosen_method == "shap":
        contributions, base_value = _compute_shap_values(
            result,
            sample_frame,
            target_index,
            background_size=shap_background,
            nsamples=shap_nsamples,
        )
        intercept_value = float(base_value) if include_intercept else None
    elif chosen_method == "lime":
        contributions, intercept_value = _compute_lime_values(
            result,
            sample_frame,
            target_index,
            num_samples=lime_num_samples,
            random_state=random_state,
        )
        if not include_intercept:
            intercept_value = None
    else:
        raise ValueError(f"不支持的解释方法: {method}")

    target_label = str(classes[target_index]) if classes else str(target_index)

    records: List[Dict[str, float]] = []
    for idx, feature_name in enumerate(feature_columns):
        original_value = sample_frame.iloc[0, idx]
        records.append(
            {
                "特征编码": feature_name,
                "原始取值": float(original_value),
                "模型输入值": float(transformed[0, idx]),
                "贡献值": float(contributions[idx]),
                "目标类别": target_label,
                "目标类别概率": target_probability,
                "解释方法": chosen_method,
            }
        )
    if include_intercept and intercept_value is not None:
        records.append(
            {
                "特征编码": "__intercept__",
                "原始取值": 1.0,
                "模型输入值": 1.0,
                "贡献值": float(intercept_value),
                "目标类别": target_label,
                "目标类别概率": target_probability,
                "解释方法": chosen_method,
            }
        )

    explanation = pd.DataFrame(records)
    dictionary = _build_feature_metadata(feature_columns)
    explanation = explanation.merge(dictionary, on="特征编码", how="left")
    explanation["特征显示名"] = explanation["特征中文名"].fillna(explanation["特征编码"])
    intercept_mask = explanation["特征编码"] == "__intercept__"
    if intercept_mask.any():
        intercept_name = "截距" if chosen_method == "linear" else "基线贡献"
        explanation.loc[intercept_mask, "特征显示名"] = intercept_name
        explanation.loc[intercept_mask, "特征类别"] = explanation.loc[intercept_mask, "特征类别"].fillna("偏置")
    explanation["贡献绝对值"] = explanation["贡献值"].abs()
    total_abs = float(explanation["贡献绝对值"].sum())
    if total_abs > 0:
        explanation["贡献占比"] = explanation["贡献绝对值"] / total_abs
    else:
        explanation["贡献占比"] = 0.0
    explanation.sort_values(["贡献绝对值", "特征显示名"], ascending=[False, True], inplace=True)
    explanation.reset_index(drop=True, inplace=True)
    return explanation


def explain_samples(
    result: TransferResult,
    dataset: str,
    indices: Sequence[int],
    target_classes: Optional[Sequence[Optional[str]]] = None,
    method: str = "auto",
    shap_background: int = 200,
    shap_nsamples: int = 200,
    lime_num_samples: int = 5000,
    random_state: int = 42,
) -> pd.DataFrame:
    if not indices:
        return pd.DataFrame()

    dataset_lower = str(dataset).lower()
    if dataset_lower == "source":
        frame = result.source_features.reset_index(drop=True)
        dataset_label = "源域"
    elif dataset_lower == "target":
        frame = result.target_features.reset_index(drop=True)
        dataset_label = "目标域"
    else:
        raise ValueError("dataset 参数仅支持 'source' 或 'target'")

    if target_classes is None:
        target_classes = [None] * len(indices)
    elif len(target_classes) == 1 and len(indices) > 1:
        target_classes = list(target_classes) * len(indices)
    elif len(target_classes) != len(indices):
        raise ValueError("target_classes 长度需与 indices 一致。")

    explanations: List[pd.DataFrame] = []
    metadata_map = {
        "file_id": "文件编号",
        "channel": "通道",
        "rpm": "转速",
        "dataset": "数据域标识",
    }

    for index, target_cls in zip(indices, target_classes):
        if index < 0 or index >= len(frame):
            LOGGER.warning("索引 %s 超出 %s 数据集范围，跳过。", index, dataset_label)
            continue
        sample = frame.iloc[int(index)]
        explanation = explain_instance(
            result,
            sample,
            target_class=target_cls,
            method=method,
            shap_background=shap_background,
            shap_nsamples=shap_nsamples,
            lime_num_samples=lime_num_samples,
            random_state=random_state,
        )
        explanation["数据集"] = dataset_label
        explanation["样本索引"] = int(index)
        for raw_name, display_name in metadata_map.items():
            if raw_name in sample.index:
                explanation[display_name] = sample.get(raw_name)
        explanations.append(explanation)

    if not explanations:
        return pd.DataFrame()
    return pd.concat(explanations, ignore_index=True)


def cluster_local_explanations(
    explanations: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if explanations is None or explanations.empty:
        return pd.DataFrame(), pd.DataFrame()

    valid = explanations[explanations["特征编码"] != "__intercept__"]
    if valid.empty:
        return pd.DataFrame(), pd.DataFrame()

    pivot = valid.pivot_table(
        index=["数据集", "样本索引", "目标类别"],
        columns="特征编码",
        values="贡献值",
        aggfunc="sum",
        fill_value=0.0,
    )
    if pivot.empty:
        return pd.DataFrame(), pd.DataFrame()

    max_clusters = len(pivot)
    n_clusters = max(1, min(int(n_clusters), max_clusters))

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(pivot.values)

    if n_clusters == 1:
        labels = np.zeros(len(pivot), dtype=int)
        centre_values = np.mean(pivot.values, axis=0, keepdims=True)
    else:
        model = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = model.fit_predict(features_scaled)
        centre_values = scaler.inverse_transform(model.cluster_centers_)

    assignment = pivot.reset_index()[["数据集", "样本索引", "目标类别"]]
    assignment["聚类标签"] = labels.astype(int)

    centre_df = pd.DataFrame(centre_values, columns=pivot.columns)
    centre_df["聚类标签"] = np.arange(centre_values.shape[0])
    centre_long = centre_df.melt(id_vars="聚类标签", var_name="特征编码", value_name="平均贡献")

    dictionary = _build_feature_metadata(list(pivot.columns))
    centre_long = centre_long.merge(dictionary, on="特征编码", how="left")
    centre_long["特征显示名"] = centre_long["特征中文名"].fillna(centre_long["特征编码"])
    centre_long["贡献绝对值"] = centre_long["平均贡献"].abs()
    return assignment, centre_long


def plot_global_importance(result: GlobalInterpretabilityResult, output_path: Path, top_n: int = 20) -> None:
    summary = result.per_feature[result.per_feature["特征编码"] != "__intercept__"]
    summary = summary.sort_values("重要性", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("全局重要度结果为空，跳过绘图。")
        return

    configure_chinese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    categories = summary["特征类别"].fillna("未分类").astype(str)
    unique_categories = list(dict.fromkeys(categories))
    palette = plt.cm.get_cmap("tab20", max(len(unique_categories), 1))
    colour_map = {cat: palette(index) for index, cat in enumerate(unique_categories)}
    bar_colours = [colour_map[cat] for cat in categories]

    plt.figure(figsize=(10.5, max(4, top_n * 0.35)))
    contributions = summary["影响值"].to_numpy()
    max_abs = float(np.max(np.abs(contributions))) if contributions.size else 1.0
    limit = max(max_abs * 1.2, 1.0)
    plt.barh(summary["特征显示名"], contributions, color=bar_colours)
    plt.xlabel("平均影响值")
    plt.ylabel("特征")
    plt.title("全局特征影响力排序")
    plt.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    plt.xlim(-limit, limit)
    plt.gca().invert_yaxis()
    legend_handles = [Patch(facecolor=colour_map[cat], edgecolor="none", label=cat) for cat in unique_categories]
    if legend_handles:
        plt.legend(handles=legend_handles, title="特征类别", fontsize="small", loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_domain_shift(contributions: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    if contributions is None or contributions.empty:
        LOGGER.warning("域偏移贡献为空，跳过绘图。")
        return

    summary = contributions.sort_values("绝对贡献", ascending=False).head(top_n)
    if summary.empty:
        LOGGER.warning("域偏移贡献为空，跳过绘图。")
        return

    configure_chinese_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    classes = summary["类别"].astype(str)
    unique_classes = list(dict.fromkeys(classes))
    palette = plt.cm.get_cmap("Set2", max(len(unique_classes), 1))
    colour_map = {cls: palette(index) for index, cls in enumerate(unique_classes)}
    bar_colours = [colour_map[cls] for cls in classes]

    values = summary["Logit贡献"].to_numpy()
    max_abs = float(np.max(np.abs(values))) if values.size else 1.0
    limit = max(max_abs * 1.2, 1.0)

    plt.figure(figsize=(10.5, max(4, top_n * 0.35)))
    plt.barh(summary["特征显示名"], values, color=bar_colours)
    plt.xlabel("Logit 变化量")
    plt.ylabel("特征")
    plt.title("域偏移对不同类别决策的贡献")
    plt.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    plt.xlim(-limit, limit)
    plt.gca().invert_yaxis()
    legend_handles = [Patch(facecolor=colour_map[cls], edgecolor="none", label=cls) for cls in unique_classes]
    if legend_handles:
        plt.legend(handles=legend_handles, title="类别", fontsize="small", loc="lower right")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_local_explanation(explanation: pd.DataFrame, output_path: Path, top_n: int = 10) -> None:
    if explanation is None or explanation.empty:
        LOGGER.warning("局部解释结果为空，跳过绘图。")
        return

    summary = explanation.head(top_n)
    configure_chinese_font()
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, max(4, top_n * 0.45)))
    contributions = summary["贡献值"].to_numpy()
    max_abs = float(np.max(np.abs(contributions))) if contributions.size else 1.0
    limit = max(max_abs * 1.2, 1.0)
    colours = ["#d62728" if value >= 0 else "#1f77b4" for value in contributions]
    plt.barh(summary["特征显示名"], contributions, color=colours)
    plt.xlabel("对 Logit 的贡献")
    plt.ylabel("特征")
    plt.title("单样本特征贡献排序")
    plt.axvline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    plt.xlim(-limit, limit)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def plot_local_cluster_heatmap(cluster_profile: pd.DataFrame, output_path: Path, top_n: int = 12) -> None:
    if cluster_profile is None or cluster_profile.empty:
        LOGGER.warning("聚类贡献结果为空，跳过聚类热力图。")
        return

    top_features = (
        cluster_profile.groupby("特征显示名")["贡献绝对值"].max().sort_values(ascending=False).head(top_n).index
    )
    subset = cluster_profile[cluster_profile["特征显示名"].isin(top_features)]
    if subset.empty:
        LOGGER.warning("聚类贡献无显著特征，跳过热力图绘制。")
        return

    pivot = subset.pivot_table(
        index="聚类标签",
        columns="特征显示名",
        values="平均贡献",
        aggfunc="mean",
        fill_value=0.0,
    )
    if pivot.empty:
        LOGGER.warning("聚类贡献透视表为空，跳过热力图绘制。")
        return

    configure_chinese_font()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    max_abs = float(np.max(np.abs(pivot.to_numpy()))) if pivot.values.size else 1.0
    limit = max(max_abs, 1e-6)
    norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-limit, vmax=limit)

    plt.figure(figsize=(max(8, pivot.shape[1] * 0.8), max(3.5, pivot.shape[0] * 0.8)))
    im = plt.imshow(pivot.values, aspect="auto", cmap="coolwarm", norm=norm)
    plt.colorbar(im, label="平均贡献值")
    plt.xticks(np.arange(pivot.shape[1]), pivot.columns, rotation=30, ha="right")
    plt.yticks(np.arange(pivot.shape[0]), [f"聚类 {label}" for label in pivot.index])
    plt.xlabel("特征")
    plt.ylabel("聚类标签")
    plt.title("多样本贡献度聚类热力图")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


__all__ = [
    "GlobalInterpretabilityResult",
    "compute_global_feature_effects",
    "compute_domain_shift_contributions",
    "explain_instance",
    "explain_samples",
    "cluster_local_explanations",
    "plot_global_importance",
    "plot_domain_shift",
    "plot_local_explanation",
    "plot_local_cluster_heatmap",
]
