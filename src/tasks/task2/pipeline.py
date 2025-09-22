"""Orchestration helpers for task 2 source-domain modelling."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import json
import logging

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_curve, auc
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.preprocessing import label_binarize

from ...modeling import (
    AlignmentConfig,
    CoralAligner,
    CrossValidationConfig,
    LogisticModelConfig,
    PermutationImportanceConfig,
    SourceDiagnosisConfig,
    TrainTestSplitConfig,
    TrainingResult,
    train_source_domain_model,
)
from ...analysis.feature_analysis import configure_chinese_font
from ...analysis.feature_dictionary import (
    LABEL_DISPLAY_ORDER,
    build_feature_name_map,
    build_label_name_map,
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
    benchmark_models: List[Dict[str, Any]] = field(default_factory=list)


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
    multi_class = config.get("multi_class")
    if multi_class is not None:
        multi_class = str(multi_class)
        if multi_class.lower() == "auto":
            multi_class = None
    return LogisticModelConfig(
        penalty=str(config.get("penalty", "l2")),
        C=float(config.get("C", 1.0)),
        solver=str(config.get("solver", "lbfgs")),
        max_iter=int(config.get("max_iter", 400)),
        multi_class=multi_class,
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
    benchmarks = raw_config.get("benchmarks", []) or []
    return Task2Config(
        features_config=features_config,
        alignment=alignment,
        split=split,
        model=model,
        cross_validation=cross_validation,
        permutation=permutation,
        outputs=outputs,
        benchmark_models=benchmarks,
    )


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _resolve_label_order(classes: Sequence[str]) -> List[str]:
    base = [label for label in LABEL_DISPLAY_ORDER if label in classes]
    extras = [str(label) for label in classes if label not in base]
    return base + extras


def _localise_confusion_dataframe(
    confusion_df: pd.DataFrame, class_order: Sequence[str], label_map: Dict[str, str]
) -> pd.DataFrame:
    classes = [str(label) for label in class_order]
    if confusion_df.empty:
        base = pd.DataFrame(0, index=classes, columns=classes, dtype=int)
    else:
        base = confusion_df.reindex(index=classes, columns=classes, fill_value=0)
    base = base.astype(int)
    display_index = [label_map.get(label, label) for label in classes]
    display_columns = [f"预测：{label_map.get(label, label)}" for label in classes]
    localised = base.copy()
    localised.insert(0, "真实标签编码", classes)
    localised.index = display_index
    localised.columns = ["真实标签编码"] + display_columns
    return localised


def _localise_classification_report(
    report: pd.DataFrame, label_map: Dict[str, str]
) -> pd.DataFrame:
    if report.empty:
        return report
    column_map = {
        "precision": "精确率",
        "recall": "召回率",
        "f1-score": "F1值",
        "support": "样本数",
    }
    rows: List[Dict[str, Any]] = []
    for index, row in report.iterrows():
        code = str(index)
        key_lower = code.lower()
        if code in label_map:
            name = label_map[code]
        elif key_lower == "accuracy":
            name = "总体准确率"
        elif key_lower == "macro avg":
            name = "宏平均"
        elif key_lower == "weighted avg":
            name = "加权平均"
        else:
            name = code
        record: Dict[str, Any] = {"指标编码": code, "指标名称": name}
        for column, value in row.items():
            record[column_map.get(column, column)] = value
        rows.append(record)
    return pd.DataFrame(rows)


def _localise_coefficients(
    coefficients: pd.DataFrame,
    label_map: Dict[str, str],
    feature_map: Dict[str, str],
) -> pd.DataFrame:
    if coefficients.empty:
        return coefficients
    localised = coefficients.copy()
    feature_map = dict(feature_map)
    feature_map.setdefault("__intercept__", "截距")
    localised["类别编码"] = localised["class"].astype(str)
    localised["类别"] = localised["类别编码"].map(label_map).fillna(localised["类别编码"])
    localised["特征编码"] = localised["feature"].astype(str)
    localised["特征名称"] = localised["特征编码"].map(feature_map).fillna(localised["特征编码"])
    renamed = localised.rename(
        columns={
            "coefficient": "回归系数",
            "odds_ratio": "优势比",
            "abs_coefficient": "系数绝对值",
        }
    )
    columns = ["类别编码", "类别", "特征编码", "特征名称", "回归系数", "优势比", "系数绝对值"]
    return renamed[columns]


def _localise_permutation(
    permutation: Optional[pd.DataFrame], feature_map: Dict[str, str]
) -> Optional[pd.DataFrame]:
    if permutation is None or permutation.empty:
        return permutation
    localised = permutation.copy()
    localised["特征编码"] = localised["feature"].astype(str)
    localised["特征名称"] = localised["特征编码"].map(feature_map).fillna(localised["特征编码"])
    renamed = localised.rename(
        columns={
            "importance_mean": "重要度均值",
            "importance_std": "重要度标准差",
        }
    )
    columns = ["特征编码", "特征名称", "重要度均值", "重要度标准差"]
    return renamed[columns]


def _localise_predictions(predictions: pd.DataFrame, label_map: Dict[str, str]) -> pd.DataFrame:
    if predictions.empty:
        return predictions
    rename_map: Dict[str, str] = {}
    for column in predictions.columns:
        if column == "true_label":
            rename_map[column] = "真实标签编码"
        elif column == "predicted_label":
            rename_map[column] = "预测标签编码"
        elif column.startswith("probability_"):
            label = column[len("probability_") :]
            rename_map[column] = f"概率_{label_map.get(label, label)}"
    localised = predictions.rename(columns=rename_map)
    if "真实标签编码" in localised.columns:
        localised["真实标签"] = localised["真实标签编码"].map(label_map).fillna(localised["真实标签编码"])
    if "预测标签编码" in localised.columns:
        localised["预测标签"] = localised["预测标签编码"].map(label_map).fillna(localised["预测标签编码"])
    preferred_order: List[str] = []
    for pair in (("真实标签编码", "真实标签"), ("预测标签编码", "预测标签")):
        for column in pair:
            if column in localised.columns and column not in preferred_order:
                preferred_order.append(column)
    for column in localised.columns:
        if column not in preferred_order:
            preferred_order.append(column)
    return localised[preferred_order]


def _localise_feature_summary(
    summary: pd.DataFrame, feature_map: Dict[str, str]
) -> pd.DataFrame:
    if summary.empty:
        return summary
    localised = summary.copy()
    codes = localised.index.astype(str)
    localised.insert(0, "特征编码", codes)
    localised.insert(1, "特征名称", [feature_map.get(code, code) for code in codes])
    localised.reset_index(drop=True, inplace=True)
    column_map = {
        "count": "样本数",
        "mean": "均值",
        "std": "标准差",
        "min": "最小值",
        "25%": "25%分位",
        "50%": "中位数",
        "75%": "75%分位",
        "max": "最大值",
    }
    localised = localised.rename(columns=column_map)
    return localised

def _build_benchmark_estimator(spec: Dict[str, Any], random_state: int):
    model_type = str(spec.get("type", "")).lower()
    params = dict(spec.get("params", {}) or {})
    if model_type == "random_forest":
        params.setdefault("n_estimators", 300)
        params.setdefault("random_state", random_state)
        return RandomForestClassifier(**params)
    if model_type == "gradient_boosting":
        params.setdefault("random_state", random_state)
        return GradientBoostingClassifier(**params)
    if model_type in {"svc", "svm"}:
        params.setdefault("probability", True)
        params.setdefault("random_state", random_state)
        return SVC(**params)
    if model_type in {"extra_trees", "extremely_randomized"}:
        params.setdefault("n_estimators", 400)
        params.setdefault("random_state", random_state)
        return ExtraTreesClassifier(**params)
    if model_type in {"logistic", "logistic_regression"}:
        params.setdefault("max_iter", 400)
        params.setdefault("solver", "lbfgs")
        params.setdefault("random_state", random_state)
        return LogisticRegression(**params)
    if model_type in {"knn", "k_neighbors"}:
        params.setdefault("n_neighbors", 5)
        return KNeighborsClassifier(**params)
    if model_type in {"mlp", "neural_network"}:
        params.setdefault("hidden_layer_sizes", (128,))
        params.setdefault("activation", "relu")
        params.setdefault("random_state", random_state)
        params.setdefault("max_iter", 500)
        return MLPClassifier(**params)
    if model_type in {"lda", "linear_discriminant"}:
        return LinearDiscriminantAnalysis(**params)
    if model_type in {"naive_bayes", "gaussian_nb"}:
        return GaussianNB(**params)
    raise ValueError(f"Unsupported benchmark model type: {model_type}")


def _evaluate_benchmarks(
    result: TrainingResult,
    feature_table: pd.DataFrame,
    config: Task2Config,
    output_dir: Path,
) -> None:
    if not config.benchmark_models:
        return

    label_column = str(config.features_config.get("label_column", "label"))
    if label_column not in feature_table.columns:
        LOGGER.warning("Label column %s missing; skipping benchmark comparison", label_column)
        return

    labelled = feature_table.dropna(subset=[label_column])
    if labelled.empty:
        LOGGER.warning("No labelled samples available for benchmark comparison")
        return

    try:
        train_frame = labelled.loc[result.train_indices]
        test_frame = labelled.loc[result.test_indices]
    except KeyError as exc:
        LOGGER.warning("Failed to align benchmark splits with training result: %s", exc)
        return

    feature_columns = result.feature_columns
    metrics_records: List[Dict[str, Any]] = []

    for spec in config.benchmark_models:
        name = str(spec.get("name") or spec.get("type") or "模型")
        try:
            estimator = _build_benchmark_estimator(spec, random_state=config.split.random_state)
        except Exception as exc:
            LOGGER.warning("跳过模型 %s：%s", name, exc)
            continue

        pipeline = clone(result.pipeline)
        try:
            pipeline.set_params(classifier=estimator)
        except ValueError:
            pipeline = Pipeline(
                [
                    ("imputer", pipeline.named_steps.get("imputer")),
                    ("aligner", pipeline.named_steps.get("aligner", CoralAligner())),
                    ("scaler", pipeline.named_steps.get("scaler")),
                    ("classifier", estimator),
                ]
            )

        try:
            pipeline.fit(train_frame[feature_columns], train_frame[label_column].astype(str))
            y_train_pred = pipeline.predict(train_frame[feature_columns])
            y_test_pred = pipeline.predict(test_frame[feature_columns])
        except Exception as exc:
            LOGGER.warning("模型 %s 训练失败：%s", name, exc)
            continue

        train_acc = accuracy_score(train_frame[label_column], y_train_pred)
        test_acc = accuracy_score(test_frame[label_column], y_test_pred)
        macro_f1 = f1_score(test_frame[label_column], y_test_pred, average="macro")

        params_repr = json.dumps(spec.get("params", {}), ensure_ascii=False)
        metrics_records.append(
            {
                "模型": name,
                "训练准确率": float(train_acc),
                "测试准确率": float(test_acc),
                "测试宏平均F1": float(macro_f1),
                "超参数": params_repr,
            }
        )

    if not metrics_records:
        return

    comparison = pd.DataFrame(metrics_records).sort_values("测试准确率", ascending=False)
    comparison_path = output_dir / config.outputs.get("model_comparison", "model_comparison.csv")
    comparison.to_csv(comparison_path, index=False, encoding="utf-8-sig")
    LOGGER.info("Benchmark comparison written to %s", comparison_path)


def _plot_confusion_matrix_figure(
    confusion_df: pd.DataFrame, output_path: Path, class_labels: Optional[Sequence[str]] = None
) -> None:
    if confusion_df.empty and not class_labels:
        LOGGER.warning("空的混淆矩阵，跳过可视化绘制")
        return

    configure_chinese_font()
    import matplotlib.pyplot as plt

    if class_labels:
        classes = [str(label) for label in class_labels]
        if confusion_df.empty:
            matrix_df = pd.DataFrame(0, index=classes, columns=classes, dtype=float)
        else:
            matrix_df = confusion_df.reindex(index=classes, columns=classes, fill_value=0)
    else:
        classes = [str(label) for label in confusion_df.index]
        matrix_df = confusion_df.loc[classes, classes]

    matrix = matrix_df.to_numpy(dtype=float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        normalised = np.divide(matrix, row_sums, out=np.zeros_like(matrix), where=row_sums != 0)

    label_map = build_label_name_map(classes)
    display_classes = [label_map.get(label, label) for label in classes]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    titles = ["混淆矩阵（数量）", "混淆矩阵（行归一化）"]
    datasets = [matrix, normalised]
    formats = ["{:d}", "{:.2f}"]

    for ax, data, title, value_format in zip(axes, datasets, titles, formats):
        im = ax.imshow(data, cmap="Blues", vmin=0.0)
        ax.set_xticks(np.arange(len(display_classes)))
        ax.set_xticklabels(display_classes, rotation=45, ha="right")
        ax.set_yticks(np.arange(len(display_classes)))
        ax.set_yticklabels(display_classes)
        ax.set_xlabel("预测标签")
        ax.set_ylabel("真实标签")
        ax.set_title(title)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                value = data[i, j]
                if value_format == "{:d}":
                    text = value_format.format(int(round(value)))
                else:
                    text = value_format.format(value)
                ax.text(j, i, text, ha="center", va="center", color="black", fontsize="small")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_multiclass_roc(result: TrainingResult, output_path: Path) -> None:
    if result.y_proba is None:
        LOGGER.warning("分类器未提供概率输出，跳过 ROC 曲线绘制")
        return
    if not result.classes:
        LOGGER.warning("分类标签信息缺失，无法绘制 ROC 曲线")
        return

    try:
        y_test_binarised = label_binarize(result.y_test, classes=result.classes)
    except Exception as exc:
        LOGGER.warning("无法对标签进行二值化处理：%s", exc)
        return

    if y_test_binarised.ndim == 1:
        y_test_binarised = y_test_binarised.reshape(-1, 1)

    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="grey", label="随机猜测")

    auc_scores: List[float] = []
    probabilities = np.asarray(result.y_proba, dtype=float)
    label_map = build_label_name_map(result.classes)
    for idx, class_name in enumerate(result.classes):
        y_true = y_test_binarised[:, idx]
        if np.all(y_true == 0) or np.all(y_true == 1):
            LOGGER.warning("测试集中类别 %s 缺少正负样本，跳过该曲线", class_name)
            continue
        fpr, tpr, _ = roc_curve(y_true, probabilities[:, idx])
        score = auc(fpr, tpr)
        auc_scores.append(score)
        display_name = label_map.get(str(class_name), str(class_name))
        ax.plot(fpr, tpr, linewidth=1.6, label=f"{display_name} (AUC={score:.3f})")

    if auc_scores:
        micro_fpr, micro_tpr, _ = roc_curve(y_test_binarised.ravel(), probabilities.ravel())
        micro_auc = auc(micro_fpr, micro_tpr)
        ax.plot(
            micro_fpr,
            micro_tpr,
            color="black",
            linestyle=":",
            linewidth=1.5,
            label=f"Micro平均 (AUC={micro_auc:.3f})",
        )
        macro_auc = float(np.mean(auc_scores))
        ax.text(0.6, 0.05, f"Macro平均AUC={macro_auc:.3f}", transform=ax.transAxes, fontsize="small")
    else:
        LOGGER.warning("所有类别均缺少正负样本对，无法生成 ROC 曲线")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("假阳性率 (FPR)")
    ax.set_ylabel("真正率 (TPR)")
    ax.set_title("多分类 ROC 曲线")
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right", fontsize="small")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


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
    confusion_plot_path = output_dir / outputs_config.get("confusion_matrix_plot", "confusion_matrix_heatmap.png")
    roc_plot_path = output_dir / outputs_config.get("roc_curve_plot", "roc_curves.png")

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

    label_order = _resolve_label_order(result.classes)
    label_map = build_label_name_map(label_order)
    feature_map = build_feature_name_map(list(result.feature_columns) + ["__intercept__"])
    feature_map.setdefault("__intercept__", "截距")

    report_localised = _localise_classification_report(result.classification_report, label_map)
    confusion_localised = _localise_confusion_dataframe(result.confusion_matrix, label_order, label_map)
    coefficients_localised = _localise_coefficients(result.coefficient_importance, label_map, feature_map)
    permutation_localised = _localise_permutation(result.permutation_importance, feature_map)
    predictions_localised = _localise_predictions(result.predictions, label_map)

    from joblib import dump

    dump(result.pipeline, model_path)

    feature_summary = feature_table[result.feature_columns].describe().transpose()
    summary_localised = _localise_feature_summary(feature_summary, feature_map)

    report_localised.to_csv(report_path, index=False, encoding="utf-8-sig")
    confusion_localised.to_csv(confusion_path, index=False, encoding="utf-8-sig")
    coefficients_localised.to_csv(coefficients_path, index=False, encoding="utf-8-sig")
    if permutation_localised is not None:
        permutation_localised.to_csv(permutation_path, index=False, encoding="utf-8-sig")
    predictions_localised.to_csv(predictions_path, index=False, encoding="utf-8-sig")
    summary_localised.to_csv(summary_path, index=False, encoding="utf-8-sig")
    with features_used_path.open("w", encoding="utf-8") as handle:
        for feature in result.feature_columns:
            handle.write(f"{feature},{feature_map.get(feature, feature)}\n")

    _plot_confusion_matrix_figure(result.confusion_matrix, confusion_plot_path, class_labels=label_order)
    _plot_multiclass_roc(result, roc_plot_path)


def run_training(
    raw_config: Dict[str, Any],
    feature_table_override: Optional[Path] = None,
    output_dir_override: Optional[Path] = None,
) -> Optional[TrainingResult]:
    config = resolve_task2_config(raw_config)

    table_path = Path(config.features_config.get("table_path", "artifacts/task1/source_features.csv"))
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
    _evaluate_benchmarks(result, feature_table, config, output_root)

    LOGGER.info("Training complete. Metrics saved to %s", output_root)
    return result


__all__ = [
    "Task2Config",
    "resolve_task2_config",
    "run_training",
]
