"""Source domain diagnostic modeling utilities for task 2."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

LOGGER = logging.getLogger(__name__)


DEFAULT_EXCLUDE_COLUMNS = {
    "dataset",
    "file_id",
    "file_path",
    "channel",
    "segment_index",
    "start_sample",
    "end_sample",
    "selection_score",
    "label",
    "label_code",
}


@dataclass
class TrainTestSplitConfig:
    """Configuration controlling the train/test split for the source domain."""

    test_size: float = 0.25
    random_state: int = 42
    stratify: bool = True

    def __post_init__(self) -> None:
        if not (0.0 < self.test_size < 1.0):
            raise ValueError("test_size must be a float between 0 and 1")


@dataclass
class AlignmentConfig:
    """Options controlling feature alignment via CORAL."""

    enabled: bool = True
    epsilon: float = 1e-6


@dataclass
class LogisticModelConfig:
    """Hyper-parameters for the logistic regression classifier."""

    penalty: str = "l2"
    C: float = 1.0
    solver: str = "lbfgs"
    max_iter: int = 400
    multi_class: Optional[str] = None
    class_weight: Optional[str] = None

    def create_classifier(self) -> LogisticRegression:
        kwargs = {
            "penalty": self.penalty,
            "C": self.C,
            "solver": self.solver,
            "max_iter": self.max_iter,
            "class_weight": self.class_weight,
        }
        if self.multi_class is not None:
            kwargs["multi_class"] = self.multi_class
        return LogisticRegression(**kwargs)


@dataclass
class CrossValidationConfig:
    """Cross-validation options for model robustness checks."""

    enabled: bool = True
    folds: int = 5
    shuffle: bool = True

    def create_splitter(self, random_state: int) -> StratifiedKFold:
        return StratifiedKFold(n_splits=self.folds, shuffle=self.shuffle, random_state=random_state)


@dataclass
class PermutationImportanceConfig:
    """Configuration for permutation importance analysis."""

    enabled: bool = True
    n_repeats: int = 10
    random_state: Optional[int] = 42
    scoring: Optional[str] = None


@dataclass
class SourceDiagnosisConfig:
    """Aggregate configuration for task 2 modeling."""

    label_column: str = "label"
    feature_columns: Optional[Sequence[str]] = None
    exclude_columns: Optional[Sequence[str]] = None
    alignment: AlignmentConfig = field(default_factory=AlignmentConfig)
    split: TrainTestSplitConfig = field(default_factory=TrainTestSplitConfig)
    model: LogisticModelConfig = field(default_factory=LogisticModelConfig)
    cross_validation: CrossValidationConfig = field(default_factory=CrossValidationConfig)
    permutation_importance: PermutationImportanceConfig = field(default_factory=PermutationImportanceConfig)

    def get_exclude_columns(self) -> Sequence[str]:
        if self.exclude_columns is None:
            return tuple(sorted(DEFAULT_EXCLUDE_COLUMNS))
        return self.exclude_columns


@dataclass
class TrainingResult:
    """Container for all artifacts generated during source-domain training."""

    pipeline: Pipeline
    feature_columns: List[str]
    classes: List[str]
    train_indices: List[int]
    test_indices: List[int]
    y_test: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray]
    metrics: Dict[str, float]
    classification_report: pd.DataFrame
    confusion_matrix: pd.DataFrame
    coefficient_importance: pd.DataFrame
    permutation_importance: Optional[pd.DataFrame]
    cv_scores: Optional[np.ndarray]
    alignment_diagnostics: Optional[Dict[str, float]]
    predictions: pd.DataFrame


class CoralAligner(BaseEstimator, TransformerMixin):
    """Correlation alignment (CORAL) transform to encourage domain-invariant features."""

    def __init__(self, enabled: bool = True, epsilon: float = 1e-6):
        self.enabled = enabled
        self.epsilon = epsilon

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "CoralAligner":
        X_array = np.asarray(X, dtype=float)
        if X_array.ndim != 2:
            raise ValueError("Expected a 2D array for CORAL alignment")
        self.n_features_in_ = X_array.shape[1]
        if not self.enabled:
            return self
        self.source_mean_ = X_array.mean(axis=0)
        centered = X_array - self.source_mean_
        n_samples = max(X_array.shape[0] - 1, 1)
        covariance = centered.T @ centered / n_samples
        covariance += self.epsilon * np.eye(self.n_features_in_)
        self.source_covariance_ = np.asarray(np.real_if_close(covariance), dtype=np.float64)
        whitening = linalg.fractional_matrix_power(self.source_covariance_, -0.5)
        if np.iscomplexobj(whitening):
            LOGGER.debug("CoralAligner whitening matrix has max imaginary %.3e; discarding.", np.max(np.abs(whitening.imag)))
            whitening = whitening.real
        whitening = np.real_if_close(whitening)
        self.whitening_ = np.asarray(whitening, dtype=np.float64)
        self.target_mean_ = np.zeros(self.n_features_in_)
        self.recoloring_ = np.eye(self.n_features_in_)
        return self

    def set_target_statistics(self, mean: np.ndarray, covariance: np.ndarray) -> "CoralAligner":
        if not self.enabled:
            return self
        if not hasattr(self, "n_features_in_"):
            raise RuntimeError("The aligner must be fitted before setting target statistics.")
        mean = np.asarray(mean, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if mean.shape[0] != self.n_features_in_:
            raise ValueError("Mean dimensionality does not match fitted data")
        if covariance.shape != (self.n_features_in_, self.n_features_in_):
            raise ValueError("Covariance dimensionality does not match fitted data")
        covariance = covariance + self.epsilon * np.eye(self.n_features_in_)
        covariance = np.asarray(np.real_if_close(covariance), dtype=np.float64)
        self.target_mean_ = mean
        recoloring = linalg.fractional_matrix_power(covariance, 0.5)
        if np.iscomplexobj(recoloring):
            LOGGER.debug("CoralAligner recoloring matrix has max imaginary %.3e; discarding.", np.max(np.abs(recoloring.imag)))
            recoloring = recoloring.real
        recoloring = np.real_if_close(recoloring)
        self.recoloring_ = np.asarray(recoloring, dtype=np.float64)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_array = np.asarray(X, dtype=np.float64)
        if not self.enabled:
            return X_array
        if not hasattr(self, "source_mean_"):
            raise RuntimeError("The aligner must be fitted before calling transform().")
        centered = X_array - self.source_mean_
        whitened = centered @ self.whitening_
        recolored = whitened @ self.recoloring_
        result = np.asarray(np.real_if_close(recolored), dtype=np.float64)
        return result + self.target_mean_

    # Compatibility with scikit-learn >=1.2 requires the attribute below.
    @property
    def n_features_in_(self) -> int:  # type: ignore[override]
        return self._n_features_in_

    @n_features_in_.setter
    def n_features_in_(self, value: int) -> None:
        self._n_features_in_ = value

    def get_alignment_summary(self) -> Dict[str, float]:
        if not self.enabled or not hasattr(self, "source_covariance_"):
            return {}
        whitened_cov = self.whitening_ @ self.source_covariance_ @ self.whitening_.T
        identity = np.eye(self.source_covariance_.shape[0])
        fro_error = float(np.linalg.norm(whitened_cov - identity, ord="fro"))
        condition = float(np.linalg.cond(self.source_covariance_))
        trace_value = float(np.trace(self.source_covariance_))
        return {
            "source_cov_trace": trace_value,
            "source_cov_condition_number": condition,
            "whiten_identity_fro_error": fro_error,
        }


class RealMedianImputer(BaseEstimator, TransformerMixin):
    """Impute missing values after enforcing real-valued numeric features."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> "RealMedianImputer":
        array = np.asarray(X, dtype=np.complex128)
        array = np.real_if_close(array)
        real = np.asarray(np.real(array), dtype=np.float64)
        medians = np.nanmedian(real, axis=0)
        medians = np.where(np.isnan(medians), 0.0, medians)
        self.median_ = medians
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        array = np.asarray(X, dtype=np.complex128)
        array = np.real_if_close(array)
        real = np.asarray(np.real(array), dtype=np.float64)
        if not hasattr(self, "median_"):
            return real
        inds = np.where(np.isnan(real))
        if inds[0].size:
            real[inds] = np.take(self.median_, inds[1])
        return real


def _resolve_feature_columns(data: pd.DataFrame, config: SourceDiagnosisConfig) -> List[str]:
    if config.feature_columns:
        missing = [col for col in config.feature_columns if col not in data.columns]
        if missing:
            raise KeyError(f"Feature columns missing from table: {missing}")
        return list(config.feature_columns)
    exclude = set(config.get_exclude_columns())
    feature_columns = [
        column
        for column in data.columns
        if column not in exclude and pd.api.types.is_numeric_dtype(data[column])
    ]
    if not feature_columns:
        raise ValueError("No numeric feature columns available for modeling.")
    return feature_columns


def _drop_near_constant_columns(data: pd.DataFrame, columns: Sequence[str]) -> List[str]:
    selected: List[str] = []
    dropped: List[str] = []
    for column in columns:
        unique_values = data[column].nunique(dropna=True)
        if unique_values <= 1:
            dropped.append(column)
        else:
            selected.append(column)
    if dropped:
        LOGGER.warning("Dropping near-constant columns: %s", ", ".join(sorted(dropped)))
    return selected


def _build_pipeline(config: SourceDiagnosisConfig) -> Pipeline:
    classifier = config.model.create_classifier()
    aligner = CoralAligner(enabled=config.alignment.enabled, epsilon=config.alignment.epsilon)
    steps = [
        ("imputer", RealMedianImputer()),
        ("aligner", aligner),
        ("scaler", StandardScaler()),
        ("classifier", classifier),
    ]
    return Pipeline(steps)


def _compute_alignment_diagnostics(pipeline: Pipeline) -> Optional[Dict[str, float]]:
    aligner = pipeline.named_steps.get("aligner")
    if not isinstance(aligner, CoralAligner) or not aligner.enabled:
        return None
    try:
        summary = aligner.get_alignment_summary()
        return summary
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Failed to compute alignment diagnostics: %s", exc)
        return None


def train_source_domain_model(data: pd.DataFrame, config: SourceDiagnosisConfig) -> TrainingResult:
    """Train a source-domain diagnostic model and compute evaluation artifacts."""

    if config.label_column not in data.columns:
        raise KeyError(f"Label column '{config.label_column}' not present in feature table")
    usable = data.dropna(subset=[config.label_column]).copy()
    if usable.empty:
        raise ValueError("No labeled samples available for training")

    feature_columns = _resolve_feature_columns(usable, config)
    feature_columns = _drop_near_constant_columns(usable, feature_columns)
    if not feature_columns:
        raise ValueError("No informative feature columns remain after preprocessing")

    raw_features = usable[feature_columns].to_numpy(dtype=np.complex128)
    real_features = np.real_if_close(raw_features)
    if np.iscomplexobj(real_features):
        LOGGER.warning("Complex features detected; discarding imaginary components for modeling.")
    real_features = np.asarray(np.real(real_features), dtype=np.float64)
    X = pd.DataFrame(real_features, columns=feature_columns, index=usable.index)
    y = usable[config.label_column].astype(str)

    X_values = X.to_numpy(dtype=np.float64, copy=False)
    y_values = y.to_numpy()
    indices = X.index.to_numpy()
    stratify_targets: Optional[np.ndarray] = y_values if config.split.stratify else None
    try:
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_values,
            y_values,
            indices,
            test_size=config.split.test_size,
            random_state=config.split.random_state,
            stratify=stratify_targets,
        )
    except ValueError as exc:
        LOGGER.warning("Stratified split failed (%s); retrying without stratification.", exc)
        X_train, X_test, y_train, y_test, train_idx, test_idx = train_test_split(
            X_values,
            y_values,
            indices,
            test_size=config.split.test_size,
            random_state=config.split.random_state,
            stratify=None,
        )
    def _to_native(values: np.ndarray) -> List[Any]:
        native: List[Any] = []
        for value in values:
            if hasattr(value, "item"):
                native.append(value.item())
            else:
                native.append(value)
        return native

    train_indices = _to_native(train_idx)
    test_indices = _to_native(test_idx)

    pipeline = _build_pipeline(config)
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    if hasattr(pipeline.named_steps["classifier"], "predict_proba"):
        y_proba = pipeline.predict_proba(X_test)
    else:
        y_proba = None

    test_accuracy = accuracy_score(y_test, y_pred)
    train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    report_dict = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report_dict).transpose()

    classes = list(pipeline.named_steps["classifier"].classes_)
    conf_matrix = confusion_matrix(y_test, y_pred, labels=classes)
    conf_df = pd.DataFrame(conf_matrix, index=classes, columns=classes)

    classifier = pipeline.named_steps["classifier"]
    coef_records: List[Dict[str, float]] = []
    if hasattr(classifier, "coef_"):
        for class_idx, class_name in enumerate(classes):
            for feature_idx, feature in enumerate(feature_columns):
                weight = float(classifier.coef_[class_idx, feature_idx])
                coef_records.append(
                    {
                        "class": class_name,
                        "feature": feature,
                        "coefficient": weight,
                        "odds_ratio": float(np.exp(weight)),
                        "abs_coefficient": abs(weight),
                    }
                )
        if hasattr(classifier, "intercept_"):
            for class_idx, class_name in enumerate(classes):
                intercept = float(classifier.intercept_[class_idx])
                coef_records.append(
                    {
                        "class": class_name,
                        "feature": "__intercept__",
                        "coefficient": intercept,
                        "odds_ratio": float(np.exp(intercept)),
                        "abs_coefficient": abs(intercept),
                    }
                )
    coefficients_df = pd.DataFrame(coef_records)

    perm_df: Optional[pd.DataFrame] = None
    if config.permutation_importance.enabled and len(X_test) > 0:
        try:
            importance = permutation_importance(
                pipeline,
                X_test,
                y_test,
                n_repeats=config.permutation_importance.n_repeats,
                random_state=config.permutation_importance.random_state,
                scoring=config.permutation_importance.scoring,
            )
            perm_df = pd.DataFrame(
                {
                    "feature": feature_columns,
                    "importance_mean": importance.importances_mean,
                    "importance_std": importance.importances_std,
                }
            ).sort_values("importance_mean", ascending=False)
        except ValueError as exc:
            LOGGER.warning("Permutation importance computation failed: %s", exc)

    cv_scores: Optional[np.ndarray] = None
    if config.cross_validation.enabled:
        splitter = config.cross_validation.create_splitter(config.split.random_state)
        try:
            cv_scores = cross_val_score(pipeline, X_values, y_values, cv=splitter, scoring="accuracy")
        except ValueError as exc:
            LOGGER.warning("Cross-validation failed: %s", exc)

    alignment_summary = _compute_alignment_diagnostics(pipeline)

    predictions = pd.DataFrame({"true_label": y_test, "predicted_label": y_pred}).reset_index(drop=True)
    if y_proba is not None:
        for index, class_name in enumerate(classes):
            predictions[f"probability_{class_name}"] = y_proba[:, index]

    metrics = {
        "train_accuracy": float(train_accuracy),
        "test_accuracy": float(test_accuracy),
        "macro_f1": float(macro_f1),
    }

    return TrainingResult(
        pipeline=pipeline,
        feature_columns=list(feature_columns),
        classes=classes,
        train_indices=train_indices,
        test_indices=test_indices,
        y_test=np.asarray(y_test),
        y_pred=np.asarray(y_pred),
        y_proba=y_proba,
        metrics=metrics,
        classification_report=report_df,
        confusion_matrix=conf_df,
        coefficient_importance=coefficients_df,
        permutation_importance=perm_df,
        cv_scores=cv_scores,
        alignment_diagnostics=alignment_summary,
        predictions=predictions,
    )
