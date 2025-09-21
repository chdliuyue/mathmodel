"""High level helpers for feature table analysis and visualisation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import logging

import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.segmentation import segment_signal

LOGGER = logging.getLogger(__name__)


FEATURE_PREFIXES: Sequence[str] = ("time_", "freq_", "env_", "fault_")


def load_feature_table(path: Path, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV feature table if it exists, returning an empty frame otherwise."""

    if not path.exists():
        LOGGER.warning("Feature table %s does not exist", path)
        return pd.DataFrame()

    frame = pd.read_csv(path)
    if dataset_name and "dataset" not in frame.columns:
        frame.insert(0, "dataset", dataset_name)
    return frame


def select_feature_columns(frame: pd.DataFrame, prefixes: Sequence[str] = FEATURE_PREFIXES) -> List[str]:
    """Return columns that contain engineered features rather than metadata."""

    candidates = []
    for prefix in prefixes:
        candidates.extend([col for col in frame.columns if col.startswith(prefix)])
    return sorted(set(candidates))


def prepare_combined_features(frames: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Combine feature tables from multiple domains for joint analysis."""

    non_empty = [frame for frame in frames if not frame.empty]
    if not non_empty:
        return pd.DataFrame()
    return pd.concat(non_empty, ignore_index=True)


def to_long_format(frame: pd.DataFrame, value_vars: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Convert a wide feature table into long format for plotting."""

    if frame.empty:
        return frame

    value_vars = value_vars or select_feature_columns(frame)
    id_vars = [col for col in frame.columns if col not in value_vars]
    return frame.melt(id_vars=id_vars, value_vars=value_vars, var_name="feature", value_name="value")


def compute_feature_statistics(
    frame: pd.DataFrame,
    group_by: Sequence[str] = ("dataset", "label"),
    prefixes: Sequence[str] = FEATURE_PREFIXES,
) -> pd.DataFrame:
    """Compute descriptive statistics for engineered features."""

    if frame.empty:
        return pd.DataFrame()

    feature_cols = select_feature_columns(frame, prefixes=prefixes)
    if not feature_cols:
        LOGGER.warning("No feature columns found to summarise")
        return pd.DataFrame()

    numeric = frame[feature_cols]
    filled = numeric.fillna(numeric.mean())
    group_cols = [col for col in group_by if col in frame.columns]

    if group_cols:
        grouped = frame[group_cols].fillna("unlabelled")
        grouped_frame = pd.concat([grouped, filled], axis=1)
        grouped_obj = grouped_frame.groupby(group_cols, dropna=False)
    else:
        grouped_obj = filled

    aggregations = grouped_obj.agg(["mean", "std", "var", "min", "max"])
    aggregations = aggregations.reset_index()
    aggregations.columns = ["_".join(map(str, col)).rstrip("_") for col in aggregations.columns.to_flat_index()]
    return aggregations


@dataclass
class EmbeddingResult:
    """Bundle together 2-D embeddings with the data they correspond to."""

    embedding: np.ndarray
    data: pd.DataFrame
    method: str


def _standardise_features(frame: pd.DataFrame, prefixes: Sequence[str]) -> pd.DataFrame:
    feature_cols = select_feature_columns(frame, prefixes=prefixes)
    features = frame[feature_cols].copy()
    features = features.fillna(features.mean())
    features = features.fillna(0.0)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)
    scaled_frame = pd.DataFrame(scaled, columns=feature_cols)
    return scaled_frame


def run_tsne(frame: pd.DataFrame, prefixes: Sequence[str] = FEATURE_PREFIXES, random_state: int = 42) -> Optional[EmbeddingResult]:
    """Project features to 2-D using t-SNE."""

    if frame.empty:
        LOGGER.warning("Cannot compute t-SNE embedding on an empty frame")
        return None

    scaled = _standardise_features(frame, prefixes=prefixes)
    n_samples = len(scaled)
    if n_samples < 5:
        LOGGER.warning("Not enough samples (%s) for t-SNE", n_samples)
        return None

    perplexity = min(30, max(5, (n_samples - 1) / 3))
    perplexity = min(perplexity, n_samples - 1)
    if perplexity <= 0:
        LOGGER.warning("Computed perplexity %.3f is not valid", perplexity)
        return None

    LOGGER.info("Running t-SNE on %s samples with perplexity %.2f", n_samples, perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random")
    embedding = tsne.fit_transform(scaled)
    result = EmbeddingResult(embedding=embedding, data=frame.reset_index(drop=True), method="tsne")
    return result


def run_umap(
    frame: pd.DataFrame,
    prefixes: Sequence[str] = FEATURE_PREFIXES,
    random_state: int = 42,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
) -> Optional[EmbeddingResult]:
    """Project features to 2-D using UMAP if the dependency is available."""

    try:
        import umap  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("Skipping UMAP embedding because umap-learn is not installed")
        return None

    if frame.empty:
        LOGGER.warning("Cannot compute UMAP embedding on an empty frame")
        return None

    scaled = _standardise_features(frame, prefixes=prefixes)
    n_samples = len(scaled)
    if n_samples < 5:
        LOGGER.warning("Not enough samples (%s) for UMAP", n_samples)
        return None

    if n_neighbors is None:
        n_neighbors = min(15, max(2, int(n_samples / 10)))
    n_neighbors = min(n_neighbors, n_samples - 1)

    LOGGER.info("Running UMAP on %s samples with %s neighbours", n_samples, n_neighbors)
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=random_state,
    )
    embedding = reducer.fit_transform(scaled)
    return EmbeddingResult(embedding=embedding, data=frame.reset_index(drop=True), method="umap")


def plot_embedding(result: EmbeddingResult, output_path: Path, title: Optional[str] = None) -> None:
    """Persist a scatter plot visualising an embedding."""

    import matplotlib.pyplot as plt

    data = result.data.copy().reset_index(drop=True)
    data["emb_x"] = result.embedding[:, 0]
    data["emb_y"] = result.embedding[:, 1]

    datasets = data.get("dataset") if "dataset" in data.columns else pd.Series(["unknown"] * len(data))
    labels = data.get("label").fillna("unlabelled") if "label" in data.columns else pd.Series(["unlabelled"] * len(data))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)

    for ax, colour_key, legend_title in zip(
        axes,
        [datasets, labels],
        ["dataset", "label"],
    ):
        unique_values = colour_key.unique()
        for value in unique_values:
            mask = colour_key == value
            subset = data.loc[mask]
            ax.scatter(
                subset["emb_x"],
                subset["emb_y"],
                label=str(value),
                alpha=0.7,
                edgecolor="none",
            )
        ax.set_xlabel("component 1")
        ax.set_ylabel("component 2")
        ax.legend(title=legend_title, fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.3)

    fig.suptitle(title or f"{result.method.upper()} embedding")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


@dataclass
class SignalPlotConfig:
    """Configuration controlling signal visualisations."""

    window_seconds: float = 1.0
    overlap: float = 0.5
    drop_last: bool = True
    preview_seconds: Optional[float] = 2.0


def _prepare_signal_preview(signal: np.ndarray, sampling_rate: float, preview_seconds: Optional[float]) -> np.ndarray:
    if preview_seconds is None:
        return signal
    samples = int(round(preview_seconds * sampling_rate))
    if samples <= 0:
        return signal
    return signal[: min(samples, signal.size)]


def plot_time_series_grid(
    summaries: Sequence,
    output_path: Path,
    config: SignalPlotConfig,
    max_records: int = 4,
) -> None:
    """Plot raw time-series signals for quick visual inspection."""

    import matplotlib.pyplot as plt

    if not summaries:
        LOGGER.warning("No summaries provided for time-series grid plot")
        return

    records: List = []
    for summary in summaries:
        for record in summary.records:
            records.append((summary, record))
    if not records:
        LOGGER.warning("Summaries did not contain any signal records")
        return

    records = records[:max_records]
    rows = len(records)
    fig, axes = plt.subplots(rows, 1, figsize=(12, 3 * rows), sharex=False)
    if rows == 1:
        axes = [axes]

    for ax, (summary, record) in zip(axes, records):
        sampling_rate = record.sampling_rate
        signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
        time_axis = np.arange(signal.size) / sampling_rate
        ax.plot(time_axis, signal, color="#1f77b4", linewidth=0.8)
        ax.set_title(f"{summary.file_id} – {record.channel} ({sampling_rate:.0f} Hz)")
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Amplitude")

        segments = list(
            segment_signal(
                signal,
                sampling_rate=sampling_rate,
                window_seconds=config.window_seconds,
                overlap=config.overlap,
                drop_last=config.drop_last,
            )
        )
        for segment in segments:
            start_t = segment.start / sampling_rate
            ax.axvline(start_t, color="grey", linestyle="--", alpha=0.2)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_signal_diagnostics(
    summary,
    record,
    output_path: Path,
    config: SignalPlotConfig,
) -> None:
    """Plot a combined time-domain and spectrogram view for a single signal."""

    import matplotlib.pyplot as plt

    sampling_rate = record.sampling_rate
    signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
    time_axis = np.arange(signal.size) / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].plot(time_axis, signal, color="#2ca02c", linewidth=0.8)
    axes[0].set_title(f"Time-domain signal – {summary.file_id} / {record.channel}")
    axes[0].set_xlabel("Time [s]")
    axes[0].set_ylabel("Amplitude")

    segments = list(
        segment_signal(
            signal,
            sampling_rate=sampling_rate,
            window_seconds=config.window_seconds,
            overlap=config.overlap,
            drop_last=config.drop_last,
        )
    )
    for segment in segments:
        start_t = segment.start / sampling_rate
        axes[0].axvline(start_t, color="grey", linestyle=":", alpha=0.2)

    nfft = min(4096, max(256, int(sampling_rate / 4)))
    axes[1].specgram(signal, NFFT=nfft, Fs=sampling_rate, noverlap=nfft // 2, cmap="magma")
    axes[1].set_title("Spectrogram")
    axes[1].set_xlabel("Time [s]")
    axes[1].set_ylabel("Frequency [Hz]")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


__all__ = [
    "EmbeddingResult",
    "SignalPlotConfig",
    "compute_feature_statistics",
    "load_feature_table",
    "plot_embedding",
    "plot_signal_diagnostics",
    "plot_time_series_grid",
    "prepare_combined_features",
    "run_tsne",
    "run_umap",
    "select_feature_columns",
    "to_long_format",
]
