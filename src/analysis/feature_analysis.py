"""High level helpers for feature table analysis and visualisation."""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import logging

import numpy as np
import pandas as pd
from scipy.signal import hilbert, stft
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from src.feature_engineering.segmentation import segment_signal
from src.feature_engineering.bearing import BearingSpec
from src.analysis.feature_dictionary import build_feature_name_map, build_label_name_map

LOGGER = logging.getLogger(__name__)


FEATURE_PREFIXES: Sequence[str] = ("time_", "freq_", "env_", "fault_")
LOG_EPS = 1e-12
CHINESE_FONT_CANDIDATES: Sequence[str] = (
    "SimHei",
    "Microsoft YaHei",
    "Noto Sans CJK SC",
    "Noto Sans CJK JP",
    "Noto Sans CJK KR",
    "Source Han Sans SC",
    "WenQuanYi Zen Hei",
    "PingFang SC",
    "Arial Unicode MS",
)

FAULT_LABEL_MAP = {
    "FTF": "FTF 保持架频率",
    "BPFO": "BPFO 外圈",
    "BPFI": "BPFI 内圈",
    "BSF": "BSF 滚动体",
}

DATASET_DISPLAY_MAP = {
    "source": "源域",
    "target": "目标域",
    "unknown": "未知域",
}


def configure_chinese_font() -> None:
    """Configure Matplotlib to use a Chinese font when available."""

    try:
        import matplotlib
        from matplotlib import font_manager
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        LOGGER.warning("无法设置中文字体：%s", exc)
        return

    try:
        font_manager._load_fontmanager()
    except Exception as exc:  # pragma: no cover - refresh failure is non-fatal
        LOGGER.debug("刷新字体缓存失败：%s", exc)

    available_fonts = {font.name for font in font_manager.fontManager.ttflist}

    for candidate in CHINESE_FONT_CANDIDATES:
        if candidate not in available_fonts:
            try:
                path = font_manager.findfont(candidate, fallback_to_default=False)
                font_manager.fontManager.addfont(path)
                available_fonts.add(candidate)
            except Exception:  # candidate still unavailable
                if "Noto Sans CJK" in candidate:
                    system_fonts = font_manager.findSystemFonts(fontext="ttf") + font_manager.findSystemFonts(fontext="otf")
                    for font_path in system_fonts:
                        if "NotoSansCJK" in font_path or "SourceHanSans" in font_path:
                            try:
                                font_manager.fontManager.addfont(font_path)
                            except Exception:
                                continue
                    font_manager._load_fontmanager()
                    available_fonts = {font.name for font in font_manager.fontManager.ttflist}
                else:
                    continue

        if candidate in available_fonts:
            matplotlib.rcParams["font.sans-serif"] = [candidate]
            matplotlib.rcParams["font.family"] = [candidate]
            matplotlib.rcParams["axes.unicode_minus"] = False
            LOGGER.debug("使用中文字体：%s", candidate)
            break
    else:
        matplotlib.rcParams["axes.unicode_minus"] = False


def load_feature_table(path: Path, dataset_name: Optional[str] = None) -> pd.DataFrame:
    """Load a CSV feature table if it exists, returning an empty frame otherwise."""

    if not path.exists():
        LOGGER.warning("未找到特征表文件：%s", path)
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
        LOGGER.warning("未检索到可供汇总的特征列")
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
        LOGGER.warning("特征表为空，无法计算 t-SNE 嵌入")
        return None

    scaled = _standardise_features(frame, prefixes=prefixes)
    n_samples = len(scaled)
    if n_samples < 5:
        LOGGER.warning("样本数量不足（%s 条），无法计算 t-SNE", n_samples)
        return None

    perplexity = min(30, max(5, (n_samples - 1) / 3))
    perplexity = min(perplexity, n_samples - 1)
    if perplexity <= 0:
        LOGGER.warning("计算得到的困惑度 %.3f 非法", perplexity)
        return None

    LOGGER.info("正在基于 %s 条样本运行 t-SNE，困惑度为 %.2f", n_samples, perplexity)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=random_state, init="random")
    embedding = tsne.fit_transform(scaled)
    result = EmbeddingResult(embedding=embedding, data=frame.reset_index(drop=True), method="tsne")
    return result


def run_umap(
    frame: pd.DataFrame,
    prefixes: Sequence[str] = FEATURE_PREFIXES,
    random_state: Optional[int] = None,
    n_neighbors: Optional[int] = None,
    min_dist: float = 0.1,
) -> Optional[EmbeddingResult]:
    """Project features to 2-D using UMAP if the dependency is available."""

    try:
        import umap  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency
        LOGGER.warning("由于缺少 umap-learn 依赖，跳过 UMAP 嵌入计算")
        return None

    if frame.empty:
        LOGGER.warning("特征表为空，无法计算 UMAP 嵌入")
        return None

    scaled = _standardise_features(frame, prefixes=prefixes)
    n_samples = len(scaled)
    if n_samples < 5:
        LOGGER.warning("样本数量不足（%s 条），无法计算 UMAP", n_samples)
        return None

    if n_neighbors is None:
        n_neighbors = min(15, max(2, int(n_samples / 10)))
    n_neighbors = min(n_neighbors, n_samples - 1)

    LOGGER.info("正在基于 %s 条样本运行 UMAP，邻居数设为 %s", n_samples, n_neighbors)
    umap_kwargs = {
        "n_components": 2,
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "init": "random",
    }
    if random_state is not None:
        umap_kwargs["random_state"] = random_state

    reducer = umap.UMAP(**umap_kwargs)
    embedding = reducer.fit_transform(scaled)
    return EmbeddingResult(embedding=embedding, data=frame.reset_index(drop=True), method="umap")


def plot_embedding(result: EmbeddingResult, output_path: Path, title: Optional[str] = None) -> None:
    """Persist a scatter plot visualising an embedding."""

    configure_chinese_font()
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
        ["数据域", "标签"],
    ):
        unique_values = colour_key.unique()
        if legend_title == "数据域":
            translation_map = {
                str(value): DATASET_DISPLAY_MAP.get(str(value), str(value)) for value in unique_values
            }
        else:
            translation_map = build_label_name_map(unique_values)

        for value in unique_values:
            mask = colour_key == value
            subset = data.loc[mask]
            display_label = translation_map.get(str(value), str(value))
            ax.scatter(
                subset["emb_x"],
                subset["emb_y"],
                label=display_label,
                alpha=0.7,
                edgecolor="none",
            )
        ax.set_xlabel("分量 1")
        ax.set_ylabel("分量 2")
        ax.legend(title=legend_title, fontsize="small")
        ax.grid(True, linestyle="--", alpha=0.3)

    if title is None:
        method = result.method.upper()
        title = f"{method} 特征嵌入"
    fig.suptitle(title)
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


def _infer_record_label(summary, record) -> str:
    label = getattr(record, "label", None)
    if label is not None and pd.notna(label):
        return str(label)
    info = getattr(summary, "label_info", None)
    if info is not None:
        info_label = getattr(info, "label", None)
        if info_label is not None and pd.notna(info_label):
            return str(info_label)
    return "unlabelled"


def _prioritise_label_diversity(records: Sequence[Tuple]) -> List[Tuple]:
    if not records:
        return []
    label_groups: defaultdict[str, deque] = defaultdict(deque)
    for summary, record in records:
        label = _infer_record_label(summary, record)
        label_groups[label].append((summary, record))
    if len(label_groups) <= 1:
        return list(records)
    labels = sorted(label_groups.keys())
    diversified: List[Tuple] = []
    seen_pairs: set[Tuple[str, str]] = set()

    while len(diversified) < len(records):
        progressed = False
        for label in labels:
            if not label_groups[label]:
                continue
            summary, record = label_groups[label].popleft()
            key = (getattr(summary, "file_id", ""), getattr(record, "channel", ""))
            if key in seen_pairs:
                continue
            diversified.append((summary, record))
            seen_pairs.add(key)
            progressed = True
            if len(diversified) >= len(records):
                break
        if not progressed:
            break

    for label in labels:
        while label_groups[label]:
            summary, record = label_groups[label].popleft()
            key = (getattr(summary, "file_id", ""), getattr(record, "channel", ""))
            if key in seen_pairs:
                continue
            diversified.append((summary, record))
            seen_pairs.add(key)

    return diversified


def plot_time_series_grid(
    summaries: Sequence,
    output_path: Path,
    config: SignalPlotConfig,
    max_records: Optional[int] = 4,
    columns: int = 2,
) -> List[Tuple]:
    """Plot raw time-series signals for quick visual inspection."""

    configure_chinese_font()
    import matplotlib.pyplot as plt

    if not summaries:
        LOGGER.warning("未提供任何信号摘要，无法绘制时序网格图")
        return []

    records: List = []
    for summary in summaries:
        for record in summary.records:
            records.append((summary, record))
    if not records:
        LOGGER.warning("信号摘要中不包含任何记录")
        return []

    # Prioritise diversity by keeping the first occurrence of each file before truncation.
    diverse_records: List[Tuple] = []
    seen_files: set[str] = set()
    for summary, record in records:
        if summary.file_id not in seen_files:
            diverse_records.append((summary, record))
            seen_files.add(summary.file_id)
    for item in records:
        if item not in diverse_records:
            diverse_records.append(item)

    records = diverse_records
    records = _prioritise_label_diversity(records)
    limit = max_records
    if limit is not None and limit <= 0:
        limit = None

    if limit is not None and len(records) > limit:
        candidate_indices = sorted(set(np.linspace(0, len(records) - 1, num=limit, dtype=int)))
        selected: List[Tuple] = []
        seen_pairs: set[Tuple[str, str]] = set()
        for idx in candidate_indices:
            summary, record = records[idx]
            key = (summary.file_id, record.channel)
            if key in seen_pairs:
                continue
            selected.append((summary, record))
            seen_pairs.add(key)
        # Fill remaining slots if rounding caused duplicates to be skipped.
        if len(selected) < limit:
            for summary, record in records:
                key = (summary.file_id, record.channel)
                if key in seen_pairs:
                    continue
                selected.append((summary, record))
                seen_pairs.add(key)
                if len(selected) >= limit:
                    break
        records = selected[:limit]
    if columns <= 0:
        columns = 1
    rows = ceil(len(records) / columns) if records else 0
    if rows <= 0:
        rows = 1
    fig, axes = plt.subplots(rows, columns, figsize=(14, 3.2 * rows), sharex=False)
    axes = np.atleast_1d(axes).reshape(rows, columns)

    for ax, (summary, record) in zip(axes.flat, records):
        sampling_rate = record.sampling_rate
        signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
        time_axis = np.arange(signal.size) / sampling_rate
        ax.plot(time_axis, signal, color="#1f77b4", linewidth=0.8)
        label_text = summary.label_info.label if getattr(summary, "label_info", None) else None
        if label_text:
            title = f"{summary.file_id}（{label_text}）"
        else:
            title = summary.file_id
        channel_text = f"{record.channel}（{sampling_rate:.0f} Hz）"
        ax.set_title(f"{title}\n{channel_text}")
        ax.set_xlabel("时间 [秒]")
        ax.set_ylabel("振幅")

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

    # Hide unused axes if any
    flat_axes = list(axes.flat)
    for ax in flat_axes[len(records) :]:
        ax.axis("off")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return records


def plot_signal_diagnostics(
    summary,
    record,
    output_path: Path,
    config: SignalPlotConfig,
) -> None:
    """Plot a combined time-domain and spectrogram view for a single signal."""

    configure_chinese_font()
    import matplotlib.pyplot as plt

    sampling_rate = record.sampling_rate
    signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
    time_axis = np.arange(signal.size) / sampling_rate

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=False)

    axes[0].plot(time_axis, signal, color="#2ca02c", linewidth=0.8)
    axes[0].set_title(f"时域波形 – {summary.file_id} / {record.channel}")
    axes[0].set_xlabel("时间 [秒]")
    axes[0].set_ylabel("振幅")

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
    axes[1].set_title("时频谱图")
    axes[1].set_xlabel("时间 [秒]")
    axes[1].set_ylabel("频率 [Hz]")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_covariance_heatmap(
    frame: pd.DataFrame,
    output_path: Path,
    prefixes: Sequence[str] = FEATURE_PREFIXES,
    max_features: int = 40,
) -> None:
    """Plot a covariance heatmap highlighting feature interdependence."""

    if frame.empty:
        LOGGER.warning("输入数据为空，跳过协方差热图绘制")
        return

    feature_cols = select_feature_columns(frame, prefixes=prefixes)
    if not feature_cols:
        LOGGER.warning("缺少可用于协方差热图的特征列")
        return

    features = frame[feature_cols].copy()
    variances = features.var(axis=0, ddof=0)
    variances = variances[variances > 1e-12].sort_values(ascending=False)
    if variances.empty:
        LOGGER.warning("候选特征几乎为常数，跳过协方差热图绘制")
        return
    selected_cols = variances.index[:max_features]
    selected = features[selected_cols].fillna(features[selected_cols].mean()).to_numpy(dtype=float)
    if selected.size == 0:
        LOGGER.warning("筛选后的特征子集为空，跳过协方差热图绘制")
        return
    correlation = np.corrcoef(selected, rowvar=False)

    configure_chinese_font()
    import matplotlib.pyplot as plt

    size = max(6.0, min(0.45 * len(selected_cols), 18.0))
    fig, ax = plt.subplots(figsize=(size, size))
    im = ax.imshow(correlation, cmap="viridis", aspect="equal", vmin=-1.0, vmax=1.0)
    ax.set_xticks(np.arange(len(selected_cols)))
    ax.set_xticklabels(selected_cols, rotation=90)
    ax.set_yticks(np.arange(len(selected_cols)))
    ax.set_yticklabels(selected_cols)
    ax.set_title("特征相关性热图（皮尔逊）")
    ax.set_xlabel("特征索引")
    ax.set_ylabel("特征索引")
    fig.colorbar(im, ax=ax, shrink=0.8, label="相关系数")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_envelope_spectrum(
    signal_array: np.ndarray,
    sampling_rate: float,
    max_frequency: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if signal_array.size == 0:
        return np.asarray([]), np.asarray([])
    analytic = hilbert(signal_array - np.mean(signal_array))
    envelope = np.abs(analytic)
    if envelope.size == 0:
        return np.asarray([]), np.asarray([])

    mean_value = float(np.mean(envelope)) if envelope.size else 0.0
    if not np.isclose(mean_value, 0.0):
        envelope = envelope - mean_value

    window = np.hanning(len(envelope)) if envelope.size > 1 else np.ones_like(envelope)
    spectrum = np.abs(np.fft.rfft(envelope * window))
    freqs = np.fft.rfftfreq(envelope.size, d=1.0 / sampling_rate)
    if max_frequency is not None and freqs.size:
        mask = freqs <= max_frequency
        freqs = freqs[mask]
        spectrum = spectrum[mask]
    return freqs, spectrum


def summarise_fault_frequency_response(
    freqs: np.ndarray,
    spectrum: np.ndarray,
    bearing: BearingSpec,
    rpm: float,
    tolerance: float = 5.0,
) -> pd.DataFrame:
    records: List[dict[str, float | str]] = []
    if freqs.size == 0 or spectrum.size == 0:
        return pd.DataFrame(records)
    try:
        fault_freqs = bearing.fault_frequencies(rpm)
    except ValueError as exc:
        LOGGER.warning("计算故障特征频率失败：%s", exc)
        return pd.DataFrame(records)

    for key, frequency in fault_freqs.items():
        if frequency <= 0 or not np.isfinite(frequency) or frequency > freqs.max():
            continue
        idx = int(np.argmin(np.abs(freqs - frequency)))
        predicted_amplitude = float(spectrum[idx])
        mask = (freqs >= max(0.0, frequency - tolerance)) & (freqs <= frequency + tolerance)
        if np.any(mask):
            local_indices = np.where(mask)[0]
            local_peak = int(local_indices[np.argmax(spectrum[mask])])
        else:
            local_peak = idx
        peak_frequency = float(freqs[local_peak])
        peak_amplitude = float(spectrum[local_peak])
        records.append(
            {
                "fault": key.upper(),
                "predicted_frequency": float(frequency),
                "peak_frequency": peak_frequency,
                "predicted_amplitude": predicted_amplitude,
                "peak_amplitude": peak_amplitude,
                "frequency_error": float(peak_frequency - frequency),
            }
        )
    return pd.DataFrame(records)


def plot_envelope_spectrum(
    summary,
    record,
    output_path: Path,
    config: SignalPlotConfig,
    max_frequency: Optional[float] = None,
    bearing: Optional[BearingSpec] = None,
    rpm: Optional[float] = None,
    tolerance: float = 5.0,
) -> pd.DataFrame:
    """绘制希尔伯特包络谱，突出滚动轴承特征频率。"""

    configure_chinese_font()
    import matplotlib.pyplot as plt

    sampling_rate = record.sampling_rate
    signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
    if signal.size == 0:
        LOGGER.warning("Signal preview为空，无法绘制包络谱")
        return

    freqs, spectrum = compute_envelope_spectrum(signal, sampling_rate, max_frequency=max_frequency)
    if freqs.size == 0:
        LOGGER.warning("包络谱计算结果为空，跳过绘图")
        return pd.DataFrame()

    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - register 3D projection

    magnitude = np.maximum(spectrum, LOG_EPS)
    max_magnitude = float(np.max(magnitude)) if magnitude.size else 0.0
    if max_magnitude > 0.0:
        display_spectrum = 20.0 * np.log10(magnitude / max_magnitude)
    else:
        display_spectrum = np.zeros_like(magnitude)

    downsample = max(1, int(np.ceil(display_spectrum.size / 600)))
    freq_sampled = freqs[::downsample]
    spec_sampled = display_spectrum[::downsample]
    if freq_sampled.size == 0:
        freq_sampled = freqs
        spec_sampled = display_spectrum
    elif freq_sampled[-1] != freqs[-1]:
        freq_sampled = np.append(freq_sampled, freqs[-1])
        spec_sampled = np.append(spec_sampled, display_spectrum[-1])

    ridge_depth = np.linspace(0.0, 1.0, 32)
    surface_x, surface_y = np.meshgrid(freq_sampled, ridge_depth)
    surface_z = np.tile(spec_sampled, (ridge_depth.size, 1))

    fig = plt.figure(figsize=(10.5, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    surface = ax.plot_surface(
        surface_x,
        surface_y,
        surface_z,
        cmap="viridis",
        linewidth=0.0,
        antialiased=True,
        alpha=0.88,
    )

    crest_y = float(ridge_depth.max() + 0.12)
    ax.plot(freq_sampled, np.full_like(freq_sampled, crest_y), spec_sampled, color="#d62728", linewidth=1.6)

    display_min = float(display_spectrum.min()) if display_spectrum.size else 0.0
    display_max = float(display_spectrum.max()) if display_spectrum.size else 0.0
    freq_min = float(freqs.min()) if freqs.size else 0.0
    freq_max = float(freqs.max()) if freqs.size else 1.0
    freq_range = max(freq_max - freq_min, 1.0)
    depth_range = crest_y + 0.4
    amplitude_span = max(display_max - display_min, 1.0)
    z_padding = max(amplitude_span * 0.15, 3.0)
    z_lower = display_min - z_padding
    z_upper = display_max + z_padding

    target_ratio = 0.18
    vertical_scale = max(1.0, (freq_range * target_ratio) / max(amplitude_span, 1e-6))
    adjusted_amplitude = amplitude_span * vertical_scale

    ax.set_title(f"三维包络谱 – {summary.file_id} / {record.channel}")
    ax.set_xlabel("频率 [Hz]")
    ax.set_ylabel("虚拟深度")
    ax.set_zlabel("包络幅值 [dB]")
    ax.set_xlim(freq_min, freq_max)
    ax.set_ylim(-0.05, crest_y + 0.35)
    ax.set_zlim(z_lower, z_upper)
    ax.set_box_aspect((freq_range, depth_range, adjusted_amplitude))
    ax.set_yticks([])
    ax.view_init(elev=28, azim=-120)

    fig.colorbar(surface, ax=ax, shrink=0.55, pad=0.12, label="相对包络幅值 [dB]")
    fig.text(
        0.02,
        0.94,
        "包络幅值已归一化到峰值，沿虚拟深度复制以便观察频率特征",
        fontsize="small",
        color="#555555",
    )

    summary_df = pd.DataFrame()
    label_positions: List[float] = []
    amplitude_span = float(np.ptp(display_spectrum)) if display_spectrum.size else 1.0
    vertical_spacing = max(amplitude_span * 0.08, 0.2)
    annotation_y = crest_y + 0.18
    if bearing is not None and rpm is not None:
        summary_df = summarise_fault_frequency_response(freqs, spectrum, bearing, float(rpm), tolerance=tolerance)
        z_upper_limit = z_upper - vertical_spacing * 0.5
        for _, row in summary_df.iterrows():
            freq_value = row.get("predicted_frequency")
            peak_frequency = row.get("peak_frequency")
            peak_amplitude = row.get("peak_amplitude")
            if freq_value is None or not np.isfinite(freq_value):
                continue
            if freq_value > freqs.max():
                continue
            label = FAULT_LABEL_MAP.get(str(row.get("fault", "")).upper(), str(row.get("fault", "")))
            base_x = float(freq_value)
            if peak_frequency is not None and peak_amplitude is not None and np.isfinite(peak_amplitude):
                scatter_x = float(peak_frequency)
                if max_magnitude > 0.0:
                    base_z = float(20.0 * np.log10(max(peak_amplitude, LOG_EPS) / max_magnitude))
                else:
                    base_z = float(display_max)
            else:
                scatter_x = base_x
                base_z = float(display_max) if display_spectrum.size else 0.0

            label_z = base_z
            while any(abs(label_z - prev) < vertical_spacing for prev in label_positions):
                label_z += vertical_spacing
            if label_z > z_upper_limit:
                label_z = z_upper_limit
                while (
                    any(abs(label_z - prev) < vertical_spacing for prev in label_positions)
                    and label_z > z_lower + vertical_spacing
                ):
                    label_z -= vertical_spacing
            label_positions.append(label_z)

            ax.plot(
                [base_x, base_x],
                [0.0, annotation_y - 0.06],
                [base_z, base_z],
                color="#1f77b4",
                linestyle="--",
                linewidth=0.9,
                alpha=0.7,
            )
            ax.scatter([scatter_x], [0.0], [base_z], color="#ff7f0e", s=42, depthshade=True, zorder=5)
            ax.text(
                base_x,
                annotation_y,
                label_z,
                label,
                color="#1f77b4",
                fontsize="small",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)
    return summary_df


def plot_window_sequence(
    summary,
    record,
    output_path: Path,
    config: SignalPlotConfig,
) -> None:
    """Plot the RMS evolution over sliding windows to reveal temporal trends."""

    configure_chinese_font()
    import matplotlib.pyplot as plt

    sampling_rate = record.sampling_rate
    signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
    segments = list(
        segment_signal(
            signal,
            sampling_rate=sampling_rate,
            window_seconds=config.window_seconds,
            overlap=config.overlap,
            drop_last=config.drop_last,
        )
    )

    if not segments:
        LOGGER.warning("未能生成任何分段，跳过窗序折线绘制")
        return

    indices = [segment.index for segment in segments]
    centers = [((segment.start + segment.end) / 2) / sampling_rate for segment in segments]
    rms_values = [float(np.sqrt(np.mean(segment.data**2))) for segment in segments]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(centers, rms_values, marker="o", color="#9467bd", linewidth=1.0)
    ax.set_title(f"窗序均方根趋势 – {summary.file_id} / {record.channel}")
    ax.set_xlabel("时间 [秒]")
    ax.set_ylabel("均方根 (RMS)")
    ax.grid(True, linestyle="--", alpha=0.3)

    secondary = ax.secondary_xaxis("top")
    secondary.set_xticks(centers)
    secondary.set_xticklabels([str(idx) for idx in indices])
    secondary.set_xlabel("窗口序号")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_time_frequency_saliency(
    summary,
    record,
    output_path: Path,
    config: SignalPlotConfig,
    nperseg: Optional[int] = None,
) -> None:
    """绘制时频显著性热图，突出能量异常集中的区域。"""

    configure_chinese_font()
    import matplotlib.pyplot as plt

    sampling_rate = record.sampling_rate
    signal = _prepare_signal_preview(record.signal, sampling_rate, config.preview_seconds)
    if signal.size == 0:
        LOGGER.warning("Signal preview为空，无法绘制时频显著性热图")
        return

    if nperseg is None:
        nperseg = min(len(signal), max(256, int(sampling_rate * config.window_seconds)))
    nperseg = max(128, nperseg)
    noverlap = int(nperseg * config.overlap)

    freqs, times, stft_values = stft(signal, fs=sampling_rate, window="hann", nperseg=nperseg, noverlap=noverlap)
    magnitude = np.abs(stft_values)
    if magnitude.size == 0:
        LOGGER.warning("STFT结果为空，跳过时频显著性绘制")
        return

    log_mag = np.log1p(magnitude)
    freq_mean = log_mag.mean(axis=1, keepdims=True)
    freq_std = log_mag.std(axis=1, keepdims=True) + 1e-6
    time_mean = log_mag.mean(axis=0, keepdims=True)
    time_std = log_mag.std(axis=0, keepdims=True) + 1e-6
    freq_score = (log_mag - freq_mean) / freq_std
    time_score = (log_mag - time_mean) / time_std
    combined = np.maximum(freq_score + time_score, 0.0)
    scale = np.percentile(combined, 99) + 1e-6
    saliency = np.clip(combined / scale, 0.0, 1.0)
    saliency = np.power(saliency, 0.9)

    fig, ax = plt.subplots(figsize=(10, 4))
    mesh = ax.pcolormesh(times, freqs, saliency, shading="auto", cmap="YlOrRd", vmin=0.0, vmax=1.0)
    ax.set_title(f"时频显著性热图 – {summary.file_id} / {record.channel}")
    ax.set_xlabel("时间 [秒]")
    ax.set_ylabel("频率 [Hz]")
    fig.colorbar(mesh, ax=ax, label="显著性 (归一化z-score)")
    ax.text(
        0.02,
        0.94,
        "基于log幅值与双向z-score归一化",
        transform=ax.transAxes,
        fontsize="small",
        color="#555555",
    )

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def compute_feature_importance(
    frame: pd.DataFrame,
    output_path: Path,
    prefixes: Sequence[str] = FEATURE_PREFIXES,
    top_n: int = 20,
) -> Optional[pd.DataFrame]:
    """Estimate feature importance using a随机森林分类器."""

    if frame.empty:
        LOGGER.warning("空特征表，无法估计特征重要度")
        return None

    if "label" not in frame.columns:
        LOGGER.warning("特征表缺少标签列，无法计算特征重要度")
        return None

    labelled = frame.dropna(subset=["label"])
    if labelled.empty or labelled["label"].nunique() < 2:
        LOGGER.warning("标签类别不足，跳过特征重要度分析")
        return None

    feature_cols = select_feature_columns(labelled, prefixes=prefixes)
    if not feature_cols:
        LOGGER.warning("没有可用特征列，跳过特征重要度分析")
        return None

    features = labelled[feature_cols].fillna(0.0)
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    try:
        from sklearn.ensemble import RandomForestClassifier
    except ImportError as exc:  # pragma: no cover - dependency should exist
        LOGGER.warning("无法导入RandomForestClassifier：%s", exc)
        return None

    classifier = RandomForestClassifier(n_estimators=300, random_state=42, class_weight="balanced_subsample")
    classifier.fit(features_scaled, labelled["label"].astype(str))

    importances = classifier.feature_importances_
    top_indices = np.argsort(importances)[::-1][:top_n]
    top_features = [feature_cols[i] for i in top_indices]
    top_values = importances[top_indices]
    name_map = build_feature_name_map(top_features)
    chinese_names = [name_map.get(feature, feature) for feature in top_features]

    importance_df = pd.DataFrame(
        {"特征编码": top_features, "特征名称": chinese_names, "重要度": top_values}
    )

    configure_chinese_font()
    import matplotlib.pyplot as plt

    height = max(4.0, len(top_features) * 0.4)
    fig, ax = plt.subplots(figsize=(10, height))
    ax.barh(range(len(top_features)), top_values[::-1], color="#17becf")
    ax.set_yticks(range(len(top_features)))
    ax.set_yticklabels(chinese_names[::-1])
    ax.invert_yaxis()
    ax.set_xlabel("重要度")
    ax.set_title("特征重要度（随机森林）")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    return importance_df


def _prepare_pairwise_scaled(
    left: pd.DataFrame, right: pd.DataFrame, prefixes: Sequence[str]
) -> Tuple[np.ndarray, np.ndarray]:
    feature_cols = select_feature_columns(left, prefixes=prefixes)
    feature_cols = [col for col in feature_cols if col in right.columns]
    if not feature_cols:
        raise ValueError("无法匹配到共同特征列")

    left_values = left[feature_cols].fillna(0.0)
    right_values = right[feature_cols].fillna(0.0)

    scaler = StandardScaler()
    combined = np.vstack([left_values.values, right_values.values])
    scaled = scaler.fit_transform(combined)
    left_scaled = scaled[: len(left_values)]
    right_scaled = scaled[len(left_values) :]
    return left_scaled, right_scaled


def maximum_mean_discrepancy(left: np.ndarray, right: np.ndarray) -> float:
    """Compute the unbiased RBF-kernel MMD between two samples."""

    if left.size == 0 or right.size == 0:
        return float("nan")

    combined = np.vstack([left, right])
    diff = combined[:, None, :] - combined[None, :, :]
    sq_dists = np.sum(diff**2, axis=2)
    upper = sq_dists[np.triu_indices_from(sq_dists, k=1)]
    median_sq = np.median(upper) if upper.size else 1.0
    gamma = 1.0 / (2.0 * median_sq + 1e-12)

    def _kernel(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        diff_xy = x[:, None, :] - y[None, :, :]
        return np.exp(-gamma * np.sum(diff_xy**2, axis=2))

    k_xx = _kernel(left, left)
    k_yy = _kernel(right, right)
    k_xy = _kernel(left, right)

    m = left.shape[0]
    n = right.shape[0]
    term_xx = (np.sum(k_xx) - np.trace(k_xx)) / (m * (m - 1)) if m > 1 else 0.0
    term_yy = (np.sum(k_yy) - np.trace(k_yy)) / (n * (n - 1)) if n > 1 else 0.0
    term_xy = (2.0 * np.sum(k_xy)) / (m * n)
    return float(term_xx + term_yy - term_xy)


def coral_distance(left: np.ndarray, right: np.ndarray) -> float:
    """Compute CORAL distance (squared Frobenius norm of covariance difference)."""

    if left.size == 0 or right.size == 0:
        return float("nan")

    cov_left = np.cov(left, rowvar=False) + np.eye(left.shape[1]) * 1e-6
    cov_right = np.cov(right, rowvar=False) + np.eye(right.shape[1]) * 1e-6
    diff = cov_left - cov_right
    return float(np.linalg.norm(diff, ord="fro") ** 2)


def compute_domain_alignment_metrics(
    frame: pd.DataFrame,
    prefixes: Sequence[str] = FEATURE_PREFIXES,
) -> pd.DataFrame:
    """Compute MMD 和 CORAL 指标评估源/目标域的分布对齐程度。"""

    if frame.empty or "dataset" not in frame.columns:
        return pd.DataFrame()

    groups = frame.groupby("dataset")
    dataset_names = list(groups.groups.keys())
    if len(dataset_names) < 2:
        return pd.DataFrame()

    results: List[dict[str, float | str | int]] = []
    for i in range(len(dataset_names)):
        for j in range(i + 1, len(dataset_names)):
            left_name = dataset_names[i]
            right_name = dataset_names[j]
            left_frame = groups.get_group(left_name)
            right_frame = groups.get_group(right_name)
            try:
                left_scaled, right_scaled = _prepare_pairwise_scaled(left_frame, right_frame, prefixes=prefixes)
            except ValueError as exc:
                LOGGER.warning("跳过域对齐计算：%s", exc)
                continue

            mmd_value = maximum_mean_discrepancy(left_scaled, right_scaled)
            coral_value = coral_distance(left_scaled, right_scaled)

            results.append(
                {
                    "域A": str(left_name),
                    "域B": str(right_name),
                    "样本数A": int(left_scaled.shape[0]),
                    "样本数B": int(right_scaled.shape[0]),
                    "MMD": float(mmd_value),
                    "CORAL": float(coral_value),
                }
            )

    return pd.DataFrame(results)


def translate_statistics_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """将统计汇总表的列名翻译为中文，便于报告展示。"""

    if frame.empty:
        return frame

    prefix_map = {
        "time_": "时域_",
        "freq_": "频域_",
        "env_": "包络域_",
        "fault_": "故障带_",
    }
    suffix_map = {
        "_mean": "_均值",
        "_std": "_标准差",
        "_var": "_方差",
        "_min": "_最小值",
        "_max": "_最大值",
    }
    direct_map = {"dataset": "数据集", "label": "标签"}

    renamed = {}
    for column in frame.columns:
        new_name = direct_map.get(column, column)
        if new_name == column:
            for prefix, translated in prefix_map.items():
                if new_name.startswith(prefix):
                    new_name = translated + new_name[len(prefix) :]
                    break
            for suffix, translated in suffix_map.items():
                if new_name.endswith(suffix):
                    new_name = new_name[: -len(suffix)] + translated
                    break
        renamed[column] = new_name

    return frame.rename(columns=renamed)


__all__ = [
    "EmbeddingResult",
    "SignalPlotConfig",
    "compute_feature_statistics",
    "compute_domain_alignment_metrics",
    "compute_feature_importance",
    "load_feature_table",
    "plot_covariance_heatmap",
    "plot_embedding",
    "plot_envelope_spectrum",
    "plot_time_frequency_saliency",
    "plot_signal_diagnostics",
    "plot_time_series_grid",
    "plot_window_sequence",
    "prepare_combined_features",
    "run_tsne",
    "run_umap",
    "select_feature_columns",
    "translate_statistics_columns",
    "to_long_format",
]
