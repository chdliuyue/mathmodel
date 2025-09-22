"""Run the transfer diagnosis pipeline (task 3)."""
from __future__ import annotations

import argparse
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from joblib import dump
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.feature_analysis import configure_chinese_font, plot_embedding, run_tsne
from src.analysis.feature_dictionary import build_feature_dictionary
from src.tasks.task3 import (
    TimeFrequencyConfig,
    TransferConfig,
    parse_transfer_config,
    run_transfer_learning,
)
from src.tasks.task3.features import compute_multimodal_representation
from src.tasks.task3.segment_loader import SegmentFetcher

LOGGER = logging.getLogger("task3_transfer")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


MODAL_NAME_MAP = {
    "time": "时域",
    "stft": "短时傅里叶",
    "cwt": "连续小波",
    "mel": "梅尔",
}
COLUMN_NAME_MAP = {
    "iteration": "迭代轮次",
    "new_samples": "新增伪标签数",
    "cumulative_pseudo": "累计伪标签数",
    "threshold": "概率阈值",
    "consistency_threshold": "一致性阈值",
    "probability_mean": "平均置信度",
    "probability_min": "最小置信度",
    "probability_max": "最大置信度",
    "consistency_mean": "平均一致性",
    "consistency_min": "最小一致性",
    "consistency_max": "最大一致性",
    "row_index": "目标索引",
    "predicted_label": "模型预测标签",
    "max_probability": "最大置信度",
    "pseudo_probability": "伪标签置信度",
    "pseudo_iteration": "伪标签轮次",
    "dataset": "数据域",
    "consistency_score": "一致性得分",
    "file_id": "文件编号",
    "channel": "通道",
    "segment_index": "分段序号",
    "start_sample": "起始样本点",
    "end_sample": "结束样本点",
    "rpm": "转速",
    "time_vote": "时域模态投票",
    "stft_vote": "STFT模态投票",
    "cwt_vote": "CWT模态投票",
    "mel_vote": "梅尔模态投票",
    "time_distance": "时域距离",
    "stft_distance": "STFT距离",
    "cwt_distance": "CWT距离",
    "mel_distance": "梅尔距离",
    "time_agree": "时域一致(0/1)",
    "stft_agree": "STFT一致(0/1)",
    "cwt_agree": "CWT一致(0/1)",
    "mel_agree": "梅尔一致(0/1)",
}


def _translate_column_name(column: str) -> str:
    mapped = COLUMN_NAME_MAP.get(column)
    if mapped is not None:
        return mapped
    if column.startswith("probability_"):
        label = column[len("probability_") :]
        return f"类别概率_{label}"
    if column.endswith("_vote"):
        prefix = column[: -len("_vote")]
        modal = MODAL_NAME_MAP.get(prefix, prefix)
        return f"{modal}模态投票"
    if column.endswith("_distance"):
        prefix = column[: -len("_distance")]
        modal = MODAL_NAME_MAP.get(prefix, prefix)
        return f"{modal}距离"
    if column.endswith("_agree"):
        prefix = column[: -len("_agree")]
        modal = MODAL_NAME_MAP.get(prefix, prefix)
        return f"{modal}一致(0/1)"
    return column


def _translate_columns(frame: pd.DataFrame) -> pd.DataFrame:
    if frame is None or frame.empty:
        return frame
    renamed = {column: _translate_column_name(column) for column in frame.columns}
    return frame.rename(columns=renamed)


def _build_transfer_config(raw: Dict[str, Any]) -> TransferConfig:
    return parse_transfer_config(raw)


def _write_dataframe(frame: pd.DataFrame, path: Path, translate: bool = True) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if translate:
        frame = _translate_columns(frame)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_pseudo_history(history: Optional[List[Dict[str, float]]], path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    frame = frame.sort_values("iteration")
    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(9.5, 5.2))
    iterations = frame["iteration"].to_numpy()
    bars = ax1.bar(iterations, frame["new_samples"], color="#4c72b0", alpha=0.75, label="新增伪标签数")
    ax1.plot(iterations, frame["cumulative_pseudo"], color="#dd8452", marker="o", linewidth=1.6, label="累计伪标签数")
    ax1.set_xlabel("迭代轮次")
    ax1.set_ylabel("样本数量")
    ax1.set_xticks(iterations)
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    if "probability_mean" in frame.columns:
        ax2.plot(
            iterations,
            frame["probability_mean"],
            color="#55a868",
            marker="s",
            linewidth=1.4,
            label="平均置信度",
        )
    if "consistency_mean" in frame.columns:
        ax2.plot(
            iterations,
            frame["consistency_mean"],
            color="#c44e52",
            marker="^",
            linewidth=1.4,
            label="平均一致性",
        )
    ax2.set_ylabel("概率/一致性")
    if "threshold" in frame.columns:
        ax2.axhline(frame["threshold"].iloc[0], color="#55a868", linestyle="--", linewidth=1.0, alpha=0.6, label="置信度阈值")
    if "consistency_threshold" in frame.columns:
        ax2.axhline(
            frame["consistency_threshold"].iloc[0],
            color="#c44e52",
            linestyle=":",
            linewidth=1.0,
            alpha=0.7,
            label="一致性阈值",
        )

    handles1, labels1 = ax1.get_legend_handles_labels()
    handles2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles1 + handles2, labels1 + labels2, loc="upper left")

    fig.suptitle("伪标签迭代与多模态一致性")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_time_frequency_distribution(result, path: Path) -> None:
    features = [
        feat
        for feat in result.time_frequency_features
        if feat in result.source_features.columns and feat in result.target_features.columns
    ]
    if not features:
        return

    source = result.source_features
    target = result.target_features
    diffs: List[Tuple[str, float]] = []
    for feature in features:
        source_mean = source[feature].dropna().astype(float)
        target_mean = target[feature].dropna().astype(float)
        if source_mean.empty and target_mean.empty:
            continue
        diff = abs(source_mean.mean() - target_mean.mean())
        diffs.append((feature, diff))
    if not diffs:
        return

    diffs.sort(key=lambda item: item[1], reverse=True)
    top_features = [name for name, _ in diffs[: min(6, len(diffs))]]

    records: List[Dict[str, float | str]] = []
    for feature in top_features:
        src_values = source[feature].dropna().astype(float)
        tgt_values = target[feature].dropna().astype(float)
        if src_values.empty and tgt_values.empty:
            continue
        records.append(
            {
                "feature": feature,
                "domain": "源域",
                "mean": float(src_values.mean()),
                "std": float(src_values.std(ddof=1) if src_values.size > 1 else 0.0),
            }
        )
        records.append(
            {
                "feature": feature,
                "domain": "目标域",
                "mean": float(tgt_values.mean()),
                "std": float(tgt_values.std(ddof=1) if tgt_values.size > 1 else 0.0),
            }
        )
    if not records:
        return

    frame = pd.DataFrame(records)
    dictionary = build_feature_dictionary(top_features)
    mapping = dictionary.set_index("feature")["chinese_name"].to_dict()
    display_names = [mapping.get(feature, feature) for feature in top_features]
    src = frame[frame["domain"] == "源域"].set_index("feature").reindex(top_features)
    tgt = frame[frame["domain"] == "目标域"].set_index("feature").reindex(top_features)
    mean_diff = tgt["mean"].fillna(0.0) - src["mean"].fillna(0.0)
    relative_diff = np.divide(
        mean_diff,
        np.where(np.abs(src["mean"].fillna(0.0)) > 1e-9, np.abs(src["mean"].fillna(0.0)), 1.0),
    )

    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, (ax_bar, ax_line) = plt.subplots(
        2,
        1,
        figsize=(10.5, 6.4),
        sharex=True,
        gridspec_kw={"height_ratios": [2.2, 1.2]},
    )
    x = np.arange(len(top_features))
    width = 0.35

    ax_bar.bar(x - width / 2, src["mean"], width, yerr=src["std"], capsize=4, label="源域均值", color="#4c72b0")
    ax_bar.bar(x + width / 2, tgt["mean"], width, yerr=tgt["std"], capsize=4, label="目标域均值", color="#dd8452")
    ax_bar.set_ylabel("统计均值")
    ax_bar.set_title("时频特征源/目标域分布对比")
    ax_bar.grid(True, linestyle="--", alpha=0.3)
    ax_bar.legend()

    ax_line.plot(x, mean_diff, color="#55a868", marker="o", linewidth=1.5, label="均值差值 (目标-源)")
    ax_line.bar(x, relative_diff, color="#c44e52", alpha=0.4, label="相对差值")
    ax_line.axhline(0.0, color="#444444", linewidth=0.8, linestyle="--")
    ax_line.set_ylabel("差异")
    ax_line.set_xlabel("特征")
    ax_line.grid(True, linestyle="--", alpha=0.3)
    ax_line.legend(loc="upper right")

    ax_line.set_xticks(x)
    ax_line.set_xticklabels(display_names, rotation=25, ha="right")

    fig.tight_layout(h_pad=0.6)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _normalise_energy_matrix(matrix: np.ndarray) -> np.ndarray:
    data = np.asarray(matrix, dtype=float)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    data = np.maximum(data, 0.0)
    data = np.log1p(data)
    max_value = float(np.max(data)) if data.size else 0.0
    if max_value > 0:
        data = data / max_value
    return data


def _resize_matrix(matrix: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    rows, cols = target_shape
    if rows <= 0 or cols <= 0:
        return np.zeros(target_shape, dtype=float)
    source = np.asarray(matrix, dtype=float)
    if source.ndim != 2 or source.size == 0:
        return np.zeros(target_shape, dtype=float)
    row_indices = np.linspace(0, source.shape[0] - 1, rows)
    col_indices = np.linspace(0, source.shape[1] - 1, cols)
    intermediate = np.zeros((rows, source.shape[1]), dtype=float)
    for idx, position in enumerate(row_indices):
        lower = int(np.floor(position))
        upper = min(lower + 1, source.shape[0] - 1)
        weight = position - lower
        if lower == upper:
            intermediate[idx] = source[lower]
        else:
            intermediate[idx] = (1 - weight) * source[lower] + weight * source[upper]
    resized = np.zeros((rows, cols), dtype=float)
    for idx, position in enumerate(col_indices):
        lower = int(np.floor(position))
        upper = min(lower + 1, source.shape[1] - 1)
        weight = position - lower
        if lower == upper:
            resized[:, idx] = intermediate[:, lower]
        else:
            resized[:, idx] = (1 - weight) * intermediate[:, lower] + weight * intermediate[:, upper]
    return resized


def _fusion_matrix(stft_energy: np.ndarray, cwt_energy: np.ndarray, mel_energy: np.ndarray) -> np.ndarray:
    target_shape = (128, 128)
    stft_norm = _resize_matrix(_normalise_energy_matrix(stft_energy), target_shape)
    cwt_norm = _resize_matrix(_normalise_energy_matrix(cwt_energy), target_shape)
    mel_norm = _resize_matrix(_normalise_energy_matrix(mel_energy), target_shape)
    fusion = (stft_norm + cwt_norm + mel_norm) / 3.0
    return fusion


def _load_multimodal_npz(path: Path) -> Optional[Dict[str, np.ndarray]]:
    if path is None or not path.exists():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            return {key: data[key] for key in data.files}
    except Exception as exc:
        LOGGER.warning("无法读取已保存的多模态数据 %s: %s", path, exc)
    return None


def _plot_multimodal_example(
    result,
    output_path: Path,
    tf_config: TimeFrequencyConfig,
    data_output: Optional[Path] = None,
    metadata_output: Optional[Path] = None,
) -> None:
    predictions = result.final_predictions
    if predictions is None or predictions.empty:
        return
    target_features = result.target_features
    if target_features.empty:
        return

    if "max_probability" in predictions.columns:
        target_idx = predictions["max_probability"].astype(float).idxmax()
    else:
        target_idx = 0

    prediction_row = predictions.iloc[target_idx]
    row_index = prediction_row.get("row_index")
    if row_index is not None and row_index in target_features.index:
        feature_row = target_features.loc[row_index]
    else:
        feature_row = target_features.iloc[target_idx]

    fetcher = SegmentFetcher()
    segment = fetcher.get_segment(feature_row)
    if segment is None or len(segment) == 0:
        LOGGER.warning("无法提取多模态示例信号，跳过图像绘制")
        return

    sampling_rate = feature_row.get("sampling_rate", 32000.0)
    try:
        sampling_rate = float(sampling_rate)
    except Exception:
        sampling_rate = 32000.0

    saved_npz = _load_multimodal_npz(data_output) if data_output else None
    if saved_npz is not None:
        stft_freqs = saved_npz.get("stft_frequencies")
        stft_times = saved_npz.get("stft_times")
        stft_energy = saved_npz.get("stft_energy")
        cwt_scales = saved_npz.get("cwt_scales")
        cwt_times = saved_npz.get("cwt_times")
        cwt_energy = saved_npz.get("cwt_energy")
        mel_freqs = saved_npz.get("mel_frequencies")
        mel_times = saved_npz.get("mel_times")
        mel_energy = saved_npz.get("mel_energy")
        fusion_matrix = saved_npz.get("fusion")
        time_axis = saved_npz.get("time_axis")
        if time_axis is None:
            time_axis = np.arange(len(segment)) / sampling_rate
    else:
        representation = compute_multimodal_representation(segment, sampling_rate, tf_config)
        if representation is None:
            LOGGER.warning("无法生成多模态表示，跳过图像绘制")
            return
        stft_freqs = representation.stft.frequencies
        stft_times = representation.stft.times
        stft_energy = representation.stft.energy
        cwt_scales = representation.cwt.scales
        cwt_times = representation.cwt.times
        cwt_energy = representation.cwt.energy
        mel_freqs = representation.mel.frequencies
        mel_times = representation.mel.times
        mel_energy = representation.mel.energy
        time_axis = representation.time_axis
        fusion_matrix = _fusion_matrix(stft_energy, cwt_energy, mel_energy)
        if data_output is not None:
            data_output.parent.mkdir(parents=True, exist_ok=True)
            np.savez(
                data_output,
                time_axis=time_axis.astype(np.float32),
                segment=segment.astype(np.float32),
                sampling_rate=np.asarray([sampling_rate], dtype=np.float32),
                stft_frequencies=stft_freqs.astype(np.float32),
                stft_times=stft_times.astype(np.float32),
                stft_energy=stft_energy.astype(np.float32),
                cwt_scales=cwt_scales.astype(np.float32),
                cwt_times=cwt_times.astype(np.float32),
                cwt_energy=cwt_energy.astype(np.float32),
                mel_frequencies=mel_freqs.astype(np.float32),
                mel_times=mel_times.astype(np.float32),
                mel_energy=mel_energy.astype(np.float32),
                fusion=fusion_matrix.astype(np.float32),
            )

    if fusion_matrix is None:
        fusion_matrix = _fusion_matrix(stft_energy, cwt_energy, mel_energy)

    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(15.5, 7.2))
    grid = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.4])
    ax_time = fig.add_subplot(grid[0, :])
    ax_stft = fig.add_subplot(grid[1, 0])
    ax_cwt = fig.add_subplot(grid[1, 1])
    ax_mel = fig.add_subplot(grid[1, 2])
    ax_fusion = fig.add_subplot(grid[1, 3])

    ax_time.plot(time_axis, segment, color="#2ca02c", linewidth=0.9)
    ax_time.set_title("时域波形")
    ax_time.set_xlabel("时间（秒）")
    ax_time.set_ylabel("幅值")
    ax_time.grid(True, linestyle="--", alpha=0.3)

    stft_display = _normalise_energy_matrix(stft_energy)
    ax_stft.pcolormesh(stft_times, stft_freqs, stft_display, shading="auto", cmap="magma")
    ax_stft.set_title("短时傅里叶谱图（STFT）")
    ax_stft.set_xlabel("时间（秒）")
    ax_stft.set_ylabel("频率（赫兹）")

    cwt_display = _normalise_energy_matrix(cwt_energy)
    ax_cwt.pcolormesh(cwt_times, cwt_scales, cwt_display, shading="auto", cmap="viridis")
    ax_cwt.set_title("连续小波尺度图（CWT）")
    ax_cwt.set_xlabel("时间（秒）")
    ax_cwt.set_ylabel("尺度")

    mel_display = _normalise_energy_matrix(mel_energy)
    ax_mel.pcolormesh(mel_times, mel_freqs, mel_display, shading="auto", cmap="plasma")
    ax_mel.set_title("梅尔谱图")
    ax_mel.set_xlabel("时间（秒）")
    ax_mel.set_ylabel("梅尔频带")

    ax_fusion.imshow(fusion_matrix, aspect="auto", origin="lower", cmap="inferno")
    ax_fusion.set_title("多模态融合热力图")
    ax_fusion.set_xlabel("归一化时间索引")
    ax_fusion.set_ylabel("归一化频率/尺度")

    predicted_label = prediction_row.get("predicted_label", "未知")
    probability = prediction_row.get("max_probability")
    if probability is not None:
        try:
            probability = float(probability)
            subtitle = f"预测类别：{predicted_label} (置信度 {probability:.2f})"
        except Exception:
            subtitle = f"预测类别：{predicted_label}"
    else:
        subtitle = f"预测类别：{predicted_label}"

    file_id = feature_row.get("file_id", "unknown")
    ax_time.set_title(f"时域波形 – {file_id}")
    fig.suptitle(f"多模态融合示例｜{subtitle}")
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        probability_value = None
        if probability is not None:
            try:
                if np.isscalar(probability):
                    probability_value = float(probability)
            except Exception:
                probability_value = None
        consistency_value = None
        if not result.pseudo_quality.empty and row_index is not None:
            quality_match = result.pseudo_quality[result.pseudo_quality.get("row_index") == row_index]
            if not quality_match.empty and "consistency_score" in quality_match.columns:
                try:
                    consistency_value = float(quality_match["consistency_score"].iloc[-1])
                except Exception:
                    consistency_value = None
        try:
            sample_index_value = int(row_index) if row_index is not None else int(target_idx)
        except Exception:
            sample_index_value = int(target_idx)
        metadata = {
            "文件编号": file_id,
            "通道": str(feature_row.get("channel", "")),
            "样本索引": sample_index_value,
            "预测标签": str(predicted_label),
            "最大置信度": probability_value,
            "一致性得分": consistency_value,
            "分段长度": int(len(segment)),
            "采样率": float(sampling_rate),
        }
        with metadata_output.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)


def _plot_consistency_scatter(pseudo_quality: pd.DataFrame, path: Path) -> None:
    if pseudo_quality is None or pseudo_quality.empty:
        return
    if "max_probability" not in pseudo_quality.columns or "consistency_score" not in pseudo_quality.columns:
        return

    frame = pseudo_quality.copy()
    configure_chinese_font()
    import matplotlib.pyplot as plt

    probabilities = frame["max_probability"].astype(float)
    consistencies = frame["consistency_score"].astype(float)
    iterations = frame.get("pseudo_iteration")
    if iterations is None:
        iterations = pd.Series([1] * len(frame))
    iteration_values = iterations.astype(float)

    fig, ax = plt.subplots(figsize=(7.8, 5.6))
    scatter = ax.scatter(
        probabilities,
        consistencies,
        c=iteration_values,
        cmap="viridis",
        alpha=0.75,
        edgecolors="k",
        linewidths=0.3,
    )
    ax.set_xlabel("最大置信度")
    ax.set_ylabel("一致性得分")
    if "threshold" in frame.columns:
        ax.axvline(
            float(frame["threshold"].iloc[0]),
            color="#55a868",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="置信度阈值",
        )
    if "consistency_threshold" in frame.columns:
        ax.axhline(
            float(frame["consistency_threshold"].iloc[0]),
            color="#c44e52",
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            label="一致性阈值",
        )
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.legend(loc="lower right")
    colorbar = fig.colorbar(scatter, ax=ax)
    colorbar.set_label("伪标签迭代轮次")
    fig.suptitle("伪标签置信度与多模态一致性分布")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)
def run_pipeline(
    config_path: Path,
    source_override: Optional[Path] = None,
    target_override: Optional[Path] = None,
    output_override: Optional[Path] = None,
) -> None:
    raw_config = _load_yaml(config_path)
    transfer_config = _build_transfer_config(raw_config)

    features_cfg = raw_config.get("features", {})
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

    result = run_transfer_learning(source_df, target_df, transfer_config)

    outputs_cfg = raw_config.get("outputs", {})
    output_dir = output_override or Path(outputs_cfg.get("directory", "artifacts/task3"))
    _ensure_directory(output_dir)

    initial_predictions_path = output_dir / outputs_cfg.get("predictions_initial", "target_predictions_initial.csv")
    final_predictions_path = output_dir / outputs_cfg.get("predictions_final", "target_predictions.csv")
    pseudo_labels_path = output_dir / outputs_cfg.get("pseudo_labels", "pseudo_labels.csv")
    pseudo_history_path = output_dir / outputs_cfg.get("pseudo_history", "pseudo_history.csv")
    alignment_before_path = output_dir / outputs_cfg.get("alignment_before", "alignment_before.csv")
    alignment_after_path = output_dir / outputs_cfg.get("alignment_after", "alignment_after.csv")
    combined_before_path = output_dir / outputs_cfg.get("combined_before", "combined_features_before.csv")
    combined_aligned_path = output_dir / outputs_cfg.get("combined_aligned", "combined_features_aligned.csv")
    pseudo_quality_path = output_dir / outputs_cfg.get("pseudo_quality", "pseudo_label_quality.csv")
    tf_feature_path = output_dir / outputs_cfg.get("time_frequency_features", "time_frequency_features.txt")
    metrics_path = output_dir / outputs_cfg.get("metrics", "metrics.json")
    embedding_before_path = output_dir / outputs_cfg.get("embedding_before", "tsne_before.png")
    embedding_after_path = output_dir / outputs_cfg.get("embedding_after", "tsne_after.png")
    model_path = output_dir / outputs_cfg.get("model_path", "transfer_model.joblib")

    _write_dataframe(result.initial_predictions, initial_predictions_path)
    _write_dataframe(result.final_predictions, final_predictions_path)
    _write_dataframe(result.pseudo_labels, pseudo_labels_path)
    _write_dataframe(pd.DataFrame(result.pseudo_label_history), pseudo_history_path)
    _write_dataframe(result.alignment_before, alignment_before_path)
    _write_dataframe(result.alignment_after, alignment_after_path)
    _write_dataframe(result.combined_before, combined_before_path, translate=False)
    _write_dataframe(result.combined_aligned, combined_aligned_path, translate=False)
    _write_dataframe(result.pseudo_quality, pseudo_quality_path)

    if result.time_frequency_features:
        tf_dictionary = build_feature_dictionary(result.time_frequency_features)
        tf_dictionary = tf_dictionary.rename(
            columns={"feature": "特征编码", "chinese_name": "特征中文名"}
        )
        tf_dictionary[["特征编码", "特征中文名"]].to_csv(
            tf_feature_path,
            index=False,
            encoding="utf-8-sig",
        )

    metrics_payload = {
        "base_model": result.base_result.metrics,
        "feature_columns": result.feature_columns,
        "time_frequency_feature_count": len(result.time_frequency_features),
        "consistency_feature_count": len(result.consistency_features),
        "pseudo_label_count": int(len(result.pseudo_labels)),
        "pseudo_iterations": int(len(result.pseudo_label_history)),
        "pseudo_quality_count": int(len(result.pseudo_quality)),
        "consistency_threshold": float(transfer_config.pseudo_label.consistency_threshold),
        "modal_feature_groups": result.modal_feature_groups,
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics_payload, handle, ensure_ascii=False, indent=2)

    dump(result.final_pipeline, model_path)

    tsne_before = run_tsne(result.combined_before)
    if tsne_before is not None:
        plot_embedding(tsne_before, embedding_before_path, title="t-SNE 嵌入（对齐前）")
    tsne_after = run_tsne(result.combined_aligned)
    if tsne_after is not None:
        plot_embedding(tsne_after, embedding_after_path, title="t-SNE 嵌入（对齐后）")

    pseudo_plot_path = output_dir / outputs_cfg.get("pseudo_history_plot", "伪标签演化曲线.png")
    _plot_pseudo_history(result.pseudo_label_history, pseudo_plot_path)

    tf_plot_path = output_dir / outputs_cfg.get("tf_feature_plot", "多模态特征分布对比.png")
    _plot_time_frequency_distribution(result, tf_plot_path)

    tf_example_path = output_dir / outputs_cfg.get("tf_example_plot", "多模态时频示例.png")
    tf_example_data = outputs_cfg.get("tf_example_data", "多模态时频示例数据.npz")
    tf_example_meta = outputs_cfg.get("tf_example_metadata", "多模态时频示例信息.json")
    _plot_multimodal_example(
        result,
        tf_example_path,
        transfer_config.time_frequency,
        data_output=output_dir / tf_example_data,
        metadata_output=output_dir / tf_example_meta,
    )

    pseudo_quality_plot = output_dir / outputs_cfg.get("pseudo_quality_plot", "伪标签一致性散点.png")
    _plot_consistency_scatter(result.pseudo_quality, pseudo_quality_plot)

    LOGGER.info("Transfer learning pipeline completed. Outputs saved to %s", output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the task 3 transfer diagnosis pipeline.")
    parser.add_argument("--config", type=Path, default=Path("config/task3_config.yaml"), help="Path to the YAML configuration file.")
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
