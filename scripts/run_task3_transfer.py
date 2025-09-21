"""Run the transfer diagnosis pipeline (task 3)."""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from joblib import dump
import yaml
import numpy as np
from scipy import signal

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.feature_analysis import configure_chinese_font, plot_embedding, run_tsne
from src.tasks.task3 import (
    TimeFrequencyConfig,
    TransferConfig,
    parse_transfer_config,
    run_transfer_learning,
)
from src.tasks.task3.features import continuous_wavelet_transform, get_cwt_wavelet
from src.tasks.task3.segment_loader import SegmentFetcher

LOGGER = logging.getLogger("task3_transfer")


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _ensure_directory(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _build_transfer_config(raw: Dict[str, Any]) -> TransferConfig:
    return parse_transfer_config(raw)


def _write_dataframe(frame: pd.DataFrame, path: Path) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _plot_pseudo_history(history: Optional[List[Dict[str, float]]], path: Path) -> None:
    if not history:
        return
    frame = pd.DataFrame(history)
    if frame.empty:
        return
    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.bar(frame["iteration"], frame["new_samples"], color="#1f77b4", alpha=0.7, label="新增伪标签数")
    ax1.set_xlabel("迭代轮次")
    ax1.set_ylabel("新增伪标签数")
    ax1.set_xticks(frame["iteration"])
    ax1.grid(True, linestyle="--", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(frame["iteration"], frame["cumulative_pseudo"], color="#d62728", marker="o", label="累计伪标签数")
    ax2.set_ylabel("累计伪标签数")

    lines, labels = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, loc="upper left")

    fig.suptitle("伪标签迭代过程")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _plot_time_frequency_distribution(result, path: Path) -> None:
    features = [feat for feat in result.time_frequency_features if feat in result.source_features.columns and feat in result.target_features.columns]
    if not features:
        return

    source = result.source_features
    target = result.target_features
    diffs = []
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

    records = []
    for feature in top_features:
        src_values = source[feature].dropna().astype(float)
        tgt_values = target[feature].dropna().astype(float)
        if src_values.empty and tgt_values.empty:
            continue
        records.append({"特征": feature, "数据域": "源域", "均值": float(src_values.mean()), "标准差": float(src_values.std(ddof=1) if src_values.size > 1 else 0.0)})
        records.append({"特征": feature, "数据域": "目标域", "均值": float(tgt_values.mean()), "标准差": float(tgt_values.std(ddof=1) if tgt_values.size > 1 else 0.0)})
    if not records:
        return

    frame = pd.DataFrame(records)
    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5.5))
    x = np.arange(len(top_features))
    width = 0.35

    src = frame[frame["数据域"] == "源域"].set_index("特征").reindex(top_features)
    tgt = frame[frame["数据域"] == "目标域"].set_index("特征").reindex(top_features)

    ax.bar(x - width / 2, src["均值"], width, yerr=src["标准差"], capsize=4, label="源域均值", color="#1f77b4")
    ax.bar(x + width / 2, tgt["均值"], width, yerr=tgt["标准差"], capsize=4, label="目标域均值", color="#ff7f0e")

    ax.set_xticks(x)
    ax.set_xticklabels(top_features, rotation=25, ha="right")
    ax.set_ylabel("特征均值")
    ax.set_title("时频特征源/目标域分布对比")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


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

    configure_chinese_font()
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    time_axis = np.arange(len(segment)) / sampling_rate
    axes[0].plot(time_axis, segment, color="#2ca02c", linewidth=0.9)
    axes[0].set_title("时域波形")
    axes[0].set_xlabel("时间 [秒]")
    axes[0].set_ylabel("幅值")
    axes[0].grid(True, linestyle="--", alpha=0.3)

    nperseg = min(len(segment), max(128, tf_config.stft_nperseg))
    nperseg = max(nperseg, 128)
    noverlap = min(tf_config.stft_noverlap, nperseg - 1)
    nfft = max(tf_config.stft_nfft, nperseg)
    try:
        freqs, times, stft_values = signal.stft(
            segment,
            fs=sampling_rate,
            window=tf_config.stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
        )
    except Exception as exc:
        LOGGER.warning("STFT 计算失败：%s", exc)
        return
    stft_magnitude = np.abs(stft_values)
    axes[1].pcolormesh(times, freqs, stft_magnitude, shading="auto", cmap="magma")
    axes[1].set_title("STFT 时频图")
    axes[1].set_xlabel("时间 [秒]")
    axes[1].set_ylabel("频率 [Hz]")

    if tf_config.cwt_num_scales < 2 or tf_config.cwt_max_scale <= tf_config.cwt_min_scale:
        LOGGER.warning("CWT 配置无效，无法绘制CWT图像")
        return
    scales = np.linspace(tf_config.cwt_min_scale, tf_config.cwt_max_scale, tf_config.cwt_num_scales)
    try:
        wavelet = get_cwt_wavelet(tf_config.cwt_wavelet)
        coefficients = continuous_wavelet_transform(segment, wavelet, scales)
    except Exception as exc:
        LOGGER.warning("CWT 计算失败：%s", exc)
        return
    cwt_magnitude = np.abs(coefficients)
    time_grid = np.arange(coefficients.shape[1]) / sampling_rate
    axes[2].pcolormesh(time_grid, scales, cwt_magnitude, shading="auto", cmap="viridis")
    axes[2].set_title("CWT 尺度图")
    axes[2].set_xlabel("时间 [秒]")
    axes[2].set_ylabel("尺度")

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
    axes[0].set_title(f"时域波形 – {file_id}")
    fig.suptitle(f"多模态融合示例 | {subtitle}")
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)

    if data_output is not None:
        data_output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "time_axis": time_axis.astype(np.float32),
            "segment": segment.astype(np.float32),
            "sampling_rate": np.asarray([sampling_rate], dtype=np.float32),
            "stft_frequencies": freqs.astype(np.float32),
            "stft_times": times.astype(np.float32),
            "stft_magnitude": stft_magnitude.astype(np.float32),
            "stft_phase": np.angle(stft_values).astype(np.float32),
            "cwt_scales": scales.astype(np.float32),
            "cwt_magnitude": cwt_magnitude.astype(np.float32),
            "cwt_phase": np.angle(coefficients).astype(np.float32),
        }
        np.savez(data_output, **payload)

    if metadata_output is not None:
        metadata_output.parent.mkdir(parents=True, exist_ok=True)
        probability_value = None
        if probability is not None:
            try:
                if np.isscalar(probability):
                    probability_value = float(probability)
            except Exception:
                probability_value = None
        metadata = {
            "file_id": file_id,
            "channel": str(feature_row.get("channel", "")),
            "row_index": int(row_index) if row_index is not None else int(target_idx),
            "predicted_label": str(predicted_label),
            "max_probability": probability_value,
            "segment_length": int(len(segment)),
            "sampling_rate": float(sampling_rate),
        }
        with metadata_output.open("w", encoding="utf-8") as handle:
            json.dump(metadata, handle, ensure_ascii=False, indent=2)
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
    _write_dataframe(result.combined_before, combined_before_path)
    _write_dataframe(result.combined_aligned, combined_aligned_path)

    if result.time_frequency_features:
        with tf_feature_path.open("w", encoding="utf-8") as handle:
            for feature in result.time_frequency_features:
                handle.write(f"{feature}\n")

    metrics_payload = {
        "base_model": result.base_result.metrics,
        "feature_columns": result.feature_columns,
        "time_frequency_feature_count": len(result.time_frequency_features),
        "pseudo_label_count": int(len(result.pseudo_labels)),
        "pseudo_iterations": int(len(result.pseudo_label_history)),
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
