"""Generate visualisations and statistics for extracted feature tables."""
from __future__ import annotations

import argparse
import logging
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd
import yaml

# Ensure the project root is on the import path when running as a script.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.analysis.feature_analysis import (
    SignalPlotConfig,
    compute_domain_alignment_metrics,
    compute_feature_importance,
    compute_feature_statistics,
    load_feature_table,
    plot_covariance_heatmap,
    plot_embedding,
    plot_envelope_spectrum,
    plot_signal_diagnostics,
    plot_time_frequency_saliency,
    plot_time_series_grid,
    plot_window_sequence,
    prepare_combined_features,
    run_tsne,
    run_umap,
    translate_statistics_columns,
)
from src.analysis.feature_dictionary import build_feature_dictionary
from src.data_io.mat_loader import load_source_directory, load_target_directory
from src.pipelines.build_feature_dataset import SegmentationConfig
from src.feature_engineering.bearing import DEFAULT_BEARINGS

LOGGER = logging.getLogger("feature_analysis")


DATASET_DISPLAY_MAP = {
    "source": "源域",
    "target": "目标域",
}


def _safe_identifier(value: Optional[str]) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    cleaned = re.sub(r"[^0-9A-Za-z_\-]+", "_", text)
    return cleaned or "unknown"


def _resolve_label(summary, record) -> Optional[str]:
    def _normalise(value: Any) -> Optional[str]:
        if value is None:
            return None
        try:
            if math.isnan(value):  # type: ignore[arg-type]
                return None
        except (TypeError, ValueError):
            pass
        if isinstance(value, bool):
            text = "true" if value else "false"
        else:
            text = str(value)
        text = text.strip()
        return text if text else None

    label = _normalise(getattr(record, "label", None))
    if label is not None:
        return label
    info = getattr(summary, "label_info", None)
    if info is not None:
        info_label = _normalise(getattr(info, "label", None))
        if info_label is not None:
            return info_label
    return None


def _resolve_bearing_map(channel_mapping: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    bearings: Dict[str, Any] = {}
    if not channel_mapping:
        return bearings
    for channel, name in channel_mapping.items():
        if name is None:
            continue
        spec = DEFAULT_BEARINGS.get(str(name))
        if spec is None:
            LOGGER.warning("未识别的轴承型号 %s (通道 %s)，将跳过故障频率标注", name, channel)
            continue
        bearings[str(channel).upper()] = spec
    return bearings


def _select_source_representatives(
    summaries: Sequence[Any],
    source_root: Path,
) -> List[Any]:
    """按“4 + 5×3”策略为源数据选择代表性样本（共 19 条）。"""

    if not summaries:
        return []

    resolved_root = source_root.resolve()
    normals: List[Any] = []
    selection: Dict[Tuple[str, str], Any] = {}

    top_priority = ["12kHz_DE_data", "12kHz_FE_data", "48kHz_DE_data"]
    category_priority = ["B", "IR", "OR_Centered", "OR_Opposite", "OR_Orthogonal"]
    category_labels = {
        "B": "B（滚动体）",
        "IR": "IR（内圈）",
        "OR_Centered": "OR-Centered（外圈-对中）",
        "OR_Opposite": "OR-Opposite（外圈-反向）",
        "OR_Orthogonal": "OR-Orthogonal（外圈-正交）",
    }

    def _summary_path(summary: Any) -> Path:
        raw_path = getattr(summary, "file_path", None)
        if raw_path is None:
            return Path(".")
        return Path(raw_path)

    for summary in sorted(summaries, key=lambda item: str(_summary_path(item))):
        path = _summary_path(summary)
        try:
            relative = path.resolve().relative_to(resolved_root)
        except Exception:
            LOGGER.debug("无法解析 %s 相对于源数据根目录 %s 的路径", path, resolved_root)
            relative = Path(path.name)

        parts = relative.parts
        if not parts:
            continue
        domain = parts[0]
        if domain == "48kHz_Normal_data":
            normals.append(summary)
            continue

        category: Optional[str] = None
        if len(parts) >= 2:
            second = parts[1]
            if second in {"B", "IR"}:
                category = second
            elif second == "OR" and len(parts) >= 3:
                category = f"OR_{parts[2]}"

        if category is None:
            continue

        key = (domain, category)
        if key not in selection:
            selection[key] = summary

    ordered: List[Any] = []
    normals_sorted = sorted(normals, key=lambda item: str(_summary_path(item)))
    max_normals = 4
    if len(normals_sorted) > max_normals:
        LOGGER.info("48kHz_Normal_data 共检测到 %d 条信号，仅展示前 %d 条作为代表", len(normals_sorted), max_normals)
    ordered.extend(normals_sorted[:max_normals])

    for domain in top_priority:
        for category in category_priority:
            summary = selection.get((domain, category))
            if summary is None:
                LOGGER.warning("未在 %s 下找到类别 %s 的代表样本", domain, category)
                continue
            if summary not in ordered:
                ordered.append(summary)

    for _key, summary in selection.items():
        if summary not in ordered:
            ordered.append(summary)

    if not ordered:
        LOGGER.warning("代表性样本选择为空，回退到全部源数据")
        return list(summaries)

    breakdown: Dict[str, List[str]] = defaultdict(list)
    for summary in ordered:
        path = _summary_path(summary)
        try:
            relative = path.resolve().relative_to(resolved_root)
        except Exception:
            relative = Path(path.name)

        parts = relative.parts
        if not parts:
            continue
        domain = parts[0]
        if domain == "48kHz_Normal_data":
            breakdown[domain].append(getattr(summary, "file_id", path.stem))
            continue

        category_name = "未识别类别"
        if len(parts) >= 2:
            second = parts[1]
            if second in {"B", "IR"}:
                category_name = category_labels.get(second, second)
            elif second == "OR" and len(parts) >= 3:
                category_key = f"OR_{parts[2]}"
                category_name = category_labels.get(category_key, category_key)
        breakdown[domain].append(category_name)

    normal_count = len(breakdown.get("48kHz_Normal_data", []))
    if normal_count:
        breakdown["48kHz_Normal_data"] = [f"共 {normal_count} 条"]

    segments: List[str] = []
    if "48kHz_Normal_data" in breakdown:
        segments.append(f"48kHz_Normal_data×{normal_count}")

    for domain in top_priority:
        items = breakdown.get(domain)
        if not items:
            continue
        segments.append(f"{domain}：{', '.join(items)}")

    for domain, items in sorted(breakdown.items()):
        if domain == "48kHz_Normal_data" or domain in top_priority:
            continue
        segments.append(f"{domain}：{', '.join(items)}")

    breakdown_text = "; ".join(segments) if segments else "未识别到具体目录结构"
    expected_total = 19
    LOGGER.info(
        "已为源数据选择 %d 个代表性样本（目标 %d = 4 + 5×3）。分布：%s",
        len(ordered),
        expected_total,
        breakdown_text,
    )
    if len(ordered) != expected_total:
        LOGGER.warning(
            "代表性样本数量 %d 未达到预期 %d（4 + 5×3），请检查源数据目录是否完整",
            len(ordered),
            expected_total,
        )
    return ordered


def _render_detail_plots(
    domain_prefix: str,
    records: Sequence[Tuple[Any, Any]],
    analysis_root: Path,
    signal_config: SignalPlotConfig,
    bearing_map: Dict[str, Any],
    verification_records: List[pd.DataFrame],
) -> None:
    for summary, record in records:
        file_id = _safe_identifier(getattr(summary, "file_id", None))
        channel = _safe_identifier(getattr(record, "channel", None))
        label_text = _resolve_label(summary, record)
        label_part = _safe_identifier(label_text) if label_text else None
        name_parts = [domain_prefix]
        if label_part and label_part != "unknown":
            name_parts.append(label_part)
        name_parts.extend([file_id, channel])
        base_name = "_".join(part for part in name_parts if part)

        diag_path = analysis_root / f"{base_name}_diagnostics.png"
        plot_signal_diagnostics(summary, record, diag_path, signal_config)
        if diag_path.exists():
            LOGGER.info("%s 诊断图已保存至 %s", domain_prefix, diag_path)

        window_path = analysis_root / f"{base_name}_窗序折线.png"
        plot_window_sequence(summary, record, window_path, signal_config)
        if window_path.exists():
            LOGGER.info("%s 窗序折线图已保存至 %s", domain_prefix, window_path)

        envelope_path = analysis_root / f"{base_name}_包络谱.png"
        channel_key = str(getattr(record, "channel", "")).upper()
        bearing_spec = bearing_map.get(channel_key)
        rpm = getattr(record, "rpm", None)
        if rpm is None:
            rpm = getattr(summary, "rpm", None)
        try:
            rpm_value = float(rpm) if rpm is not None else None
        except Exception:
            rpm_value = None
        summary_df = plot_envelope_spectrum(
            summary,
            record,
            envelope_path,
            signal_config,
            max_frequency=None,
            bearing=bearing_spec,
            rpm=rpm_value,
        )
        if summary_df is not None and not summary_df.empty:
            summary_copy = summary_df.copy()
            summary_copy["数据域"] = DATASET_DISPLAY_MAP.get(domain_prefix, domain_prefix)
            summary_copy["文件"] = getattr(summary, "file_id", "")
            summary_copy["通道"] = getattr(record, "channel", "")
            summary_copy["转速(rpm)"] = rpm_value
            verification_records.append(summary_copy)
        if envelope_path.exists():
            LOGGER.info("%s 包络谱已保存至 %s", domain_prefix, envelope_path)

        saliency_path = analysis_root / f"{base_name}_时频显著性.png"
        plot_time_frequency_saliency(summary, record, saliency_path, signal_config)
        if saliency_path.exists():
            LOGGER.info("%s 时频显著性图已保存至 %s", domain_prefix, saliency_path)


def _load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _save_dataframe_chinese(frame, path: Path) -> None:
    if frame is None:
        return
    if hasattr(frame, "empty") and frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def _parse_segmentation(config: Dict[str, Any]) -> SegmentationConfig:
    return SegmentationConfig(
        window_seconds=float(config.get("window_seconds", 1.0)),
        overlap=float(config.get("overlap", 0.5)),
        drop_last=bool(config.get("drop_last", True)),
    )


def _load_target_summaries(config: Dict[str, Any]):
    root = Path(config.get("root", "targetData"))
    pattern = config.get("pattern", "*.mat")
    sampling_rate = float(config.get("sampling_rate", 32000))
    rpm_value = config.get("rpm")
    rpm = float(rpm_value) if rpm_value is not None else None
    LOGGER.info("正在从 %s 载入目标域信号", root)
    return load_target_directory(root, sampling_rate=sampling_rate, rpm=rpm, pattern=pattern)


def _load_source_summaries(config: Dict[str, Any]):
    root = Path(config.get("root", "sourceData"))
    pattern = config.get("pattern", "**/*.mat")
    default_sampling_rate = float(config.get("default_sampling_rate", 12000))
    LOGGER.info("正在从 %s 载入源域信号", root)
    return load_source_directory(root, pattern=pattern, default_sampling_rate=default_sampling_rate)


def analyse_features(
    config_path: Path,
    output_dir: Optional[Path] = None,
    analysis_dir: Optional[Path] = None,
    max_records: Optional[int] = None,
    preview_seconds: float = 2.0,
    target_max_records: Optional[int] = None,
    source_max_records: Optional[int] = None,
    source_preview_mode: str = "representative",
) -> None:
    config = _load_yaml(config_path)
    outputs = config.get("outputs", {})

    source_bearings = _resolve_bearing_map(config.get("source", {}).get("channel_bearings"))
    target_bearings = _resolve_bearing_map(config.get("target", {}).get("channel_bearings"))
    verification_records: List[pd.DataFrame] = []

    feature_root = output_dir or Path(outputs.get("directory", "artifacts/task1"))
    feature_root.mkdir(parents=True, exist_ok=True)

    analysis_root = analysis_dir or (feature_root / "analysis")
    analysis_root.mkdir(parents=True, exist_ok=True)

    source_path = feature_root / outputs.get("source_feature_table", "source_features.csv")
    target_path = feature_root / outputs.get("target_feature_table", "target_features.csv")

    source_features = load_feature_table(source_path, dataset_name="source") if source_path.exists() else None
    target_features = load_feature_table(target_path, dataset_name="target") if target_path.exists() else None

    frames = [frame for frame in [source_features, target_features] if frame is not None]
    combined = prepare_combined_features(frames)
    if combined.empty:
        LOGGER.warning("未找到可供分析的特征表")
    else:
        stats = compute_feature_statistics(combined)
        if not stats.empty:
            stats_localised = translate_statistics_columns(stats)
            stats_path = analysis_root / "特征统计汇总.csv"
            LOGGER.info("正在写出特征统计表：%s", stats_path)
            _save_dataframe_chinese(stats_localised, stats_path)
        else:
            LOGGER.warning("无法计算特征统计量")

        tsne_result = run_tsne(combined)
        if tsne_result:
            tsne_path = analysis_root / "tsne_embedding.png"
            plot_embedding(tsne_result, tsne_path, title="t-SNE 特征嵌入")
            LOGGER.info("t-SNE 嵌入图已保存至 %s", tsne_path)

        umap_result = run_umap(combined)
        if umap_result:
            umap_path = analysis_root / "umap_embedding.png"
            plot_embedding(umap_result, umap_path, title="UMAP 特征嵌入")
            LOGGER.info("UMAP 嵌入图已保存至 %s", umap_path)

        cov_path = analysis_root / "特征协方差热图.png"
        plot_covariance_heatmap(combined, cov_path)
        if cov_path.exists():
            LOGGER.info("特征协方差热图已保存至 %s", cov_path)

        alignment = compute_domain_alignment_metrics(combined)
        if not alignment.empty:
            alignment_path = analysis_root / "域对齐指标.csv"
            LOGGER.info("正在写出域对齐指标：%s", alignment_path)
            _save_dataframe_chinese(alignment, alignment_path)

        importance_path = analysis_root / "特征重要度.png"
        importance_df = compute_feature_importance(combined, importance_path)
        if importance_df is not None and not importance_df.empty:
            if importance_path.exists():
                LOGGER.info("特征重要度图已保存至 %s", importance_path)
            importance_table = analysis_root / "特征重要度.csv"
            LOGGER.info("正在写出特征重要度表：%s", importance_table)
            _save_dataframe_chinese(importance_df, importance_table)

        combined_report = translate_statistics_columns(combined)
        combined_path = analysis_root / "特征整合表.csv"
        LOGGER.info("正在写出特征整合表：%s", combined_path)
        _save_dataframe_chinese(combined_report, combined_path)

        dictionary_path = analysis_root / "特征中英文对照表.csv"
        dictionary = build_feature_dictionary(combined.columns)
        LOGGER.info("正在写出特征中英文对照表：%s", dictionary_path)
        _save_dataframe_chinese(dictionary, dictionary_path)

    signal_config = SignalPlotConfig(preview_seconds=preview_seconds)

    target_config = config.get("target")
    if target_config:
        segmentation = _parse_segmentation(target_config.get("segmentation", {}))
        signal_config.window_seconds = segmentation.window_seconds
        signal_config.overlap = segmentation.overlap
        signal_config.drop_last = segmentation.drop_last

        target_summaries = _load_target_summaries(target_config)
        if target_summaries:
            grid_path = analysis_root / "target_time_series_overview.png"
            if target_max_records is not None:
                target_limit = target_max_records
            elif max_records is not None:
                target_limit = max_records
            else:
                target_limit = len(target_summaries)
            if target_limit is not None and target_limit <= 0:
                target_limit = None

            target_records = plot_time_series_grid(
                target_summaries,
                grid_path,
                signal_config,
                max_records=target_limit,
                columns=2,
            )
            if grid_path.exists():
                LOGGER.info("目标域时序概览图已保存至 %s", grid_path)
            if target_records:
                _render_detail_plots(
                    "target",
                    target_records,
                    analysis_root,
                    signal_config,
                    target_bearings,
                    verification_records,
                )
            else:
                LOGGER.warning("目标域信号记录为空，无法绘制详细图像")
        else:
            LOGGER.warning("未找到可用于时序可视化的目标域信号")

    source_config = config.get("source")
    if source_config:
        segmentation = _parse_segmentation(source_config.get("segmentation", {}))
        signal_config.window_seconds = segmentation.window_seconds
        signal_config.overlap = segmentation.overlap
        signal_config.drop_last = segmentation.drop_last

        source_summaries = _load_source_summaries(source_config)
        if source_summaries:
            preview_mode = source_preview_mode.lower().strip()
            if preview_mode == "representative":
                source_root = Path(source_config.get("root", "sourceData"))
                preview_summaries = _select_source_representatives(source_summaries, source_root)
            else:
                preview_summaries = list(source_summaries)

            if source_max_records is not None:
                source_limit = source_max_records
            elif max_records is not None:
                source_limit = max_records
            elif preview_mode == "representative":
                source_limit = len(preview_summaries)
            else:
                source_limit = None
            if source_limit is not None and source_limit <= 0:
                source_limit = None

            grid_path = analysis_root / "source_time_series_overview.png"
            source_records = plot_time_series_grid(
                preview_summaries,
                grid_path,
                signal_config,
                max_records=source_limit,
                columns=2,
            )
            if grid_path.exists():
                LOGGER.info("源域时序概览图已保存至 %s", grid_path)
            if source_records:
                _render_detail_plots(
                    "source",
                    source_records,
                    analysis_root,
                    signal_config,
                    source_bearings,
                    verification_records,
                )
            else:
                LOGGER.warning("源域信号记录为空，无法绘制详细图像")
        else:
            LOGGER.warning("未找到可用于时序可视化的源域信号")

    if verification_records:
        verification_df = pd.concat(verification_records, ignore_index=True)
        rename_map = {
            "fault": "特征频率类型",
            "predicted_frequency": "理论频率(Hz)",
            "peak_frequency": "实测峰值频率(Hz)",
            "predicted_amplitude": "理论频率幅值",
            "peak_amplitude": "峰值幅值",
            "frequency_error": "频率误差(Hz)",
            "数据域": "数据域",
            "文件": "文件编号",
            "通道": "通道",
            "转速(rpm)": "转速(rpm)",
        }
        verification_df = verification_df.rename(columns=rename_map)
        verification_path = analysis_root / "故障特征频率验证.csv"
        LOGGER.info("正在写出故障特征频率验证表：%s", verification_path)
        _save_dataframe_chinese(verification_df, verification_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="生成任务1特征分析的图表与报表。")
    parser.add_argument("--config", type=Path, default=Path("config/dataset_config.yaml"), help="指定数据配置的 YAML 文件路径。")
    parser.add_argument("--output-dir", type=Path, default=None, help="覆盖特征 CSV 文件所在的目录。")
    parser.add_argument(
        "--analysis-dir",
        type=Path,
        default=None,
        help="自定义可视化图表与分析报表的输出目录。",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="控制时序网格图展示的信号数量上限；未设置时，将自动按数据域确定数量。",
    )
    parser.add_argument(
        "--preview-seconds",
        type=float,
        default=2.0,
        help="时域图每条信号预览的秒数，可根据需求放大细节。",
    )
    parser.add_argument(
        "--target-max-records",
        type=int,
        default=None,
        help="单独覆盖目标域信号的展示数量（<=0 表示展示全部）。",
    )
    parser.add_argument(
        "--source-max-records",
        type=int,
        default=None,
        help="单独覆盖源域信号的展示数量（<=0 表示展示全部）。",
    )
    parser.add_argument(
        "--source-preview-mode",
        choices=["representative", "diverse"],
        default="representative",
        help="源数据可视化采样策略：representative=按目录挑选 19 条代表样本（48kHz_Normal×4 + 三个故障目录各 5 条），diverse=使用完整样本集。",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    analyse_features(
        args.config,
        output_dir=args.output_dir,
        analysis_dir=args.analysis_dir,
        max_records=args.max_records,
        preview_seconds=args.preview_seconds,
        target_max_records=args.target_max_records,
        source_max_records=args.source_max_records,
        source_preview_mode=args.source_preview_mode,
    )


if __name__ == "__main__":
    main()
