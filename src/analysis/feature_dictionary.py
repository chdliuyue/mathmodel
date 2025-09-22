"""Generate bilingual descriptions for engineered features."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional

import pandas as pd


@dataclass(frozen=True)
class FeatureDescription:
    """Container describing a feature in both English and Chinese."""

    english: str
    chinese: str
    category: str
    description: str


_METADATA_TRANSLATIONS: Mapping[str, FeatureDescription] = {
    "dataset": FeatureDescription("Dataset", "数据域", "metadata", "源域(source)或目标域(target)标识"),
    "file_id": FeatureDescription("File identifier", "文件编号", "metadata", "原始数据文件的唯一编号"),
    "file_path": FeatureDescription("File path", "文件路径", "metadata", "原始 MAT 文件所在的完整路径"),
    "channel": FeatureDescription("Channel", "测点通道", "metadata", "传感器或测点名称(如DE/FE/BA/SENSOR)"),
    "segment_index": FeatureDescription("Segment index", "分段序号", "metadata", "滑窗分段后的索引号(从0开始)"),
    "start_sample": FeatureDescription("Start sample", "起始样本点", "metadata", "该分段在原始信号中的起始采样点"),
    "end_sample": FeatureDescription("End sample", "结束样本点", "metadata", "该分段在原始信号中的结束采样点"),
    "segment_length": FeatureDescription("Segment length", "分段样本长度", "metadata", "该分段包含的采样点数量"),
    "segment_duration": FeatureDescription("Segment duration", "分段时长", "metadata", "该分段对应的时间长度(秒)"),
    "sampling_rate": FeatureDescription("Sampling rate", "采样率", "metadata", "振动信号采样频率(Hz)"),
    "rpm": FeatureDescription("Shaft speed", "转速", "metadata", "设备轴的转速(RPM)"),
    "label": FeatureDescription("Label", "故障类别", "metadata", "源域人工标注的故障类型"),
    "label_code": FeatureDescription("Label code", "故障编码", "metadata", "故障类别的缩写编码(如IR/OR/B/N)"),
    "load_hp": FeatureDescription("Load level", "负载等级", "metadata", "试验工况对应的马力等级"),
    "fault_size_inch": FeatureDescription("Fault size (inch)", "故障尺寸(英寸)", "metadata", "滚动体/滚道故障尺寸(英寸)"),
    "fault_size_mm": FeatureDescription("Fault size (mm)", "故障尺寸(毫米)", "metadata", "滚动体/滚道故障尺寸(毫米)"),
    "selection_score": FeatureDescription("Selection score", "样本匹配得分", "metadata", "源域样本与目标工况的相似度评分"),
}

_TIME_BASE: Mapping[str, FeatureDescription] = {
    "mean": FeatureDescription("Mean", "均值", "time", "信号幅值的算术平均值"),
    "std": FeatureDescription("Standard deviation", "标准差", "time", "信号幅值的波动程度"),
    "var": FeatureDescription("Variance", "方差", "time", "幅值方差, 衡量能量离散程度"),
    "rms": FeatureDescription("Root mean square", "均方根", "time", "幅值平方平均的平方根, 能量指标"),
    "abs_mean": FeatureDescription("Mean absolute value", "绝对均值", "time", "幅值绝对值的平均, 对冲击敏感"),
    "peak": FeatureDescription("Peak amplitude", "峰值", "time", "信号绝对幅值的最大值"),
    "peak_to_peak": FeatureDescription("Peak-to-peak", "峰峰值", "time", "最大值与最小值之差"),
    "skewness": FeatureDescription("Skewness", "偏度", "time", "分布的不对称程度"),
    "kurtosis": FeatureDescription("Kurtosis", "峭度", "time", "分布尖锐程度, 对冲击敏感"),
    "mean_crossing_rate": FeatureDescription("Mean crossing rate", "均值过零率", "time", "信号过均值的次数比例"),
    "shape_factor": FeatureDescription("Shape factor", "波形因子", "time", "均方根与绝对均值之比"),
    "crest_factor": FeatureDescription("Crest factor", "峰值因子", "time", "峰值与均方根之比"),
    "impulse_factor": FeatureDescription("Impulse factor", "脉冲因子", "time", "峰值与绝对均值之比"),
    "clearance_factor": FeatureDescription("Clearance factor", "裕度因子", "time", "峰值与平方根绝对均值平方之比"),
    "squared_mean": FeatureDescription("Mean of squares", "平方均值", "time", "幅值平方的平均值"),
    "absolute_deviation": FeatureDescription("Absolute deviation", "平均绝对偏差", "time", "相对于均值的平均偏差"),
    "variance_ratio": FeatureDescription("Variance ratio", "方差比", "time", "方差与平方均值之比"),
    "signal_to_noise_ratio": FeatureDescription("Signal-to-noise ratio", "信噪比", "time", "均方根与平均绝对偏差之比"),
}

_ENVELOPE_EXTRA: Mapping[str, FeatureDescription] = {
    "peak_frequency": FeatureDescription("Envelope peak frequency", "包络谱峰值频率", "envelope", "包络谱幅度最大处的频率"),
    "bandwidth": FeatureDescription("Envelope bandwidth", "包络谱带宽", "envelope", "包络谱能量分布的带宽"),
    "spectral_entropy": FeatureDescription("Envelope spectral entropy", "包络谱熵", "envelope", "包络谱能量分布的熵值"),
}

_FREQUENCY_TRANSLATIONS: Mapping[str, FeatureDescription] = {
    "centroid": FeatureDescription("Spectral centroid", "谱质心", "frequency", "频谱能量的加权平均频率"),
    "spread": FeatureDescription("Spectral spread", "谱带宽", "frequency", "频谱围绕质心的离散程度"),
    "skewness": FeatureDescription("Spectral skewness", "谱偏度", "frequency", "频谱能量分布的不对称性"),
    "kurtosis": FeatureDescription("Spectral kurtosis", "谱峭度", "frequency", "频谱能量的尖峰程度"),
    "peak_frequency": FeatureDescription("Spectral peak frequency", "谱峰频率", "frequency", "频谱幅度最大处的频率"),
    "median_frequency": FeatureDescription("Median frequency", "中值频率", "frequency", "累计谱能量达到50%时的频率"),
    "spectral_entropy": FeatureDescription("Spectral entropy", "谱熵", "frequency", "频谱能量分布的熵值"),
    "spectral_rms": FeatureDescription("Spectral RMS", "谱均方根", "frequency", "频谱幅值平方的均方根"),
    "spectral_peak": FeatureDescription("Spectral peak", "谱峰值", "frequency", "频谱幅值的最大值"),
    "spectral_crest_factor": FeatureDescription("Spectral crest factor", "谱峰值因子", "frequency", "谱峰与谱均方根之比"),
    "total_energy": FeatureDescription("Total spectral energy", "总能量", "frequency", "频域能量总和"),
}

_FAULT_TRANSLATIONS: Mapping[str, FeatureDescription] = {
    "ftf_band_energy": FeatureDescription(
        "FTF band energy",
        "保持架特征频带能量",
        "fault",
        "保持架（保持架公转）特征频率附近的能量聚集情况",
    ),
    "bpfo_band_energy": FeatureDescription("BPFO band energy", "外圈故障带能量", "fault", "外圈故障特征频带能量"),
    "bpfi_band_energy": FeatureDescription("BPFI band energy", "内圈故障带能量", "fault", "内圈故障特征频带能量"),
    "bsf_band_energy": FeatureDescription("BSF band energy", "滚动体故障带能量", "fault", "滚动体故障特征频带能量"),
    "ftf_band_ratio": FeatureDescription(
        "FTF energy ratio",
        "保持架频带能量占比",
        "fault",
        "保持架特征频带能量占总能量的比例",
    ),
    "bpfo_band_ratio": FeatureDescription("BPFO energy ratio", "外圈能量占比", "fault", "外圈特征频带能量占比"),
    "bpfi_band_ratio": FeatureDescription("BPFI energy ratio", "内圈能量占比", "fault", "内圈特征频带能量占比"),
    "bsf_band_ratio": FeatureDescription("BSF energy ratio", "滚动体能量占比", "fault", "滚动体特征频带能量占比"),
}

_TIME_FREQUENCY_TRANSLATIONS: Mapping[str, FeatureDescription] = {
    "tf_stft_total_energy": FeatureDescription(
        "STFT total energy",
        "STFT总能量",
        "time_frequency",
        "短时傅里叶变换幅度平方的总能量",
    ),
    "tf_stft_entropy": FeatureDescription(
        "STFT entropy",
        "STFT熵",
        "time_frequency",
        "STFT能量分布的熵值, 反映谱的聚集程度",
    ),
    "tf_stft_freq_mean": FeatureDescription(
        "STFT frequency mean",
        "STFT频率均值",
        "time_frequency",
        "STFT能量沿频率轴的加权平均",
    ),
    "tf_stft_freq_std": FeatureDescription(
        "STFT frequency std",
        "STFT频率标准差",
        "time_frequency",
        "STFT能量沿频率轴的离散度",
    ),
    "tf_stft_freq_skew": FeatureDescription(
        "STFT frequency skewness",
        "STFT频率偏度",
        "time_frequency",
        "STFT频率分布的不对称性",
    ),
    "tf_stft_freq_kurt": FeatureDescription(
        "STFT frequency kurtosis",
        "STFT频率峭度",
        "time_frequency",
        "STFT频率分布的尖锐程度",
    ),
    "tf_stft_time_mean": FeatureDescription(
        "STFT time mean",
        "STFT时间重心",
        "time_frequency",
        "STFT能量沿时间轴的加权平均",
    ),
    "tf_stft_time_std": FeatureDescription(
        "STFT time std",
        "STFT时间标准差",
        "time_frequency",
        "STFT能量沿时间轴的离散程度",
    ),
    "tf_stft_peak_frequency": FeatureDescription(
        "STFT peak frequency",
        "STFT峰值频率",
        "time_frequency",
        "STFT能量最大的频率位置",
    ),
    "tf_stft_peak_time": FeatureDescription(
        "STFT peak time",
        "STFT峰值时间",
        "time_frequency",
        "STFT能量最大的时间位置",
    ),
    "tf_cwt_total_energy": FeatureDescription(
        "CWT total energy",
        "CWT总能量",
        "time_frequency",
        "连续小波变换系数的能量总和",
    ),
    "tf_cwt_entropy": FeatureDescription(
        "CWT entropy",
        "CWT熵",
        "time_frequency",
        "CWT能量分布的熵值",
    ),
    "tf_cwt_scale_mean": FeatureDescription(
        "CWT scale mean",
        "CWT尺度均值",
        "time_frequency",
        "CWT能量沿尺度轴的加权平均",
    ),
    "tf_cwt_scale_std": FeatureDescription(
        "CWT scale std",
        "CWT尺度标准差",
        "time_frequency",
        "CWT能量沿尺度轴的离散程度",
    ),
    "tf_cwt_scale_skew": FeatureDescription(
        "CWT scale skewness",
        "CWT尺度偏度",
        "time_frequency",
        "CWT尺度能量分布的不对称性",
    ),
    "tf_cwt_scale_kurt": FeatureDescription(
        "CWT scale kurtosis",
        "CWT尺度峭度",
        "time_frequency",
        "CWT尺度能量分布的尖锐程度",
    ),
    "tf_cwt_max_scale": FeatureDescription(
        "CWT peak scale",
        "CWT峰值尺度",
        "time_frequency",
        "CWT能量最大的尺度位置",
    ),
    "tf_cwt_ridge_energy": FeatureDescription(
        "CWT ridge energy",
        "CWT能量脊",
        "time_frequency",
        "沿最大能量轨迹积分得到的能量, 反映主振动模态",
    ),
    "tf_mel_total_energy": FeatureDescription(
        "Mel total energy",
        "梅尔谱总能量",
        "time_frequency",
        "梅尔滤波后能量的总和, 体现感知频带的总体强度",
    ),
    "tf_mel_entropy": FeatureDescription(
        "Mel entropy",
        "梅尔谱熵",
        "time_frequency",
        "梅尔谱能量分布的熵值, 描述频带能量的均匀程度",
    ),
    "tf_mel_band_mean": FeatureDescription(
        "Mel band mean",
        "梅尔带中心",
        "time_frequency",
        "梅尔能量沿频带方向的加权平均频率",
    ),
    "tf_mel_band_std": FeatureDescription(
        "Mel band std",
        "梅尔带宽度",
        "time_frequency",
        "梅尔能量沿频带方向的离散程度",
    ),
    "tf_mel_band_skew": FeatureDescription(
        "Mel band skewness",
        "梅尔带偏度",
        "time_frequency",
        "梅尔频带能量分布的不对称性",
    ),
    "tf_mel_band_kurt": FeatureDescription(
        "Mel band kurtosis",
        "梅尔带峭度",
        "time_frequency",
        "梅尔频带能量分布的尖锐程度",
    ),
    "tf_mel_time_mean": FeatureDescription(
        "Mel time mean",
        "梅尔时间重心",
        "time_frequency",
        "梅尔能量沿时间轴的加权平均",
    ),
    "tf_mel_time_std": FeatureDescription(
        "Mel time std",
        "梅尔时间扩展",
        "time_frequency",
        "梅尔能量沿时间轴的离散程度",
    ),
    "tf_mel_peak_frequency": FeatureDescription(
        "Mel peak frequency",
        "梅尔峰值频率",
        "time_frequency",
        "梅尔谱能量最大的频带位置",
    ),
    "tf_mel_peak_time": FeatureDescription(
        "Mel peak time",
        "梅尔峰值时间",
        "time_frequency",
        "梅尔谱能量峰值对应的时间",
    ),
    "tf_mel_contrast": FeatureDescription(
        "Mel spectral contrast",
        "梅尔谱对比度",
        "time_frequency",
        "梅尔高能与低能频带的差值, 描述谱结构鲜明程度",
    ),
    "tf_consistency_time_stft": FeatureDescription(
        "Time-STFT mutual information",
        "时域与STFT互信息",
        "time_frequency",
        "时域包络与STFT时间能量分布的一致性指标",
    ),
    "tf_consistency_time_cwt": FeatureDescription(
        "Time-CWT mutual information",
        "时域与CWT互信息",
        "time_frequency",
        "时域包络与CWT时间能量分布的一致性指标",
    ),
    "tf_consistency_time_mel": FeatureDescription(
        "Time-Mel mutual information",
        "时域与梅尔谱互信息",
        "time_frequency",
        "时域包络与梅尔谱时间能量分布的一致性指标",
    ),
    "tf_consistency_stft_cwt": FeatureDescription(
        "STFT-CWT mutual information",
        "STFT与CWT互信息",
        "time_frequency",
        "STFT与CWT在时间维度上的能量一致性",
    ),
    "tf_consistency_stft_mel": FeatureDescription(
        "STFT-Mel mutual information",
        "STFT与梅尔谱互信息",
        "time_frequency",
        "STFT与梅尔谱在时间维度上的能量一致性",
    ),
    "tf_consistency_cwt_mel": FeatureDescription(
        "CWT-Mel mutual information",
        "CWT与梅尔谱互信息",
        "time_frequency",
        "CWT与梅尔谱在时间维度上的能量一致性",
    ),
    "tf_consistency_average": FeatureDescription(
        "Modal consistency score",
        "多模态一致性得分",
        "time_frequency",
        "多种时频模态互信息的平均值, 衡量跨模态鲁棒性",
    ),
}


def _translate_time_feature(column: str, prefix: str) -> Optional[FeatureDescription]:
    base = column[len(prefix) :]
    description = _TIME_BASE.get(base)
    if description is None:
        return None
    if prefix == "env_":
        # Envelope inherits time-domain semantics but属于 envelope category.
        return FeatureDescription(
            english=f"Envelope {description.english.lower()}",
            chinese=f"包络{description.chinese}",
            category="envelope",
            description=f"Hilbert包络信号的{description.chinese}",
        )
    return FeatureDescription(
        english=f"Time-domain {description.english.lower()}",
        chinese=f"时域{description.chinese}",
        category="time",
        description=f"原始时域信号的{description.chinese}",
    )


def _translate_envelope_extra(column: str) -> Optional[FeatureDescription]:
    base = column[len("env_") :]
    extra = _ENVELOPE_EXTRA.get(base)
    if extra is None:
        return None
    return extra


def _translate_frequency_feature(column: str) -> Optional[FeatureDescription]:
    base = column[len("freq_") :]
    description = _FREQUENCY_TRANSLATIONS.get(base)
    if description is None:
        return None
    return description


def _translate_fault_feature(column: str) -> Optional[FeatureDescription]:
    base = column[len("fault_") :]
    description = _FAULT_TRANSLATIONS.get(base)
    if description is None:
        return None
    return description


def _translate_time_frequency_feature(column: str) -> Optional[FeatureDescription]:
    return TIME_FREQUENCY_TRANSLATIONS.get(column)


TIME_FREQUENCY_TRANSLATIONS = dict(_TIME_FREQUENCY_TRANSLATIONS)


def build_feature_dictionary(columns: Iterable[str]) -> pd.DataFrame:
    """Return a DataFrame mapping feature codes to bilingual descriptions."""

    rows: List[Dict[str, str]] = []
    for column in columns:
        description: Optional[FeatureDescription] = None
        if column in _METADATA_TRANSLATIONS:
            description = _METADATA_TRANSLATIONS[column]
        elif column.startswith("time_"):
            description = _translate_time_feature(column, "time_")
        elif column.startswith("env_"):
            description = _translate_envelope_extra(column)
            if description is None:
                description = _translate_time_feature(column, "env_")
        elif column.startswith("freq_"):
            description = _translate_frequency_feature(column)
        elif column.startswith("fault_"):
            description = _translate_fault_feature(column)
        elif column in TIME_FREQUENCY_TRANSLATIONS:
            description = TIME_FREQUENCY_TRANSLATIONS[column]

        if description is None:
            description = FeatureDescription(
                english=column,
                chinese=column,
                category="other",
                description="暂未提供对应释义，请在报告中补充说明",
            )

        rows.append(
            {
                "feature": column,
                "english_name": description.english,
                "chinese_name": description.chinese,
                "category": description.category,
                "description": description.description,
            }
        )

    return pd.DataFrame(rows)


LABEL_DISPLAY_MAP: Mapping[str, str] = {
    "normal": "正常工况",
    "ball_fault": "滚动体故障",
    "inner_race_fault": "内圈故障",
    "outer_race_fault": "外圈故障",
    "ftf": "保持架特征频率",
    "bpfo": "外圈故障",
    "bpfi": "内圈故障",
    "bsf": "滚动体故障",
    "unlabelled": "未标注",
    "unknown": "未知类别",
}

LABEL_DISPLAY_ORDER: Sequence[str] = (
    "normal",
    "ball_fault",
    "inner_race_fault",
    "outer_race_fault",
)


def build_feature_name_map(columns: Iterable[str]) -> Dict[str, str]:
    """返回特征编码到中文名称的映射。"""

    dictionary = build_feature_dictionary(columns)
    return dict(zip(dictionary["feature"], dictionary["chinese_name"]))


def build_label_name_map(labels: Iterable[str]) -> Dict[str, str]:
    """将标签值转换为中文显示名称。"""

    mapping: Dict[str, str] = {}
    for label in labels:
        key = str(label)
        mapping[key] = LABEL_DISPLAY_MAP.get(key, key)
    return mapping


def translate_labels(labels: Iterable[str]) -> List[str]:
    """批量翻译标签列表。"""

    name_map = build_label_name_map(labels)
    return [name_map[str(label)] for label in labels]


__all__ = [
    "FeatureDescription",
    "build_feature_dictionary",
    "build_feature_name_map",
    "build_label_name_map",
    "translate_labels",
    "TIME_FREQUENCY_TRANSLATIONS",
    "LABEL_DISPLAY_MAP",
    "LABEL_DISPLAY_ORDER",
]
