"""Time-frequency feature generation utilities for task 3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Sequence, Tuple

import logging

import numpy as np
from scipy import signal

LOGGER = logging.getLogger(__name__)


EPS = 1e-12
_RICKER_FALLBACK_WARNED = False
_CWT_FALLBACK_WARNED = False


@dataclass
class TimeFrequencyConfig:
    """Configuration controlling time-frequency feature extraction."""

    stft_nperseg: int = 512
    stft_noverlap: int = 256
    stft_window: str = "hann"
    stft_nfft: int = 1024
    cwt_wavelet: str = "ricker"
    cwt_min_scale: float = 1.0
    cwt_max_scale: float = 128.0
    cwt_num_scales: int = 64
    mel_num_mels: int = 48
    mel_fmin: float = 20.0
    mel_fmax: float = 20000.0
    mel_power: float = 2.0
    mel_log_amplitude: bool = True
    consistency_bins: int = 48


@dataclass
class STFTRepresentation:
    frequencies: np.ndarray
    times: np.ndarray
    magnitude: np.ndarray
    energy: np.ndarray


@dataclass
class CWTRepresentation:
    scales: np.ndarray
    times: np.ndarray
    magnitude: np.ndarray
    coefficients: np.ndarray
    energy: np.ndarray


@dataclass
class MelRepresentation:
    frequencies: np.ndarray
    times: np.ndarray
    magnitude: np.ndarray
    energy: np.ndarray


@dataclass
class MultimodalRepresentation:
    """Bundle containing all intermediate matrices for multi-modal visualisation."""

    stft: STFTRepresentation
    cwt: CWTRepresentation
    mel: MelRepresentation
    time_axis: np.ndarray


def _normalise_weights(energy: np.ndarray) -> Tuple[np.ndarray, float]:
    total = float(np.sum(energy))
    if not np.isfinite(total) or total <= EPS:
        return np.zeros_like(energy), 0.0
    weights = energy / (total + EPS)
    return weights, total


def _moment_statistics(values: np.ndarray, weights: np.ndarray) -> Tuple[float, float, float, float]:
    mean = float(np.sum(values * weights))
    variance = float(np.sum(((values - mean) ** 2) * weights))
    std = float(np.sqrt(max(variance, 0.0)))
    if std <= EPS:
        skew = 0.0
        kurt = 0.0
    else:
        skew = float(np.sum(((values - mean) ** 3) * weights) / (std**3 + EPS))
        kurt = float(np.sum(((values - mean) ** 4) * weights) / (std**4 + EPS))
    return mean, std, skew, kurt


def _compute_stft(
    signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig
) -> Tuple[Dict[str, float], Optional[STFTRepresentation]]:
    nperseg = min(config.stft_nperseg, signal_array.size)
    if nperseg < 16:
        LOGGER.debug("Segment too short for STFT (%s samples)", signal_array.size)
        default = {key: 0.0 for key in (
            "tf_stft_total_energy",
            "tf_stft_entropy",
            "tf_stft_freq_mean",
            "tf_stft_freq_std",
            "tf_stft_freq_skew",
            "tf_stft_freq_kurt",
            "tf_stft_time_mean",
            "tf_stft_time_std",
            "tf_stft_peak_frequency",
            "tf_stft_peak_time",
        )}
        return default, None

    noverlap = min(config.stft_noverlap, nperseg - 1)
    nfft = max(config.stft_nfft, nperseg)

    try:
        freqs, times, Zxx = signal.stft(
            signal_array,
            fs=sampling_rate,
            window=config.stft_window,
            nperseg=nperseg,
            noverlap=noverlap,
            nfft=nfft,
            boundary=None,
        )
    except Exception as exc:
        LOGGER.warning("STFT computation failed: %s", exc)
        return {}, None

    magnitude = np.abs(Zxx)
    energy = np.square(magnitude)
    weights, total = _normalise_weights(energy)
    flat_weights = weights.reshape(-1)
    entropy = float(-np.sum(flat_weights * np.log(flat_weights + EPS)))

    freq_weights = weights.sum(axis=1)
    freq_mean, freq_std, freq_skew, freq_kurt = _moment_statistics(freqs, freq_weights)

    time_weights = weights.sum(axis=0)
    time_mean, time_std, _, _ = _moment_statistics(times, time_weights)

    peak_index = np.unravel_index(int(np.argmax(energy)), energy.shape)
    peak_frequency = float(freqs[peak_index[0]]) if freqs.size else 0.0
    peak_time = float(times[peak_index[1]]) if times.size else 0.0

    features = {
        "tf_stft_total_energy": total,
        "tf_stft_entropy": entropy,
        "tf_stft_freq_mean": freq_mean,
        "tf_stft_freq_std": freq_std,
        "tf_stft_freq_skew": freq_skew,
        "tf_stft_freq_kurt": freq_kurt,
        "tf_stft_time_mean": time_mean,
        "tf_stft_time_std": time_std,
        "tf_stft_peak_frequency": peak_frequency,
        "tf_stft_peak_time": peak_time,
    }
    representation = STFTRepresentation(frequencies=freqs, times=times, magnitude=magnitude, energy=energy)
    return features, representation


def _ricker_fallback(points: int, scale: float) -> np.ndarray:
    if scale <= 0:
        raise ValueError("Scale parameter for the Ricker wavelet must be positive")
    points = int(points)
    if points <= 0:
        return np.zeros(0, dtype=float)
    half_width = (points - 1) / 2.0
    x = np.linspace(-half_width, half_width, points)
    xsq = (x / scale) ** 2
    prefactor = 2.0 / (np.sqrt(3.0 * scale) * (np.pi ** 0.25))
    return prefactor * (1 - xsq) * np.exp(-xsq / 2.0)


def _cwt_wavelet(wavelet: str):
    wavelet_lower = wavelet.lower()
    if wavelet_lower == "ricker":
        if hasattr(signal, "ricker"):
            return signal.ricker
        global _RICKER_FALLBACK_WARNED
        if not _RICKER_FALLBACK_WARNED:
            LOGGER.warning("scipy.signal.ricker unavailable; using analytical fallback implementation.")
            _RICKER_FALLBACK_WARNED = True
        return _ricker_fallback
    if wavelet_lower in {"morlet", "morlet2"}:
        return lambda points, scale: signal.morlet2(points, s=scale, w=6.0)
    raise ValueError(f"Unsupported wavelet type: {wavelet}")


def get_cwt_wavelet(name: str) -> Callable[[int, float], np.ndarray]:
    """Expose the configured wavelet generator for external consumers."""

    return _cwt_wavelet(name)


def _fallback_cwt(
    data: np.ndarray,
    wavelet: Callable[[int, float], np.ndarray],
    scales: np.ndarray,
) -> np.ndarray:
    """Numerically approximate the CWT when SciPy does not expose it."""

    global _CWT_FALLBACK_WARNED
    if not _CWT_FALLBACK_WARNED:
        LOGGER.warning("scipy.signal.cwt unavailable; using numerical fallback implementation.")
        _CWT_FALLBACK_WARNED = True

    samples = data.size
    if samples == 0 or scales.size == 0:
        return np.zeros((scales.size, samples), dtype=np.complex128)

    coefficients = np.zeros((scales.size, samples), dtype=np.complex128)
    use_fft = hasattr(signal, "fftconvolve")
    for idx, scale in enumerate(scales):
        scale = float(scale)
        if not np.isfinite(scale) or scale <= 0:
            raise ValueError("CWT scales must be positive finite values")
        try:
            wavelet_samples = wavelet(samples, scale)
        except TypeError:
            wavelet_samples = wavelet(int(samples), float(scale))
        kernel = np.asarray(wavelet_samples, dtype=np.complex128).reshape(-1)
        if kernel.size != samples:
            target = np.linspace(0, kernel.size - 1, samples)
            original = np.arange(kernel.size)
            real_part = np.interp(target, original, np.real(kernel))
            if np.iscomplexobj(kernel):
                imag_part = np.interp(target, original, np.imag(kernel))
                kernel = real_part + 1j * imag_part
            else:
                kernel = real_part
        kernel = np.conjugate(kernel[::-1])
        if use_fft:
            conv = signal.fftconvolve(data, kernel, mode="same")
        else:  # pragma: no cover - NumPy fall-back
            conv = np.convolve(data, kernel, mode="same")
        coefficients[idx] = conv / np.sqrt(scale)

    return coefficients


def continuous_wavelet_transform(
    signal_array: np.ndarray,
    wavelet: Callable[[int, float], np.ndarray],
    scales: np.ndarray,
) -> np.ndarray:
    """Compute the CWT coefficients using SciPy if available, otherwise fall back."""

    data = np.asarray(signal_array, dtype=float)
    data = data - np.mean(data)
    if data.size == 0:
        return np.zeros((len(scales), 0), dtype=np.complex128)

    scales = np.asarray(scales, dtype=float)
    if np.any(~np.isfinite(scales)):
        raise ValueError("Scales must be finite real numbers")

    if hasattr(signal, "cwt"):
        try:
            return signal.cwt(data, wavelet, scales)
        except Exception as exc:  # pragma: no cover - defensive fallback
            LOGGER.debug("scipy.signal.cwt failed (%s); falling back to numerical implementation.", exc)
    return _fallback_cwt(data, wavelet, scales)


def compute_stft_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    features, _ = _compute_stft(signal_array, sampling_rate, config)
    return features


def _compute_cwt(
    signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig
) -> Tuple[Dict[str, float], Optional[CWTRepresentation]]:
    if config.cwt_num_scales <= 1 or config.cwt_max_scale <= config.cwt_min_scale:
        default = {key: 0.0 for key in (
            "tf_cwt_total_energy",
            "tf_cwt_entropy",
            "tf_cwt_scale_mean",
            "tf_cwt_scale_std",
            "tf_cwt_scale_skew",
            "tf_cwt_scale_kurt",
            "tf_cwt_max_scale",
            "tf_cwt_ridge_energy",
        )}
        return default, None

    scales = np.linspace(config.cwt_min_scale, config.cwt_max_scale, config.cwt_num_scales)
    wavelet = _cwt_wavelet(config.cwt_wavelet)
    try:
        coefficients = continuous_wavelet_transform(signal_array, wavelet, scales)
    except Exception as exc:
        LOGGER.warning("CWT computation failed: %s", exc)
        return {}, None

    magnitude = np.abs(coefficients)
    energy = np.square(magnitude)
    weights, total = _normalise_weights(energy)
    flat_weights = weights.reshape(-1)
    entropy = float(-np.sum(flat_weights * np.log(flat_weights + EPS)))

    scale_weights = weights.sum(axis=1)
    scale_mean, scale_std, scale_skew, scale_kurt = _moment_statistics(scales, scale_weights)

    ridge_indices = np.argmax(energy, axis=0)
    max_scales = scales[ridge_indices]
    ridge_energy = float(np.sum(energy[ridge_indices, np.arange(energy.shape[1])]))
    peak_scale = float(max_scales.mean()) if max_scales.size else 0.0

    time_axis = np.arange(coefficients.shape[1]) / float(sampling_rate if sampling_rate > 0 else 1.0)

    features = {
        "tf_cwt_total_energy": total,
        "tf_cwt_entropy": entropy,
        "tf_cwt_scale_mean": scale_mean,
        "tf_cwt_scale_std": scale_std,
        "tf_cwt_scale_skew": scale_skew,
        "tf_cwt_scale_kurt": scale_kurt,
        "tf_cwt_max_scale": peak_scale,
        "tf_cwt_ridge_energy": ridge_energy,
    }
    representation = CWTRepresentation(scales=scales, times=time_axis, magnitude=magnitude, coefficients=coefficients, energy=energy)
    return features, representation


def compute_cwt_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    features, _ = _compute_cwt(signal_array, sampling_rate, config)
    return features


def _hz_to_mel(hz: np.ndarray) -> np.ndarray:
    hz = np.asarray(hz, dtype=float)
    return 2595.0 * np.log10(1.0 + np.maximum(hz, 0.0) / 700.0)


def _mel_to_hz(mel: np.ndarray) -> np.ndarray:
    mel = np.asarray(mel, dtype=float)
    return 700.0 * (10 ** (mel / 2595.0) - 1.0)


def _build_mel_filterbank(
    frequencies: np.ndarray,
    sampling_rate: float,
    num_mels: int,
    fmin: float,
    fmax: float,
) -> Tuple[np.ndarray, np.ndarray]:
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if frequencies.size == 0 or num_mels <= 0:
        return np.zeros((0, frequencies.size)), np.zeros(0)

    nyquist = float(sampling_rate) / 2.0 if sampling_rate > 0 else frequencies.max()
    lower = float(max(0.0, min(fmin, nyquist)))
    upper = float(min(max(lower + EPS, fmax), nyquist if nyquist > 0 else frequencies.max()))

    mel_points = np.linspace(_hz_to_mel(lower), _hz_to_mel(upper), num_mels + 2)
    hz_points = _mel_to_hz(mel_points)

    filter_bank = np.zeros((num_mels, frequencies.size), dtype=float)
    for index in range(1, num_mels + 1):
        left = hz_points[index - 1]
        centre = hz_points[index]
        right = hz_points[index + 1]
        if centre <= left or right <= centre:
            continue
        left_mask = (frequencies >= left) & (frequencies < centre)
        right_mask = (frequencies >= centre) & (frequencies <= right)
        if np.any(left_mask):
            filter_bank[index - 1, left_mask] = (frequencies[left_mask] - left) / (centre - left + EPS)
        if np.any(right_mask):
            filter_bank[index - 1, right_mask] = (right - frequencies[right_mask]) / (right - centre + EPS)

    mel_frequencies = hz_points[1:-1]
    return filter_bank, mel_frequencies


def _compute_mel(
    signal_array: np.ndarray,
    sampling_rate: float,
    config: TimeFrequencyConfig,
    stft_repr: Optional[STFTRepresentation],
) -> Tuple[Dict[str, float], Optional[MelRepresentation]]:
    if stft_repr is None:
        stft_features, stft_repr = _compute_stft(signal_array, sampling_rate, config)
        if stft_repr is None:
            return stft_features, None

    filter_bank, mel_frequencies = _build_mel_filterbank(
        stft_repr.frequencies,
        sampling_rate,
        config.mel_num_mels,
        config.mel_fmin,
        config.mel_fmax,
    )
    if filter_bank.size == 0:
        return {key: 0.0 for key in (
            "tf_mel_total_energy",
            "tf_mel_entropy",
            "tf_mel_band_mean",
            "tf_mel_band_std",
            "tf_mel_band_skew",
            "tf_mel_band_kurt",
            "tf_mel_time_mean",
            "tf_mel_time_std",
            "tf_mel_peak_frequency",
            "tf_mel_peak_time",
            "tf_mel_contrast",
        )}, None

    base = stft_repr.magnitude ** float(max(config.mel_power, 1.0))
    mel_spectrogram = filter_bank @ base
    mel_energy = np.maximum(mel_spectrogram, 0.0)
    if config.mel_log_amplitude:
        mel_display = np.log1p(mel_energy)
    else:
        mel_display = mel_energy

    weights, total = _normalise_weights(mel_energy)
    flat_weights = weights.reshape(-1)
    entropy = float(-np.sum(flat_weights * np.log(flat_weights + EPS)))

    band_weights = weights.sum(axis=1)
    band_mean, band_std, band_skew, band_kurt = _moment_statistics(mel_frequencies, band_weights)

    time_weights = weights.sum(axis=0)
    time_mean, time_std, _, _ = _moment_statistics(stft_repr.times, time_weights)

    peak_index = np.unravel_index(int(np.argmax(mel_energy)), mel_energy.shape)
    peak_frequency = float(mel_frequencies[peak_index[0]]) if mel_frequencies.size else 0.0
    peak_time = float(stft_repr.times[peak_index[1]]) if stft_repr.times.size else 0.0

    if mel_energy.size:
        high = np.percentile(mel_energy, 90, axis=0)
        low = np.percentile(mel_energy, 10, axis=0)
        contrast = float(np.mean(high - low))
    else:
        contrast = 0.0

    features = {
        "tf_mel_total_energy": total,
        "tf_mel_entropy": entropy,
        "tf_mel_band_mean": band_mean,
        "tf_mel_band_std": band_std,
        "tf_mel_band_skew": band_skew,
        "tf_mel_band_kurt": band_kurt,
        "tf_mel_time_mean": time_mean,
        "tf_mel_time_std": time_std,
        "tf_mel_peak_frequency": peak_frequency,
        "tf_mel_peak_time": peak_time,
        "tf_mel_contrast": contrast,
    }
    representation = MelRepresentation(
        frequencies=mel_frequencies,
        times=stft_repr.times,
        magnitude=mel_display,
        energy=mel_energy,
    )
    return features, representation


def compute_mel_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    features, _ = _compute_mel(signal_array, sampling_rate, config, None)
    return features


def _project_time_profile(signal_array: np.ndarray, target_length: int) -> np.ndarray:
    if target_length <= 0:
        return np.zeros(0, dtype=float)
    samples = np.asarray(signal_array, dtype=float).ravel()
    if samples.size == 0:
        return np.zeros(target_length, dtype=float)
    absolute = np.abs(samples)
    indices = np.linspace(0, absolute.size - 1, target_length)
    return np.interp(indices, np.arange(absolute.size), absolute)


def _prepare_profile_for_mi(profile: np.ndarray) -> np.ndarray:
    values = np.asarray(profile, dtype=float).ravel()
    if values.size == 0:
        return values
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    values = np.maximum(values, 0.0)
    if np.allclose(values, 0.0):
        return values
    values = values / (values.sum() + EPS)
    return np.log1p(values * values.size)


def _profile_mutual_information(profile_a: np.ndarray, profile_b: np.ndarray, bins: int) -> float:
    a = _prepare_profile_for_mi(profile_a)
    b = _prepare_profile_for_mi(profile_b)
    if a.size < 2 or b.size < 2 or np.allclose(a, 0.0) or np.allclose(b, 0.0):
        return 0.0
    bins = int(max(4, min(bins, max(16, int(np.sqrt(min(a.size, b.size))) * 2))))
    hist, _, _ = np.histogram2d(a, b, bins=bins)
    total = float(np.sum(hist))
    if total <= 0:
        return 0.0
    pxy = hist / total
    px = pxy.sum(axis=1)
    py = pxy.sum(axis=0)
    px_py = px[:, None] * py[None, :]
    mask = pxy > 0
    if not np.any(mask):
        return 0.0
    values = pxy[mask]
    reference = px_py[mask]
    return float(np.sum(values * np.log((values + EPS) / (reference + EPS))))


def _compute_consistency_features(
    signal_array: np.ndarray,
    sampling_rate: float,
    stft_repr: Optional[STFTRepresentation],
    cwt_repr: Optional[CWTRepresentation],
    mel_repr: Optional[MelRepresentation],
    config: TimeFrequencyConfig,
) -> Dict[str, float]:
    base_keys = [
        "tf_consistency_time_stft",
        "tf_consistency_time_cwt",
        "tf_consistency_time_mel",
        "tf_consistency_stft_cwt",
        "tf_consistency_stft_mel",
        "tf_consistency_cwt_mel",
        "tf_consistency_average",
    ]
    features = {key: 0.0 for key in base_keys}
    if stft_repr is None or cwt_repr is None or mel_repr is None:
        return features

    target_length = stft_repr.times.size
    if target_length == 0:
        return features

    bins = max(int(config.consistency_bins), 8)
    time_profile = _project_time_profile(signal_array, target_length)
    stft_profile = stft_repr.energy.sum(axis=0)
    cwt_profile = cwt_repr.energy.sum(axis=0)
    mel_profile = mel_repr.energy.sum(axis=0)

    features["tf_consistency_time_stft"] = _profile_mutual_information(time_profile, stft_profile, bins)
    features["tf_consistency_time_cwt"] = _profile_mutual_information(time_profile, cwt_profile, bins)
    features["tf_consistency_time_mel"] = _profile_mutual_information(time_profile, mel_profile, bins)
    features["tf_consistency_stft_cwt"] = _profile_mutual_information(stft_profile, cwt_profile, bins)
    features["tf_consistency_stft_mel"] = _profile_mutual_information(stft_profile, mel_profile, bins)
    features["tf_consistency_cwt_mel"] = _profile_mutual_information(cwt_profile, mel_profile, bins)

    valid = [value for key, value in features.items() if key != "tf_consistency_average" and value > 0]
    features["tf_consistency_average"] = float(np.mean(valid)) if valid else 0.0
    return features


def _compute_features_and_representation(
    signal_array: np.ndarray,
    sampling_rate: float,
    config: TimeFrequencyConfig,
    need_representation: bool,
) -> Tuple[Dict[str, float], Optional[MultimodalRepresentation]]:
    stft_features, stft_repr = _compute_stft(signal_array, sampling_rate, config)
    cwt_features, cwt_repr = _compute_cwt(signal_array, sampling_rate, config)
    mel_features, mel_repr = _compute_mel(signal_array, sampling_rate, config, stft_repr)
    consistency_features = _compute_consistency_features(
        signal_array,
        sampling_rate,
        stft_repr,
        cwt_repr,
        mel_repr,
        config,
    )

    features: Dict[str, float] = {}
    features.update(stft_features)
    features.update(cwt_features)
    features.update(mel_features)
    features.update(consistency_features)

    representation: Optional[MultimodalRepresentation] = None
    if need_representation and stft_repr is not None and cwt_repr is not None and mel_repr is not None:
        time_axis = np.arange(signal_array.size) / float(sampling_rate if sampling_rate > 0 else 1.0)
        representation = MultimodalRepresentation(
            stft=stft_repr,
            cwt=cwt_repr,
            mel=mel_repr,
            time_axis=time_axis,
        )

    return {key: float(value) for key, value in features.items()}, representation


def extract_time_frequency_features(
    signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig
) -> Dict[str, float]:
    features, _ = _compute_features_and_representation(signal_array, sampling_rate, config, need_representation=False)
    return features


def compute_multimodal_representation(
    signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig
) -> Optional[MultimodalRepresentation]:
    _, representation = _compute_features_and_representation(signal_array, sampling_rate, config, need_representation=True)
    return representation


__all__ = [
    "TimeFrequencyConfig",
    "STFTRepresentation",
    "CWTRepresentation",
    "MelRepresentation",
    "MultimodalRepresentation",
    "compute_stft_features",
    "compute_cwt_features",
    "compute_mel_features",
    "continuous_wavelet_transform",
    "compute_multimodal_representation",
    "get_cwt_wavelet",
    "extract_time_frequency_features",
]
