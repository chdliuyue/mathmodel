"""Time-frequency feature generation utilities for task 3."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import logging

import numpy as np
from scipy import signal

LOGGER = logging.getLogger(__name__)


EPS = 1e-12


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


def compute_stft_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    """Return descriptive features derived from the STFT magnitude."""

    nperseg = min(config.stft_nperseg, signal_array.size)
    if nperseg < 16:
        LOGGER.debug("Segment too short for STFT (%s samples)", signal_array.size)
        return {key: 0.0 for key in (
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
        return {}

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

    return {
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


def _cwt_wavelet(wavelet: str):
    wavelet_lower = wavelet.lower()
    if wavelet_lower == "ricker":
        return signal.ricker
    if wavelet_lower in {"morlet", "morlet2"}:
        return lambda points, scale: signal.morlet2(points, s=scale, w=6.0)
    raise ValueError(f"Unsupported wavelet type: {wavelet}")


def compute_cwt_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    """Return descriptive features derived from the CWT scalogram."""

    if config.cwt_num_scales <= 1 or config.cwt_max_scale <= config.cwt_min_scale:
        return {key: 0.0 for key in (
            "tf_cwt_total_energy",
            "tf_cwt_entropy",
            "tf_cwt_scale_mean",
            "tf_cwt_scale_std",
            "tf_cwt_scale_skew",
            "tf_cwt_scale_kurt",
            "tf_cwt_max_scale",
            "tf_cwt_ridge_energy",
        )}

    scales = np.linspace(config.cwt_min_scale, config.cwt_max_scale, config.cwt_num_scales)
    wavelet = _cwt_wavelet(config.cwt_wavelet)
    try:
        coefficients = signal.cwt(signal_array - np.mean(signal_array), wavelet, scales)
    except Exception as exc:
        LOGGER.warning("CWT computation failed: %s", exc)
        return {}

    energy = np.square(np.abs(coefficients))
    weights, total = _normalise_weights(energy)
    flat_weights = weights.reshape(-1)
    entropy = float(-np.sum(flat_weights * np.log(flat_weights + EPS)))

    scale_weights = weights.sum(axis=1)
    scale_mean, scale_std, scale_skew, scale_kurt = _moment_statistics(scales, scale_weights)

    ridge_indices = np.argmax(energy, axis=0)
    max_scales = scales[ridge_indices]
    ridge_energy = float(np.sum(energy[ridge_indices, np.arange(energy.shape[1])]))

    peak_scale = float(max_scales.mean()) if max_scales.size else 0.0

    return {
        "tf_cwt_total_energy": total,
        "tf_cwt_entropy": entropy,
        "tf_cwt_scale_mean": scale_mean,
        "tf_cwt_scale_std": scale_std,
        "tf_cwt_scale_skew": scale_skew,
        "tf_cwt_scale_kurt": scale_kurt,
        "tf_cwt_max_scale": peak_scale,
        "tf_cwt_ridge_energy": ridge_energy,
    }


def extract_time_frequency_features(signal_array: np.ndarray, sampling_rate: float, config: TimeFrequencyConfig) -> Dict[str, float]:
    """Combine STFT and CWT statistics into a single feature dictionary."""

    stft_features = compute_stft_features(signal_array, sampling_rate, config)
    cwt_features = compute_cwt_features(signal_array, sampling_rate, config)
    features = {**stft_features, **cwt_features}
    return {key: float(value) for key, value in features.items()}


__all__ = [
    "TimeFrequencyConfig",
    "compute_stft_features",
    "compute_cwt_features",
    "extract_time_frequency_features",
]
