"""Frequency-domain and envelope-domain feature computations."""
from __future__ import annotations

from typing import Dict, Mapping, Tuple

import numpy as np
from scipy import signal

from .statistics import time_domain_features

EPS = 1e-12


def _to_numpy(signal_array: np.ndarray) -> np.ndarray:
    return np.asarray(signal_array, dtype=float).ravel()


def _power_spectrum(array: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    windowed = array * np.hanning(array.size)
    spectrum = np.fft.rfft(windowed)
    freqs = np.fft.rfftfreq(array.size, d=1.0 / float(sampling_rate))
    power = np.square(np.abs(spectrum))
    return freqs, power


def frequency_domain_features(signal_array: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """Compute descriptive statistics of the amplitude spectrum."""

    array = _to_numpy(signal_array)
    array = array - np.mean(array)
    freqs, power = _power_spectrum(array, sampling_rate)
    power_sum = float(np.sum(power)) + EPS
    norm_power = power / power_sum

    centroid = float(np.sum(freqs * norm_power))
    spread = float(np.sqrt(np.sum(((freqs - centroid) ** 2) * norm_power)))
    skew = float(np.sum(((freqs - centroid) ** 3) * norm_power) / (spread**3 + EPS))
    kurt = float(np.sum(((freqs - centroid) ** 4) * norm_power) / (spread**4 + EPS))
    peak_index = int(np.argmax(power))
    peak_freq = float(freqs[peak_index])
    median_freq = float(freqs[np.searchsorted(np.cumsum(norm_power), 0.5)])

    # Spectral entropy is useful to quantify the presence of strong harmonics.
    spectral_entropy = float(-np.sum(norm_power * np.log(norm_power + EPS)))

    # Spectral crest factor compares the spectral peak to the RMS level.
    spectral_rms = float(np.sqrt(np.mean(np.square(power))))
    spectral_peak = float(np.max(power))

    return {
        "centroid": centroid,
        "spread": spread,
        "skewness": skew,
        "kurtosis": kurt,
        "peak_frequency": peak_freq,
        "median_frequency": median_freq,
        "spectral_entropy": spectral_entropy,
        "spectral_rms": spectral_rms,
        "spectral_peak": spectral_peak,
        "spectral_crest_factor": float(spectral_peak / (spectral_rms + EPS)),
        "total_energy": power_sum,
    }


def envelope_features(signal_array: np.ndarray, sampling_rate: float) -> Dict[str, float]:
    """Compute Hilbert-envelope based statistics."""

    array = _to_numpy(signal_array)
    analytic = signal.hilbert(array)
    envelope = np.abs(analytic)
    time_features = {f"env_{k}": v for k, v in time_domain_features(envelope).items()}

    freqs, power = _power_spectrum(envelope - np.mean(envelope), sampling_rate)
    peak_freq = float(freqs[int(np.argmax(power))]) if power.size else 0.0
    bandwidth = float(np.sqrt(np.sum(freqs**2 * power) / (np.sum(power) + EPS))) if power.size else 0.0

    spectral_entropy = 0.0
    if power.size:
        norm_power = power / (np.sum(power) + EPS)
        spectral_entropy = float(-np.sum(norm_power * np.log(norm_power + EPS)))

    time_features.update(
        {
            "env_peak_frequency": peak_freq,
            "env_bandwidth": bandwidth,
            "env_spectral_entropy": spectral_entropy,
        }
    )
    return time_features


def band_energy(features: Mapping[str, np.ndarray], band: Tuple[float, float]) -> float:
    freqs = features["freqs"]
    power = features["power"]
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.sum(power[mask]))


def fault_frequency_band_features(
    signal_array: np.ndarray,
    sampling_rate: float,
    fault_bands: Mapping[str, Tuple[float, float]],
) -> Dict[str, float]:
    array = _to_numpy(signal_array)
    array = array - np.mean(array)
    freqs, power = _power_spectrum(array, sampling_rate)
    features = {"freqs": freqs, "power": power}
    total_energy = float(np.sum(power)) + EPS

    band_features: Dict[str, float] = {}
    for name, band in fault_bands.items():
        energy = band_energy(features, band)
        band_features[f"{name}_band_energy"] = energy
        band_features[f"{name}_band_ratio"] = float(energy / total_energy)
    return band_features


def welch_density(signal_array: np.ndarray, sampling_rate: float) -> tuple[np.ndarray, np.ndarray]:
    freqs, psd = signal.welch(signal_array, fs=sampling_rate, window="hann", nperseg=min(len(signal_array), 4096))
    return freqs, psd


def welch_band_energy(
    freqs: np.ndarray, psd: np.ndarray, band: Tuple[float, float]
) -> float:
    mask = (freqs >= band[0]) & (freqs <= band[1])
    if not np.any(mask):
        return 0.0
    return float(np.trapz(psd[mask], freqs[mask]))


__all__ = [
    "frequency_domain_features",
    "envelope_features",
    "fault_frequency_band_features",
    "welch_density",
    "welch_band_energy",
]
