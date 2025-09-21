"""Time-domain statistical feature computations."""
from __future__ import annotations

from typing import Dict

import numpy as np
from scipy import stats


EPS = 1e-12


def _safe_divide(numerator: float, denominator: float) -> float:
    if abs(denominator) < EPS:
        return 0.0
    return float(numerator) / float(denominator)


def time_domain_features(signal: np.ndarray) -> Dict[str, float]:
    """Compute a suite of descriptive statistics for a vibration signal."""

    array = np.asarray(signal, dtype=float).ravel()
    if array.size == 0:
        raise ValueError("signal must contain at least one sample")

    mean = float(np.mean(array))
    std = float(np.std(array, ddof=0))
    var = float(np.var(array, ddof=0))
    rms = float(np.sqrt(np.mean(np.square(array))))
    abs_mean = float(np.mean(np.abs(array)))
    peak = float(np.max(np.abs(array)))
    peak_to_peak = float(np.ptp(array))
    skewness = float(stats.skew(array, bias=False)) if array.size > 2 else 0.0
    kurtosis = float(stats.kurtosis(array, fisher=False, bias=False)) if array.size > 3 else 3.0

    mean_crossing_rate = 0.0
    if array.size > 1:
        centered = array - mean
        signs = np.signbit(centered)
        mean_crossing_rate = float(np.count_nonzero(signs[:-1] != signs[1:]) / (array.size - 1))

    sqrt_abs_mean = float(np.mean(np.sqrt(np.abs(array))))

    features: Dict[str, float] = {
        "mean": mean,
        "std": std,
        "var": var,
        "rms": rms,
        "abs_mean": abs_mean,
        "peak": peak,
        "peak_to_peak": peak_to_peak,
        "skewness": skewness,
        "kurtosis": kurtosis,
        "mean_crossing_rate": mean_crossing_rate,
        "shape_factor": _safe_divide(rms, abs_mean),
        "crest_factor": _safe_divide(peak, rms),
        "impulse_factor": _safe_divide(peak, abs_mean),
        "clearance_factor": _safe_divide(peak, sqrt_abs_mean**2),
        "squared_mean": float(np.mean(np.square(array))),
    }

    # Higher order statistics help differentiate impulsive faults.
    absolute_deviation = float(np.mean(np.abs(array - mean)))
    features.update(
        {
            "absolute_deviation": absolute_deviation,
            "variance_ratio": _safe_divide(var, features["squared_mean"]),
            "signal_to_noise_ratio": _safe_divide(rms, absolute_deviation),
        }
    )

    return features


__all__ = ["time_domain_features"]
