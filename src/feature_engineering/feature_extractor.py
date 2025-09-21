"""High level helpers for vibration feature extraction."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from .bearing import BearingSpec
from .spectral import envelope_features, fault_frequency_band_features, frequency_domain_features
from .statistics import time_domain_features


def _prefix(features: Dict[str, float], prefix: str) -> Dict[str, float]:
    """Prefix feature dictionary keys to keep column names unique."""

    return {f"{prefix}{key}": value for key, value in features.items()}


@dataclass
class FeatureExtractorConfig:
    """Configuration flags controlling which feature families are computed."""

    include_frequency_domain: bool = True
    include_envelope_domain: bool = True
    include_fault_bands: bool = True
    fault_bandwidth: float = 5.0


class FeatureExtractor:
    """Bundle together the different feature computations used in task 1."""

    def __init__(self, config: FeatureExtractorConfig | None = None):
        self.config = config or FeatureExtractorConfig()

    def extract(
        self,
        signal: np.ndarray,
        sampling_rate: float,
        rpm: Optional[float] = None,
        bearing: Optional[BearingSpec] = None,
    ) -> Dict[str, float]:
        """Return a dictionary of features for ``signal``."""

        features: Dict[str, float] = {}
        features.update(_prefix(time_domain_features(signal), "time_"))

        if self.config.include_frequency_domain:
            features.update(_prefix(frequency_domain_features(signal, sampling_rate), "freq_"))

        if self.config.include_envelope_domain:
            features.update(_prefix(envelope_features(signal, sampling_rate), "env_"))

        if self.config.include_fault_bands and rpm is not None and bearing is not None:
            bands = bearing.fault_frequency_bands(rpm, bandwidth=self.config.fault_bandwidth)
            freq_values = {f"{name}_frequency": float(value) for name, value in bearing.fault_frequencies(rpm).items()}
            features.update(_prefix(freq_values, "fault_"))
            features.update(_prefix(fault_frequency_band_features(signal, sampling_rate, bands), "fault_"))
        else:
            defaults = {
                "fault_ftf_frequency": 0.0,
                "fault_bpfo_frequency": 0.0,
                "fault_bpfi_frequency": 0.0,
                "fault_bsf_frequency": 0.0,
                "fault_ftf_band_energy": 0.0,
                "fault_bpfo_band_energy": 0.0,
                "fault_bpfi_band_energy": 0.0,
                "fault_bsf_band_energy": 0.0,
                "fault_ftf_band_ratio": 0.0,
                "fault_bpfo_band_ratio": 0.0,
                "fault_bpfi_band_ratio": 0.0,
                "fault_bsf_band_ratio": 0.0,
            }
            for key, value in defaults.items():
                features.setdefault(key, value)

        return features


__all__ = ["FeatureExtractor", "FeatureExtractorConfig"]
