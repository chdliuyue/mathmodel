"""Bearing geometry utilities.

This module defines :class:`BearingSpec` which bundles together the key
geometrical parameters required to compute theoretical fault frequencies for
rolling element bearings.  These fault frequencies are routinely used in
condition monitoring because peaks around them (or their harmonics) in the
frequency spectrum often indicate specific defect types.  The implementation
follows the standard equations for ball bearings under radial load with a
contact angle that can be configured when known.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping


@dataclass(frozen=True)
class BearingSpec:
    """Geometry for a rolling element bearing.

    Attributes
    ----------
    name:
        Human readable bearing name.
    rolling_element_count:
        Number of rolling elements (balls or rollers).
    rolling_element_diameter:
        Diameter of a single rolling element.  Units cancel out in the
        frequency equations so inches or millimetres can be used as long as
        the same unit is applied to ``pitch_diameter``.
    pitch_diameter:
        Pitch diameter of the bearing (distance between the centres of
        opposing rolling elements).
    contact_angle_deg:
        Contact angle in degrees.  ``0`` corresponds to pure radial load.
    """

    name: str
    rolling_element_count: int
    rolling_element_diameter: float
    pitch_diameter: float
    contact_angle_deg: float = 0.0

    @property
    def geometry_factor(self) -> float:
        """Return the ratio ``(d / D) * cos(theta)`` used in fault equations."""

        if self.pitch_diameter == 0:
            raise ValueError("Pitch diameter must be non-zero to compute geometry factor.")
        ratio = self.rolling_element_diameter / self.pitch_diameter
        return ratio * math.cos(math.radians(self.contact_angle_deg))

    @staticmethod
    def rotation_frequency(rpm: float) -> float:
        """Convert revolutions-per-minute into Hz (revolutions-per-second)."""

        return float(rpm) / 60.0

    def fault_frequencies(self, rpm: float) -> Dict[str, float]:
        """Compute the classic bearing fault characteristic frequencies.

        Parameters
        ----------
        rpm:
            Shaft speed in revolutions-per-minute for the measurement.

        Returns
        -------
        dict
            Keys are ``ftf`` (fundamental train frequency), ``bpfo`` (ball
            pass frequency outer race), ``bpfi`` (ball pass frequency inner
            race) and ``bsf`` (ball spin frequency).  Units are Hz.
        """

        fr = self.rotation_frequency(rpm)
        g = self.geometry_factor
        g_sq = g * g

        ftf = 0.5 * fr * (1.0 - g)
        bpfo = 0.5 * self.rolling_element_count * fr * (1.0 - g)
        bpfi = 0.5 * self.rolling_element_count * fr * (1.0 + g)
        # Prevent a divide-by-zero if rolling_element_diameter is zero which is
        # physically unrealistic but may occur in badly curated metadata.
        if self.rolling_element_diameter == 0:
            raise ValueError("Rolling element diameter must be non-zero.")
        bsf = (
            (self.pitch_diameter / (2.0 * self.rolling_element_diameter))
            * fr
            * (1.0 - g_sq)
        )

        return {"ftf": ftf, "bpfo": bpfo, "bpfi": bpfi, "bsf": bsf}

    def fault_frequency_bands(self, rpm: float, bandwidth: float) -> Dict[str, tuple[float, float]]:
        """Return ±bandwidth frequency bands centred on each characteristic frequency."""

        base = self.fault_frequencies(rpm)
        return {name: (max(0.0, freq - bandwidth), freq + bandwidth) for name, freq in base.items()}


def default_bearing_library() -> Mapping[str, BearingSpec]:
    """Return bearing specifications extracted from the problem statement."""

    # Parameters sourced from the competition brief (in inches).
    return {
        "SKF6205": BearingSpec(
            name="SKF6205", rolling_element_count=9, rolling_element_diameter=0.3126, pitch_diameter=1.537
        ),
        "SKF6203": BearingSpec(
            name="SKF6203", rolling_element_count=9, rolling_element_diameter=0.2656, pitch_diameter=1.122
        ),
    }


DEFAULT_BEARINGS: Mapping[str, BearingSpec] = default_bearing_library()
"""Convenience mapping so callers do not have to repeatedly construct bearings."""

__all__ = ["BearingSpec", "default_bearing_library", "DEFAULT_BEARINGS"]
