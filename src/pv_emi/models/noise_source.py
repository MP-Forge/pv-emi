"""Noise-source extraction for the PV-EMI equivalent-source model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class NoiseSourceEstimate:
    """Frequency-domain equivalent noise-source current estimate."""

    frequency_hz: np.ndarray
    current_a: np.ndarray
    mode: str

    @property
    def magnitude_a(self) -> np.ndarray:
        return np.abs(self.current_a)


def _as_complex_array(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    return array


def estimate_noise_source_current(
    frequency_hz,
    measured_current_a,
    source_impedance_ohm,
    load_impedance_ohm,
    path_impedance_ohm=None,
    *,
    mode: str = "unknown",
    measured_current_scale: float = 1.0,
) -> NoiseSourceEstimate:
    """Implement I_source=((Z_source+Z_load+Z_path)/Z_source)*I_measured."""
    frequency = np.asarray(frequency_hz, dtype=float)
    measured = _as_complex_array(measured_current_a, "measured_current_a")
    source = _as_complex_array(source_impedance_ohm, "source_impedance_ohm")
    load = _as_complex_array(load_impedance_ohm, "load_impedance_ohm")
    path = (
        np.zeros_like(source)
        if path_impedance_ohm is None
        else _as_complex_array(path_impedance_ohm, "path_impedance_ohm")
    )

    if frequency.ndim != 1 or len(frequency) == 0:
        raise ValueError("frequency_hz must be a non-empty one-dimensional array.")
    if not np.isfinite(frequency).all():
        raise ValueError("frequency_hz contains non-finite values.")
    if not np.isfinite(measured_current_scale) or measured_current_scale == 0:
        raise ValueError("measured_current_scale must be finite and non-zero.")

    arrays = {
        "measured_current_a": measured,
        "source_impedance_ohm": source,
        "load_impedance_ohm": load,
        "path_impedance_ohm": path,
    }
    for name, array in arrays.items():
        if len(array) != len(frequency):
            raise ValueError(f"{name} has length {len(array)}; expected {len(frequency)}.")

    if np.any(source == 0):
        raise ZeroDivisionError("source_impedance_ohm contains zero.")

    current = ((source + load + path) / source) * (measured / measured_current_scale)
    return NoiseSourceEstimate(frequency, current, mode)


def estimate_cm_noise_source_current(
    frequency_hz,
    measured_current_a,
    source_impedance_ohm,
    pv_impedance_ohm,
    cable_impedance_ohm=None,
):
    """Estimate the CM equivalent noise-source current."""
    return estimate_noise_source_current(
        frequency_hz,
        measured_current_a,
        source_impedance_ohm,
        pv_impedance_ohm,
        cable_impedance_ohm,
        mode="CM",
    )


def estimate_dm_noise_source_current(
    frequency_hz,
    measured_current_a,
    source_impedance_ohm,
    pv_impedance_ohm,
    cable_impedance_ohm=None,
):
    """Estimate the DM source current; original MATLAB divides measured 2*I_DM by 2."""
    return estimate_noise_source_current(
        frequency_hz,
        measured_current_a,
        source_impedance_ohm,
        pv_impedance_ohm,
        cable_impedance_ohm,
        mode="DM",
        measured_current_scale=2.0,
    )
