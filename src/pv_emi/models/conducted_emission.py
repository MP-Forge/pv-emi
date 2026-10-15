"""Conducted-emission prediction using a Norton-equivalent noise source.

The model assumes a Norton source current in parallel with source impedance Zs,
feeding a load impedance ZL through an optional *series* path impedance Zpath.

This is a deliberately small model. A real EMI filter with shunt branches or
multiple ports should be represented by a network/two-port model rather than
being treated as a single series impedance.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ConductedEmissionResult:
    """Predicted frequency-domain current and the corresponding load voltage."""

    frequency_hz: np.ndarray
    current_a: np.ndarray
    load_voltage_v: np.ndarray
    mode: str = "unknown"

    @property
    def current_magnitude_a(self) -> np.ndarray:
        """Magnitude of predicted current, in amperes."""
        return np.abs(self.current_a)

    @property
    def current_magnitude_dbua(self) -> np.ndarray:
        """Predicted current magnitude in dBµA."""
        with np.errstate(divide="ignore"):
            return 20.0 * np.log10(np.abs(self.current_a) / 1e-6)

    @property
    def load_voltage_magnitude_v(self) -> np.ndarray:
        """Magnitude of voltage across the specified load, in volts."""
        return np.abs(self.load_voltage_v)


def _complex_vector(value, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=complex)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if not np.isfinite(array.real).all() or not np.isfinite(array.imag).all():
        raise ValueError(f"{name} must contain only finite values.")
    return array


def predict_conducted_emission(
    frequency_hz,
    noise_source_current_a,
    source_impedance_ohm,
    load_impedance_ohm,
    path_impedance_ohm=None,
    *,
    mode: str = "unknown",
) -> ConductedEmissionResult:
    """Predict current through a load from a Norton-equivalent noise source.

    The current-divider equation is

        I_load = I_source * Z_source / (Z_source + Z_path + Z_load)

    where all currents and impedances are complex frequency-domain quantities.
    ``path_impedance_ohm`` represents only a series path impedance. If omitted,
    it is treated as zero.

    The returned load voltage is ``I_load * Z_load``. Current direction follows
    the assumed Norton source orientation and the passive load convention.
    """
    frequency = np.asarray(frequency_hz, dtype=float)
    if frequency.ndim != 1 or frequency.size == 0:
        raise ValueError("frequency_hz must be a non-empty one-dimensional array.")
    if not np.isfinite(frequency).all() or np.any(frequency <= 0):
        raise ValueError("frequency_hz must contain finite, positive values.")

    source_current = _complex_vector(noise_source_current_a, "noise_source_current_a")
    source_z = _complex_vector(source_impedance_ohm, "source_impedance_ohm")
    load_z = _complex_vector(load_impedance_ohm, "load_impedance_ohm")
    path_z = (
        np.zeros_like(source_z)
        if path_impedance_ohm is None
        else _complex_vector(path_impedance_ohm, "path_impedance_ohm")
    )

    expected = frequency.size
    for name, values in (
        ("noise_source_current_a", source_current),
        ("source_impedance_ohm", source_z),
        ("load_impedance_ohm", load_z),
        ("path_impedance_ohm", path_z),
    ):
        if values.size != expected:
            raise ValueError(f"{name} has length {values.size}; expected {expected}.")

    total_impedance = source_z + path_z + load_z
    if np.any(total_impedance == 0):
        raise ZeroDivisionError(
            "source_impedance + path_impedance + load_impedance is zero "
            "at one or more frequency points."
        )

    current = source_current * source_z / total_impedance
    voltage = current * load_z

    return ConductedEmissionResult(
        frequency_hz=frequency.copy(),
        current_a=current,
        load_voltage_v=voltage,
        mode=mode,
    )


def predict_cm_conducted_emission(
    frequency_hz,
    noise_source_current_a,
    source_impedance_ohm,
    load_impedance_ohm,
    path_impedance_ohm=None,
) -> ConductedEmissionResult:
    """Predict common-mode conducted current for the simple series-path model."""
    return predict_conducted_emission(
        frequency_hz,
        noise_source_current_a,
        source_impedance_ohm,
        load_impedance_ohm,
        path_impedance_ohm,
        mode="CM",
    )


def predict_dm_conducted_emission(
    frequency_hz,
    noise_source_current_a,
    source_impedance_ohm,
    load_impedance_ohm,
    path_impedance_ohm=None,
) -> ConductedEmissionResult:
    """Predict differential-mode conducted current for the simple series-path model."""
    return predict_conducted_emission(
        frequency_hz,
        noise_source_current_a,
        source_impedance_ohm,
        load_impedance_ohm,
        path_impedance_ohm,
        mode="DM",
    )
