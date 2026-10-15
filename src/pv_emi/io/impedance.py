"""Input/output helpers for measured impedance data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
from scipy.io import loadmat

from pv_emi.core.impedance import Impedance


def interpolate_complex(
    source_frequency: np.ndarray,
    source_value: np.ndarray,
    target_frequency: np.ndarray,
) -> np.ndarray:
    """Linearly interpolate complex data."""

    f = np.asarray(source_frequency, dtype=float).reshape(-1)
    z = np.asarray(source_value, dtype=complex).reshape(-1)
    target = np.asarray(target_frequency, dtype=float).reshape(-1)

    if f.size != z.size:
        raise ValueError("source_frequency and source_value must have equal length")
    if f.size < 2:
        raise ValueError("at least two source points are required")
    if np.any(~np.isfinite(f)):
        raise ValueError("source frequency contains non-finite values")
    if np.any(np.diff(f) <= 0):
        raise ValueError("source frequency must be strictly increasing")

    real = np.interp(target, f, z.real)
    imag = np.interp(target, f, z.imag)

    outside = (target < f[0]) | (target > f[-1])
    result = real + 1j * imag
    result[outside] = np.nan + 1j * np.nan
    return result


def load_impedance_csv(
    path: str | Path,
    *,
    frequency_column: int = 0,
    magnitude_column: int = 1,
    phase_column: int = 2,
    phase_unit: Literal["deg", "rad"] = "deg",
    skiprows: int = 0,
    delimiter: str = ",",
    name: str | None = None,
) -> Impedance:
    """Load frequency, magnitude and phase impedance data from CSV.

    The expected columns are, by default::

        frequency, magnitude, phase

    Magnitude is in ohms and phase is in degrees unless ``phase_unit="rad"``.
    """
    data = np.loadtxt(path, delimiter=delimiter, skiprows=skiprows)

    frequency = data[:, frequency_column]
    magnitude = data[:, magnitude_column]
    phase = data[:, phase_column]

    if phase_unit == "deg":
        value = magnitude * np.exp(1j * np.deg2rad(phase))
    elif phase_unit == "rad":
        value = magnitude * np.exp(1j * phase)
    else:
        raise ValueError("phase_unit must be 'deg' or 'rad'")

    return Impedance(
        frequency=frequency,
        value=value,
        name=name or Path(path).stem,
    )


def load_impedance_mat(
    path: str | Path,
    *,
    frequency_field: str = "Frequency",
    impedance_field: str = "BwoCable",
    name: str | None = None,
) -> Impedance:
    """Load an impedance vector from a MATLAB ``.mat`` file.
    Some data are in mat files.
    This supports the structure used by the original PV-EMI project, e.g.
    ``Frequency`` + ``BwoCable`` or ``Frequency`` + ``Impedance``.
    """
    raw = loadmat(path, squeeze_me=True)

    missing = [field for field in (frequency_field, impedance_field) if field not in raw]
    if missing:
        available = ", ".join(k for k in raw if not k.startswith("__"))
        raise KeyError(f"Missing MATLAB field(s): {missing}. Available fields: {available}")

    return Impedance(
        frequency=np.asarray(raw[frequency_field], dtype=float).reshape(-1),
        value=np.asarray(raw[impedance_field], dtype=complex).reshape(-1),
        name=name or Path(path).stem,
    )


def interpolate_impedance(
    impedance: Impedance,
    target_frequency: np.ndarray,
) -> Impedance:
    """Return an impedance interpolated onto a target frequency grid."""
    return Impedance(
        frequency=np.asarray(target_frequency, dtype=float),
        value=interpolate_complex(
            impedance.frequency,
            impedance.value,
            target_frequency,
        ),
        name=impedance.name,
    )
