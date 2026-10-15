"""Frequency-grid utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FrequencyGrid:
    """A one-dimensional frequency grid in Hz."""

    values: np.ndarray

    def __post_init__(self) -> None:
        values = np.asarray(self.values, dtype=float)
        if values.ndim != 1:
            raise ValueError("Frequency grid must be one-dimensional")
        if len(values) == 0:
            raise ValueError("Frequency grid cannot be empty")
        if not np.all(np.isfinite(values)):
            raise ValueError("Frequency grid contains non-finite values")
        if np.any(values <= 0):
            raise ValueError("Frequency values must be positive")
        if np.any(np.diff(values) <= 0):
            raise ValueError("Frequency values must be strictly increasing")
        object.__setattr__(self, "values", values)

    def __len__(self) -> int:
        return len(self.values)

    @property
    def start(self) -> float:
        return float(self.values[0])

    @property
    def stop(self) -> float:
        return float(self.values[-1])

    def contains(self, frequency_hz: float) -> bool:
        return self.start <= frequency_hz <= self.stop

    def interpolate(self, source_frequency_hz: np.ndarray, source_values: np.ndarray) -> np.ndarray:
        """Linearly interpolate complex or real data onto this grid."""

        source_frequency_hz = np.asarray(source_frequency_hz, dtype=float)
        source_values = np.asarray(source_values)
        if source_frequency_hz.ndim != 1 or source_values.ndim != 1:
            raise ValueError("Source frequency and data must be one-dimensional")
        if len(source_frequency_hz) != len(source_values):
            raise ValueError("Source frequency and data must have equal length")
        if np.any(np.diff(source_frequency_hz) <= 0):
            raise ValueError("Source frequency must be strictly increasing")
        if self.start < source_frequency_hz[0] or self.stop > source_frequency_hz[-1]:
            raise ValueError("Interpolation requested outside source frequency range")
        return (
            np.interp(self.values, source_frequency_hz, source_values.real)
            + 1j * np.interp(self.values, source_frequency_hz, source_values.imag)
            if np.iscomplexobj(source_values)
            else np.interp(self.values, source_frequency_hz, source_values)
        )
