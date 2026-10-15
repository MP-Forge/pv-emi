"""Frequency-dependent complex impedance."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .frequency import FrequencyGrid


@dataclass(frozen=True)
class Impedance:
    """Complex impedance as a function of frequency.

    Frequency is stored in Hz and impedance in ohms.
    """

    frequency: FrequencyGrid
    value: np.ndarray
    name: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        value = np.asarray(self.value, dtype=complex)
        if value.ndim != 1:
            raise ValueError("Impedance values must be one-dimensional")
        if len(value) != len(self.frequency):
            raise ValueError("Frequency and impedance arrays must have equal length")
        if not np.all(np.isfinite(value)):
            raise ValueError("Impedance contains non-finite values")
        object.__setattr__(self, "value", value)

    @property
    def magnitude(self) -> np.ndarray:
        return np.abs(self.value)

    @property
    def phase_deg(self) -> np.ndarray:
        return np.angle(self.value, deg=True)

    @property
    def real(self) -> np.ndarray:
        return self.value.real

    @property
    def imag(self) -> np.ndarray:
        return self.value.imag

    def __add__(self, other: Impedance) -> Impedance:
        self._check_compatible(other)
        return Impedance(self.frequency, self.value + other.value, name=self.name)

    def __sub__(self, other: Impedance) -> Impedance:
        self._check_compatible(other)
        return Impedance(self.frequency, self.value - other.value, name=self.name)

    def scale(self, factor: complex, name: str | None = None) -> Impedance:
        return Impedance(self.frequency, self.value * factor, name=name or self.name)

    def _check_compatible(self, other: Impedance) -> None:
        if not np.array_equal(self.frequency.values, other.frequency.values):
            raise ValueError("Impedances must use the same frequency grid")

    def interpolate(self, frequency: FrequencyGrid) -> Impedance:
        values = frequency.interpolate(self.frequency.values, self.value)
        return Impedance(frequency, values, name=self.name, metadata=dict(self.metadata))
