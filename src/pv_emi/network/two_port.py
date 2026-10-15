"""Frequency-dependent two-port network representation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pv_emi.core.frequency import FrequencyGrid


@dataclass(frozen=True)
class TwoPortNetwork:
    """A frequency-dependent ABCD two-port network.

    Each ABCD element is an array over the same frequency grid.
    """

    frequency: FrequencyGrid
    a: np.ndarray
    b: np.ndarray
    c: np.ndarray
    d: np.ndarray
    name: str | None = None

    def __post_init__(self) -> None:
        arrays = [np.asarray(x, dtype=complex) for x in (self.a, self.b, self.c, self.d)]
        for x in arrays:
            if x.ndim != 1 or len(x) != len(self.frequency):
                raise ValueError("ABCD arrays must be one-dimensional and match frequency length")
            if not np.all(np.isfinite(x)):
                raise ValueError("ABCD arrays contain non-finite values")
        object.__setattr__(self, "a", arrays[0])
        object.__setattr__(self, "b", arrays[1])
        object.__setattr__(self, "c", arrays[2])
        object.__setattr__(self, "d", arrays[3])

    def matrix(self, index: int) -> np.ndarray:
        """Return the 2x2 ABCD matrix at one frequency index."""
        return np.array(
            [[self.a[index], self.b[index]], [self.c[index], self.d[index]]],
            dtype=complex,
        )

    def cascade(self, other: TwoPortNetwork, name: str | None = None) -> TwoPortNetwork:
        """Cascade two networks: T = T_self @ T_other."""
        if not np.array_equal(self.frequency.values, other.frequency.values):
            raise ValueError("Two-port networks must use the same frequency grid")
        a = self.a * other.a + self.b * other.c
        b = self.a * other.b + self.b * other.d
        c = self.c * other.a + self.d * other.c
        d = self.c * other.b + self.d * other.d
        return TwoPortNetwork(self.frequency, a, b, c, d, name=name)

    def inverse(self, name: str | None = None) -> TwoPortNetwork:
        """Return the matrix inverse at every frequency."""
        det = self.a * self.d - self.b * self.c
        if np.any(np.isclose(det, 0.0)):
            raise ValueError("Two-port network is singular at one or more frequencies")
        return TwoPortNetwork(
            self.frequency,
            self.d / det,
            -self.b / det,
            -self.c / det,
            self.a / det,
            name=name,
        )

    def deembed(
        self, input_network: TwoPortNetwork, output_network: TwoPortNetwork, name: str | None = None
    ) -> TwoPortNetwork:
        """Remove input and output networks from a measured cascade.
        T_dut = T_in^-1 @ T_sys @ T_out^-1
        """
        return input_network.inverse().cascade(self).cascade(output_network.inverse(), name=name)
