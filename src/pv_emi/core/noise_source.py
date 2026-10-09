"""Equivalent EMI noise-source representation."""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .impedance import Impedance


@dataclass(frozen=True)
class NoiseSource:
    """Norton equivalent noise source.

    ``current`` is the Norton noise current in amperes. ``impedance`` is the
    source impedance in ohms. Both use the same frequency grid.
    """

    current: np.ndarray
    impedance: Impedance
    mode: str

    def __post_init__(self) -> None:
        current = np.asarray(self.current, dtype=complex)
        if current.ndim != 1:
            raise ValueError("Noise-source current must be one-dimensional")
        if len(current) != len(self.impedance.frequency):
            raise ValueError("Noise-source current and impedance must have equal length")
        if not np.all(np.isfinite(current)):
            raise ValueError("Noise-source current contains non-finite values")
        if self.mode not in {"CM", "DM"}:
            raise ValueError("Noise-source mode must be 'CM' or 'DM'")
        object.__setattr__(self, "current", current)

    @property
    def voltage(self) -> np.ndarray:
        """The equivalent Thevenin source voltage V = I_N Z_S."""
        return self.current * self.impedance.value
