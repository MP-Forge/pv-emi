import numpy as np
import pytest

from pv_emi.models.noise_source import (
    estimate_cm_noise_source_current,
    estimate_dm_noise_source_current,
    estimate_noise_source_current,
)

def test_noise_source_extraction_matches_equation():
    f = np.array([1e5, 2e5, 3e5])
    measured = np.array([1, 2, 3]) * 1e-3
    zs = np.array([2+1j, 3+2j, 4+3j])
    zpv = np.array([8+2j, 9+3j, 10+4j])
    result = estimate_noise_source_current(f, measured, zs, zpv)
    np.testing.assert_allclose(result.current_a, (zs + zpv) / zs * measured)

def test_path_impedance_is_included():
    result = estimate_noise_source_current([1e6], [1], [2], [3], [5])
    np.testing.assert_allclose(result.current_a, [5])

def test_dm_divides_measured_quantity_by_two():
    result = estimate_dm_noise_source_current([1e6], [4], [2], [2])
    np.testing.assert_allclose(result.current_a, [4])
    assert result.mode == "DM"

def test_cm_does_not_apply_factor_two():
    result = estimate_cm_noise_source_current([1e6], [2], [2], [2])
    np.testing.assert_allclose(result.current_a, [4])
    assert result.mode == "CM"

def test_mismatched_lengths_are_rejected():
    with pytest.raises(ValueError, match="expected"):
        estimate_noise_source_current([1e6, 2e6], [1], [2, 2], [3, 3])

def test_zero_source_impedance_is_rejected():
    with pytest.raises(ZeroDivisionError):
        estimate_noise_source_current([1e6], [1], [0], [3])
