import numpy as np
import pytest

from pv_emi.models.conducted_emission import (
    predict_cm_conducted_emission,
    predict_conducted_emission,
    predict_dm_conducted_emission,
)


def test_current_divider_equation():
    f = np.array([150e3, 1e6, 10e6])
    source_current = np.array([1, 2, 3]) * 1e-3 * (1 + 1j)
    zs = np.array([10 + 2j, 20 + 5j, 30 + 10j])
    zl = np.array([40 + 3j, 50 + 7j, 60 + 9j])
    zp = np.array([1j, 2j, 3j])

    result = predict_conducted_emission(f, source_current, zs, zl, zp)
    expected = source_current * zs / (zs + zl + zp)
    np.testing.assert_allclose(result.current_a, expected)
    np.testing.assert_allclose(result.load_voltage_v, expected * zl)


def test_zero_path_impedance_is_default():
    result = predict_conducted_emission([1e6], [2], [10], [10])
    np.testing.assert_allclose(result.current_a, [1])


def test_cm_and_dm_helpers_set_mode():
    cm = predict_cm_conducted_emission([1e6], [2], [10], [10])
    dm = predict_dm_conducted_emission([1e6], [2], [10], [10])
    assert cm.mode == "CM"
    assert dm.mode == "DM"
    np.testing.assert_allclose(cm.current_a, dm.current_a)


def test_current_dbua_property():
    result = predict_conducted_emission([1e6], [1e-3], [10], [10])
    np.testing.assert_allclose(result.current_magnitude_dbua, [60 + 20*np.log10(0.5)])


def test_rejects_mismatched_array_lengths():
    with pytest.raises(ValueError, match="expected"):
        predict_conducted_emission([1e6, 2e6], [1], [10, 10], [10, 10])


def test_rejects_zero_total_impedance():
    with pytest.raises(ZeroDivisionError):
        predict_conducted_emission([1e6], [1], [10], [-10])


def test_rejects_nonpositive_frequency():
    with pytest.raises(ValueError, match="positive"):
        predict_conducted_emission([0], [1], [10], [10])
