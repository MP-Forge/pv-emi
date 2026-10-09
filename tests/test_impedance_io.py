import numpy as np
import pytest

from pv_emi.io.impedance import interpolate_complex


def test_interpolate_complex_matches_linear_real_and_imaginary_parts():
    source_f = np.array([1.0, 2.0, 3.0])
    source_z = np.array([1 + 2j, 3 + 4j, 5 + 8j])
    target_f = np.array([1.5, 2.5])

    result = interpolate_complex(source_f, source_z, target_f)

    np.testing.assert_allclose(result, [2 + 3j, 4 + 6j])


def test_interpolation_outside_range_returns_nan_like_matlab_interp1():
    source_f = np.array([1.0, 2.0, 3.0])
    source_z = np.array([1 + 1j, 2 + 2j, 3 + 3j])
    target_f = np.array([0.5, 1.5, 3.5])

    result = interpolate_complex(source_f, source_z, target_f)

    assert np.isnan(result[0].real)
    assert np.isnan(result[0].imag)
    assert result[1] == 1.5 + 1.5j
    assert np.isnan(result[2].real)
    assert np.isnan(result[2].imag)


def test_frequency_must_be_strictly_increasing():
    with pytest.raises(ValueError):
        interpolate_complex(
            np.array([1.0, 2.0, 2.0]),
            np.array([1 + 1j, 2 + 2j, 3 + 3j]),
            np.array([1.5]),
        )
