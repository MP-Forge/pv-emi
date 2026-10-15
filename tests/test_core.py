import numpy as np
import pytest

from pv_emi.core import FrequencyGrid, Impedance, NoiseSource
from pv_emi.network import TwoPortNetwork


def test_impedance_series_addition():
    f = FrequencyGrid(np.array([1e3, 2e3]))
    z1 = Impedance(f, np.array([10 + 1j, 20 + 2j]))
    z2 = Impedance(f, np.array([5 + 3j, 7 + 4j]))
    np.testing.assert_allclose((z1 + z2).value, [15 + 4j, 27 + 6j])


def test_impedance_subtraction():
    f = FrequencyGrid(np.array([1e3, 2e3]))
    z1 = Impedance(f, np.array([10 + 1j, 20 + 2j]))
    z2 = Impedance(f, np.array([5 + 3j, 7 + 4j]))
    np.testing.assert_allclose((z1 - z2).value, [5 - 2j, 13 - 2j])


def test_noise_source_voltage():
    f = FrequencyGrid(np.array([1e3, 2e3]))
    z = Impedance(f, np.array([10, 20], dtype=complex))
    source = NoiseSource(np.array([1, 2], dtype=complex), z, "CM")
    np.testing.assert_allclose(source.voltage, [10, 40])


def test_two_port_cascade_identity():
    f = FrequencyGrid(np.array([1e3, 2e3]))
    identity = TwoPortNetwork(f, np.ones(2), np.zeros(2), np.zeros(2), np.ones(2))
    dut = TwoPortNetwork(f, np.array([2, 3]), np.array([4, 5]), np.array([6, 7]), np.array([8, 9]))
    result = identity.cascade(dut)
    np.testing.assert_allclose(result.a, dut.a)
    np.testing.assert_allclose(result.b, dut.b)
    np.testing.assert_allclose(result.c, dut.c)
    np.testing.assert_allclose(result.d, dut.d)


def test_frequency_grid_rejects_unsorted_data():
    with pytest.raises(ValueError):
        FrequencyGrid(np.array([2e3, 1e3]))
