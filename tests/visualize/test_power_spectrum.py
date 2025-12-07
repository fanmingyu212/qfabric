import numpy as np

from qfabric.visualizer.power_spectrum import PowerSpectrum


def test_power_spectrum_shapes_and_monotonic_frequency():
    num_samples = 4096
    dt = 1e-6  # 1 MHz sampling
    t = np.arange(num_samples) * dt
    # arbitrary trace
    trace = np.sin(2 * np.pi * 50e3 * t) + 0.3 * np.sin(2 * np.pi * 120e3 * t)

    ps = PowerSpectrum(trace, dt)

    # basic shape checks
    assert ps.f.ndim == 1
    assert ps.power_spectrum.ndim == 1
    assert ps.f.shape == ps.power_spectrum.shape

    # frequencies should be strictly positive and sorted
    assert np.all(ps.f > 0)
    assert np.all(np.diff(ps.f) > 0)


def test_power_spectrum_peak_at_expected_frequency():
    num_samples = 8192
    dt = 1e-6  # 1 MHz sampling
    f0 = 123e3
    t = np.arange(num_samples) * dt
    trace = np.sin(2 * np.pi * f0 * t)

    ps = PowerSpectrum(trace, dt)

    peak_index = np.argmax(ps.power_spectrum)
    f_peak = ps.f[peak_index]

    # within a few frequency bins
    df = ps.f[1] - ps.f[0]
    assert abs(f_peak - f0) < 3 * df
