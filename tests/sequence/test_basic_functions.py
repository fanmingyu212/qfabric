import numpy as np

from qfabric import (
    ConstantAnalog,
    DigitalMultiPulses,
    DigitalOff,
    DigitalOn,
    DigitalPulse,
    HammingWindow,
    LinearRamp,
    SineSweep,
    SineWave,
)


def test_SineWave():
    f = 10e6
    amp = 0.1
    sine_func = SineWave(frequency=f, amplitude=amp)
    np.testing.assert_allclose(sine_func.min_duration, 0)

    times = np.linspace(0, 1 / f, 4)
    out = sine_func.output(times)
    val = amp * np.sin(2 * np.pi * f * times)
    np.testing.assert_allclose(out, val)

    # shifts sine to cosine.
    time_offset = 1 / (4 * f)
    out = sine_func.output(times, time_offset=time_offset)
    val = amp * np.cos(2 * np.pi * f * times)
    np.testing.assert_allclose(out, val)

    out = sine_func.to_dict()
    val = {
        "import": {"module": "qfabric.sequence.basic_functions", "name": "SineWave"},
        "fields": {
            "frequency": 10000000.0,
            "amplitude": 0.1,
            "phase": 0,
            "start_time": None,
            "stop_time": None,
        },
    }
    assert out == val


def test_SineSweep():
    start_f = 1.0e6
    stop_f = 3.0e6
    start_amp = 0.2
    stop_amp = 0.8
    start_t = 0.0
    stop_t = 2.0e-6

    sweep = SineSweep(
        start_frequency=start_f,
        stop_frequency=stop_f,
        start_amplitude=start_amp,
        stop_amplitude=stop_amp,
        start_time=start_t,
        stop_time=stop_t,
    )

    # min_duration should be stop_time
    np.testing.assert_allclose(sweep.min_duration, stop_t)

    # Times spanning before, within, and after the sweep window
    times = np.linspace(-0.5e-6, 2.5e-6, 9)
    out = sweep.output(times)

    # Outside [start_t, stop_t) should be zero
    outside = (times < start_t) | (times >= stop_t)
    np.testing.assert_allclose(out[outside], 0.0)

    # Inside window should follow the implemented formula
    inside = ~outside
    ts = times[inside]

    frequency_change = stop_f - start_f
    amplitude_change = stop_amp - start_amp
    duration = stop_t - start_t
    freq_sweep_rate = frequency_change / duration
    amplitude_sweep_rate = amplitude_change / duration

    inst_phases = 2 * np.pi * start_f * ts + np.pi * freq_sweep_rate * (ts - start_t) ** 2
    inst_amplitudes = start_amp + amplitude_sweep_rate * (ts - start_t)
    val_inside = inst_amplitudes * np.sin(inst_phases)

    np.testing.assert_allclose(out[inside], val_inside)


def test_ConstantAnalog_no_window():
    amp = 0.3
    func = ConstantAnalog(amplitude=amp)
    np.testing.assert_allclose(func.min_duration, 0)

    times = np.linspace(0.0, 1e-6, 5)
    out = func.output(times)
    val = amp * np.ones_like(times)
    np.testing.assert_allclose(out, val)


def test_ConstantAnalog_with_window():
    amp = 0.3
    start_t = 1e-6
    stop_t = 3e-6
    func = ConstantAnalog(amplitude=amp, start_time=start_t, stop_time=stop_t)

    np.testing.assert_allclose(func.min_duration, stop_t)

    times = np.array([0.0, start_t, 2e-6, stop_t, 4e-6])
    out = func.output(times)

    val = np.array([0.0, amp, amp, 0.0, 0.0])
    np.testing.assert_allclose(out, val)


def test_ConstantAnalog_start_only():
    amp = 0.5
    start_t = 2e-6
    func = ConstantAnalog(amplitude=amp, start_time=start_t)

    np.testing.assert_allclose(func.min_duration, start_t)

    times = np.array([0.0, start_t - 1e-6, start_t, start_t + 1e-6])
    out = func.output(times)
    val = np.array([0.0, 0.0, amp, amp])
    np.testing.assert_allclose(out, val)


def test_LinearRamp():
    start_amp = 0.0
    stop_amp = 1.0
    start_t = 1e-6
    stop_t = 3e-6

    ramp = LinearRamp(
        start_amplitude=start_amp,
        stop_amplitude=stop_amp,
        start_time=start_t,
        stop_time=stop_t,
    )

    np.testing.assert_allclose(ramp.min_duration, stop_t)

    times = np.array([0.0, start_t, 2e-6, stop_t, 4e-6])
    out = ramp.output(times)

    # Expected: zero outside [start_t, stop_t), linear inside
    val = np.zeros_like(times)
    inside = (times >= start_t) & (times < stop_t)
    ts = times[inside]

    amplitude_change = stop_amp - start_amp
    duration = stop_t - start_t
    amplitude_sweep_rate = amplitude_change / duration
    val[inside] = start_amp + amplitude_sweep_rate * (ts - start_t)

    np.testing.assert_allclose(out, val)


def test_HammingWindow():
    amp = 1.0
    start_t = 0.0
    stop_t = 1.0
    func = HammingWindow(amplitude=amp, start_time=start_t, stop_time=stop_t)

    np.testing.assert_allclose(func.min_duration, stop_t)

    times = np.linspace(start_t, stop_t, 5)
    out = func.output(times)

    alpha = 0.54
    T = stop_t - start_t
    val = amp * (alpha - (1 - alpha) * np.cos(2 * np.pi * (times - start_t) / T))

    np.testing.assert_allclose(out, val)

    # Check endpoints: should be alpha - (1 - alpha) = 2*alpha - 1 = 0.08
    np.testing.assert_allclose(out[0], amp * (2 * alpha - 1))
    np.testing.assert_allclose(out[-1], amp * (2 * alpha - 1))


def test_DigitalOn():
    func = DigitalOn()
    np.testing.assert_allclose(func.min_duration, 0)

    times = np.array([0.0, 1e-6, 2e-6])
    out = func.output(times)
    val = np.ones(len(times), dtype=bool)
    np.testing.assert_equal(out, val)


def test_DigitalOff():
    func = DigitalOff()
    np.testing.assert_allclose(func.min_duration, 0)

    times = np.array([0.0, 1e-6, 2e-6])
    out = func.output(times)
    val = np.zeros(len(times), dtype=bool)
    np.testing.assert_equal(out, val)


def test_DigitalPulse_default_off_true():
    start_t = 1e-6
    stop_t = 3e-6
    func = DigitalPulse(start_time=start_t, stop_time=stop_t, default_off=True)

    np.testing.assert_allclose(func.min_duration, stop_t)

    times = np.array([0.0, start_t, 2e-6, stop_t, 4e-6])
    out = func.output(times)

    # default_off=True → outside window False, inside True
    val = np.array([False, True, True, False, False])
    np.testing.assert_equal(out, val)


def test_DigitalPulse_default_off_false():
    start_t = 1e-6
    stop_t = 3e-6
    func = DigitalPulse(start_time=start_t, stop_time=stop_t, default_off=False)

    np.testing.assert_allclose(func.min_duration, stop_t)

    times = np.array([0.0, start_t, 2e-6, stop_t, 4e-6])
    out = func.output(times)

    # default_off=False → outside window True, inside False
    val = np.array([True, False, False, True, True])
    np.testing.assert_equal(out, val)


def test_DigitalMultiPulses():
    start_times = [1e-6, 10e-6]
    stop_times = [2e-6, 15e-6]
    multi_pulses_func = DigitalMultiPulses(start_times, stop_times)
    np.testing.assert_allclose(multi_pulses_func.min_duration, 15e-6)
    times = np.array([0, 1e-6, 2e-6, 3e-6, 9e-6, 10e-6, 12e-6, 15e-6])
    out = multi_pulses_func.output(times)
    val = np.array([False, True, False, False, False, True, True, False])
    np.testing.assert_equal(out, val)
