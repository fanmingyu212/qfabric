import numpy as np
import numpy.typing as npt

from qfabric.sequence.function import AnalogFunction, DigitalFunction


# Analog Functions
class SineWave(AnalogFunction):
    """
    Sine wave.

    Args:
        frequency (float): Cyclic frequency of the sine wave.
        amplitude (float): Amplitude of the sine wave.
        phase (float): Phase of the sine wave. Default 0.
        start_time (float):
            Start time. If None, it starts from the beginning of the step. Default None.
        stop_time (float):
            Stop time. If None, it stops at the end of the step. Default None.
    """

    frequency: float
    amplitude: float
    phase: float
    start_time: float
    stop_time: float

    def __init__(
        self,
        frequency: float,
        amplitude: float,
        phase: float = 0,
        start_time: float = None,
        stop_time: float = None,
    ):
        self.frequency: float = frequency
        self.amplitude: float = amplitude
        self.phase: float = phase
        self.start_time: float = start_time
        self.stop_time: float = stop_time

    @property
    def min_duration(self) -> float:
        if self.stop_time is None:
            if self.start_time is None:
                return 0
            else:
                return self.start_time
        else:
            return self.stop_time

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        if self.start_time is not None:
            condlist.append(times < self.start_time)
            funclist.append(0)
        if self.stop_time is not None:
            condlist.append(times >= self.stop_time)
            funclist.append(0)

        def sine(ts):
            inst_phases = 2 * np.pi * self.frequency * (ts + time_offset) + self.phase
            return self.amplitude * np.sin(inst_phases)

        funclist.append(sine)
        if len(condlist) == 0:
            return sine(times)
        else:
            return np.piecewise(times, condlist, funclist)


class SineSweep(AnalogFunction):
    """
    Sine frequency and/or amplitude sweep.

    Args:
        start_frequency (float): Cyclic frequency at start.
        stop_frequency (float): Cyclic frequency at stop.
        start_amplitude (float): Amplitude at start.
        stop_amplitude (float): Amplitude at stop.
        start_time (float): Start time.
        stop_time (float): Stop time.
        phase (float): Phase of the sine sweep. Default 0.
    """

    start_frequency: float
    stop_frequency: float
    start_amplitude: float
    stop_amplitude: float
    start_time: float
    stop_time: float
    phase: float

    def __init__(
        self,
        start_frequency: float,
        stop_frequency: float,
        start_amplitude: float,
        stop_amplitude: float,
        start_time: float,
        stop_time: float,
        phase: float = 0,
    ):
        self.start_frequency = start_frequency
        self.stop_frequency = stop_frequency
        self.start_amplitude = start_amplitude
        self.stop_amplitude = stop_amplitude
        self.start_time = start_time
        self.stop_time = stop_time
        self.phase = phase

    @property
    def min_duration(self) -> float:
        return self.stop_time

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        condlist.append((times < self.start_time) | (times >= self.stop_time))
        funclist.append(0)

        def sine_sweep(ts):
            frequency_change = self.stop_frequency - self.start_frequency
            amplitude_change = self.stop_amplitude - self.start_amplitude
            duration = self.stop_time - self.start_time
            freq_sweep_rate = frequency_change / duration
            amplitude_sweep_rate = amplitude_change / duration
            # here we set the time offset to shift the instantaneous phase according to the start frequency.
            # Note that the instantaneous frequency sweeps twice as fast as the phase,
            # thus the np.pi factor without 2 on the second line.
            inst_phases = (
                2 * np.pi * self.start_frequency * (ts + time_offset)
                + np.pi * freq_sweep_rate * (ts - self.start_time) ** 2
                + self.phase
            )
            inst_amplitudes = self.start_amplitude + amplitude_sweep_rate * (ts - self.start_time)
            return inst_amplitudes * np.sin(inst_phases)

        funclist.append(sine_sweep)
        return np.piecewise(times, condlist, funclist)


class ConstantAnalog(AnalogFunction):
    """
    Constant analog function.

    Args:
        amplitude (float): Amplitude.
        start_time (float):
            Start time. If None, it starts from the beginning of the step. Default None.
        stop_time (float):
            Stop time. If None, it stops at the end of the step. Default None.
    """

    amplitude: float
    start_time: float
    stop_time: float

    def __init__(
        self,
        amplitude: float,
        start_time: float = None,
        stop_time: float = None,
    ):
        self.amplitude = amplitude
        self.start_time = start_time
        self.stop_time = stop_time

    @property
    def min_duration(self) -> float:
        if self.stop_time is None:
            if self.start_time is None:
                return 0
            else:
                return self.start_time
        else:
            return self.stop_time

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        if self.start_time is not None:
            condlist.append(times < self.start_time)
            funclist.append(0)
        if self.stop_time is not None:
            condlist.append(times >= self.stop_time)
            funclist.append(0)
        funclist.append(self.amplitude)
        if len(condlist) == 0:
            return self.amplitude * np.ones(len(times))
        else:
            return np.piecewise(times, condlist, funclist)


class LinearRamp(AnalogFunction):
    """
    Linear ramp function.

    Args:
        start_amplitude (float): Start amplitude.
        stop_amplitude (float): Stop amplitude.
        start_time (float): Start time.
        stop_time (float): Stop time.
    """

    start_amplitude: float
    stop_amplitude: float
    start_time: float
    stop_time: float

    def __init__(
        self,
        start_amplitude: float,
        stop_amplitude: float,
        start_time: float,
        stop_time: float,
    ):
        self.start_amplitude = start_amplitude
        self.stop_amplitude = stop_amplitude
        self.start_time = start_time
        self.stop_time = stop_time

    @property
    def min_duration(self) -> float:
        return self.stop_time

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        condlist.append((times < self.start_time) | (times >= self.stop_time))
        funclist.append(0)

        def linear_ramp(ts):
            amplitude_change = self.stop_amplitude - self.start_amplitude
            duration = self.stop_time - self.start_time
            amplitude_sweep_rate = amplitude_change / duration
            return self.start_amplitude + amplitude_sweep_rate * (ts - self.start_time)

        funclist.append(linear_ramp)
        return np.piecewise(times, condlist, funclist)


class HammingWindow(AnalogFunction):
    """
    Hamming Window function.

    This is usually used with another :class:`AnalogFunction` in :class:`AnalogProduct`
    to produce an analog pulse without sinc-square lumps.

    Args:
        amplitude (float): Maximum amplitude of the window. Default 1.
        start_time (float):
            Start time. If None, it starts from the beginning of the step. Default None.
        stop_time (float):
            Stop time. If None, it stops at the end of the step. Default None.
    """

    amplitude: float
    start_time: float
    stop_time: float

    def __init__(self, amplitude: float = 1, start_time: float = None, stop_time: float = None):
        self.amplitude = amplitude
        self.start_time = start_time
        self.stop_time = stop_time

    @property
    def min_duration(self) -> float:
        if self.stop_time is None:
            if self.start_time is None:
                return 0
            else:
                return self.start_time
        else:
            return self.stop_time

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        if self.start_time is None:
            start_time = 0
        else:
            start_time = self.start_time
        if self.stop_time is None:
            stop_time = times[-1]
        else:
            stop_time = self.stop_time
        alpha = 0.54
        T = stop_time - start_time
        return self.amplitude * (alpha - (1 - alpha) * np.cos(2 * np.pi * (times - start_time) / T))


# Digital Functions
class DigitalOn(DigitalFunction):
    """
    Digital constant true.
    """

    @property
    def min_duration(self) -> float:
        return 0

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.ones(len(times), dtype=bool)


class DigitalOff(DigitalFunction):
    """
    Digital constant false.
    """

    @property
    def min_duration(self) -> float:
        return 0

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        return np.zeros(len(times), dtype=bool)


class DigitalPulse(DigitalFunction):
    """
    Digital pulse.

    Args:
        start_time (float): Start time of the pulse.
        stop_time (float): Stop time of the pulse.
        default_off (bool):
            Whether to output false when the pulse is off.
            This switch flips the polarity of the output. Default True.
    """

    start_time: float
    stop_time: float
    default_off: bool

    def __init__(self, start_time: float, stop_time: float, default_off: bool = True):
        self.start_time = start_time
        self.stop_time = stop_time
        self.default_off = default_off

    @property
    def min_duration(self) -> float:
        return self.stop_time

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        condlist.append((times < self.start_time) | (times >= self.stop_time))
        funclist.append(not self.default_off)
        funclist.append(self.default_off)
        return np.piecewise(times, condlist, funclist).astype(bool)


class DigitalMultiPulses(DigitalFunction):
    """
    Multiple digital pulses.

    Args:
        start_times (npt.NDArray[np.float64]): Start times of the pulses.
        stop_time (npt.NDArray[np.float64]): Stop times of the pulses.
        default_off (bool):
            Whether to output false when the pulse is off.
            This switch flips the polarity of the output. Default True.
    """

    start_times: npt.NDArray[np.float64]
    stop_times: npt.NDArray[np.float64]
    default_off: bool

    def __init__(
        self,
        start_times: npt.NDArray[np.float64],
        stop_times: npt.NDArray[np.float64],
        default_off: bool = True,
    ):
        self.start_times = start_times
        self.stop_times = stop_times
        if len(self.start_times) != len(self.stop_times):
            raise ValueError("start_times and stop_times must have the same length.")
        self.default_off = default_off

    @property
    def min_duration(self) -> float:
        return np.max(self.stop_times)

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        condlist = []
        funclist = []
        condition_pulse = np.zeros(len(times), dtype=bool)
        for kk in range(len(self.start_times)):
            condition_pulse |= (times >= self.start_times[kk]) & (times < self.stop_times[kk])
        condlist.append(condition_pulse)
        funclist.append(self.default_off)
        funclist.append(not self.default_off)
        return np.piecewise(times, condlist, funclist).astype(bool)
