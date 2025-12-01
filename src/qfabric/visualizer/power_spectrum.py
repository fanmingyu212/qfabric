import numpy as np
import numpy.typing as npt


class PowerSpectrum:
    """
    Calculates the power spectrum of a voltage signal.

    Args:
        num_of_samples (int): number of samples per trace
        time_resolution (float): error signal time resolution in s.
    """

    def __init__(self, signal: npt.NDArray[np.float64], time_resolution: float):
        self._num_of_samples = len(signal)
        self._time_resolution = time_resolution
        self._duration = time_resolution * self._num_of_samples
        self.f = self._calculate_frequencies()
        self.power_spectrum = self._voltages_to_power_spectrum(signal)

    def _calculate_frequencies(self):
        """
        Calculates the frequencies at which we will find the spectrum.
        """
        frequencies = np.fft.fftfreq(self._num_of_samples, self._time_resolution)
        self.fft_mask = frequencies > 0
        f = frequencies[self.fft_mask]
        return f

    def _voltages_to_power_spectrum(self, trace):
        """
        Calculate the power spectrum.
        """
        V_T = trace / np.sqrt(self._duration)
        V_f = np.fft.fft(V_T) * self._time_resolution
        W_V_calculated = np.abs(V_f) ** 2
        W_V = 2 * W_V_calculated[self.fft_mask]
        return W_V
