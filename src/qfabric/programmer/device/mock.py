from typing import Any

import matplotlib.pyplot as plt
import numpy as np

from qfabric.planner.segmenter.mock import MockSegment, MockSegmenter
from qfabric.programmer.device import Device


class MockDevice(Device):
    """
    A mock :class:`~qfabric.programmer.device.Device` that shows a plot of the sequence programmed.

    Args:
        segmenter (MockSegmenter): Segmenter for this AWG device.
        resource (str): Resource name of the device. This is unused as it is a mock device.
        show_plot (bool): Whether shows a plot of the analog and digital data.
    """

    def __init__(self, segmenter: MockSegmenter, resource: str, show_plot: bool):
        super().__init__(segmenter, resource)
        self._show_plot = show_plot
        if self.is_principal_device:
            self._external_trigger = False
        else:
            self._external_trigger = True

    def program_memory(self, instructions: dict[str, Any]):
        self._segments: list[MockSegment] = instructions["segments"]

    def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
        self._segment_indices_and_repeats = segment_indices_and_repeats

    def start(self):
        analog_channels = self._segmenter._analog_channels
        digital_channels = self._segmenter._digital_channels
        channel_num = len(analog_channels) + len(digital_channels)

        self.fig, axes = plt.subplots(channel_num, 1, sharex=True)
        axes[0].set_title(f"device {self._resource}")
        axes[-1].set_xlabel("time (s)")

        time_offset = 0
        for kk, (segment_index, segment_repeat) in enumerate(self._segment_indices_and_repeats):
            segment = self._segments[segment_index]
            times = np.arange(segment._sample_size) / self._segmenter._sample_rate

            for _ in range(segment_repeat):
                channel_index = 0
                for ll, analog_data in enumerate(segment.analog_data):
                    axes[channel_index].plot(times + time_offset, analog_data)
                    if kk == 0:
                        analog_channel = self._segmenter._analog_channels[ll]
                        axes[channel_index].set_ylabel(f"Analog\nCH{analog_channel}")
                    channel_index += 1

                for ll, digital_data in enumerate(segment.digital_data):
                    axes[channel_index].plot(times + time_offset, digital_data)
                    if kk == 0:
                        digital_channel = self._segmenter._digital_channels[ll]
                        axes[channel_index].set_ylabel(f"Digital\nCH{digital_channel}")
                    channel_index += 1

                time_offset += segment._sample_size / self._segmenter._sample_rate
        if self._show_plot:
            plt.show()

    def wait_until_complete(self):
        pass

    def stop(self):
        try:
            plt.close(self.fig)
        except Exception:
            pass

    def setup_external_trigger(self):
        self._external_trigger = True

    def setup_software_trigger(self):
        self._external_trigger = False
