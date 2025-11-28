from typing import Any

import numpy as np
import numpy.typing as npt

from qfabric.planner.segmenter import Segment, Segmenter
from qfabric.sequence.step import DeviceStep, Step


class MockSegment(Segment):
    """
    Segment for a mock AWG device.
    """

    def __init__(
        self,
        device_step: DeviceStep,
        sample_rate: int,
        analog_channels: list[int],
        digital_channels: list[int],
    ):
        super().__init__(device_step)
        self.analog_data: list[npt.ArrayLike[np.float64]] = []
        self.digital_data: list[npt.ArrayLike[np.bool]] = []
        self._sample_size = self._device_step.duration * sample_rate
        self._get_data(sample_rate, analog_channels, digital_channels)

    def _get_data(self, sample_rate: int, analog_channels: list[int], digital_channels: list[int]):
        times = np.arange(self._sample_size) / sample_rate
        for analog_channel in analog_channels:
            if analog_channel in self._device_step.analog_functions:
                function = self._device_step.analog_functions[analog_channel]
                self.analog_data.append(function.output(times))
            else:
                self.analog_data.append(np.zeros(len(times), dtype=float))
        for digital_channel in digital_channels:
            if digital_channel in self._device_step.digital_functions:
                function = self._device_step.digital_functions[digital_channel]
                self.digital_data.append(function.output(times))
            else:
                self.digital_data.append(np.zeros(len(times), dtype=bool))

    def __eq__(self, other: "MockSegment") -> bool:
        """For comparing segments to check if they contain the same content."""
        if self._device_step != other._device_step:
            return False
        return True


class MockSegmenter(Segmenter):
    """
    A mock segmenter.

    It does not do anything other than checking for segment duplicates.
    """

    def __init__(
        self,
        sample_rate: int,
        analog_channels: list[int],
        digital_channels: list[int],
    ):
        super().__init__(analog_channels, digital_channels)
        self._sample_rate = sample_rate

    def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
        super().set_steps(steps, sequence_to_steps_map)
        self._device_steps_to_segments()
        self._get_sequence_to_segments_map()

    def _device_steps_to_segments(self):
        """
        Sets :attr:`_segments` and :attr:`_device_step_to_segment_map`.

        Gets rid of duplicates in segments.
        """
        self._segments: list[MockSegment] = []
        self._device_step_to_segment_map: dict[int, int] = {}
        for device_step_index, device_step in enumerate(self._device_steps):
            segment = MockSegment(
                device_step, self._sample_rate, self._analog_channels, self._digital_channels
            )
            # if the segment is already in self._segments, do not add it again.
            try:
                segment_index = self._segments.index(segment)
            except ValueError:
                self._segments.append(segment)
                segment_index = len(self._segments) - 1
            self._device_step_to_segment_map[device_step_index] = segment_index

    def _get_sequence_to_segments_map(self):
        """
        Gets a mapping from sequence indices to segment indices in execution order.
        """
        self._sequence_to_segments_map: dict[int, list[int]] = {}
        for sequence_index in self._sequence_to_device_steps_map:
            self._sequence_to_segments_map[sequence_index] = []
            for device_step_index in self._sequence_to_device_steps_map[sequence_index]:
                self._sequence_to_segments_map[sequence_index].append(
                    self._device_step_to_segment_map[device_step_index]
                )

    def get_awg_memory_data(
        self, sequence_indices: list[int]
    ) -> tuple[dict[str, Any], dict[int, int], list[int]]:
        # list of step indices that is used in the sequences requested
        step_indices: list[int] = []
        for sequence_index in sequence_indices:
            step_indices.extend(self._sequence_to_device_steps_map[sequence_index])
        # removes duplicates
        step_indices = list(dict.fromkeys(step_indices))

        # segments to be programmed
        segments: list[MockSegment] = []
        # mapping from step indices to indices in the above segments list.
        step_to_segment_map: dict[int, int] = {}
        for step_index in step_indices:
            segment = self._segments[self._device_step_to_segment_map[step_index]]
            # check for duplicates.
            try:
                segment_index = segments.index(segment)
            except ValueError:
                segments.append(segment)
                segment_index = len(segments) - 1
            step_to_segment_map[step_index] = segment_index

        # this is the minimum amount of data to program the AWG.
        # if the AWG needs more data, it can be added as long as the Device class is compatible.
        awg_data = {"segments": segments}
        return awg_data, step_to_segment_map, sequence_indices
