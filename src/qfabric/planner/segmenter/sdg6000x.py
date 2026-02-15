from typing import Any

import numpy as np

from qfabric.planner.segmenter import Segment, Segmenter
from qfabric.sequence.function import AnalogEmpty
from qfabric.sequence.step import DeviceStep, Step

DEFAULT_SAMPLE_RATE = int(300e6)

DIGITAL_CHANNELS = 0
ANALOG_CHANNELS = 2
MIN_SAMPLES_PER_SEGMENT_BLOCK = 32
MAX_SAMPLES_ALL_SEGMENT_BLOCKS = 20000000
MAX_SEQUENCE_STEPS = 1024
MAX_LOOP_COUNT = 102400


def get_segment_sample_size_from_time(nominal_segment_time: float, sample_rate: int) -> int:
    """
    Gets the minimum allowed sample size given the segment duration.

    Args:
        nominal_segment_time (float): Desired duration of a segment.
        sample_rate (int): Sample rate.

    Returns:
        int:
            Minimum number of samples of a segment
            longer or equal to the desired duration.
    """
    nominal_sample_size = int(nominal_segment_time * sample_rate)
    return get_segment_sample_size(nominal_sample_size)


def get_segment_sample_size(nominal_sample_size: int) -> int:
    """
    Gets the minimum allowed sample size given desired sample size.

    Args:
        nominal_sample_size (int): Desired number of samples.

    Returns:
        int:
            Minimum number of samples of a segment
            longer or equal to the desired size.
    """
    if nominal_sample_size < MIN_SAMPLES_PER_SEGMENT_BLOCK:
        nominal_sample_size = MIN_SAMPLES_PER_SEGMENT_BLOCK
    if nominal_sample_size > MAX_SAMPLES_ALL_SEGMENT_BLOCKS:
        raise ValueError(f"Sample size cannot exceed {MAX_SAMPLES_ALL_SEGMENT_BLOCKS}.")
    return nominal_sample_size


class SDG6000XSegment(Segment):
    """
    Represents a device step for the Siglent SDG6000X AWG.

    Args:
        device_step (DeviceStep): Device step containing AWG functions on a device.
        analog_channels (list[int]): Analog channels of the device.
        sample_rate (int): Sample rate.
    """

    def __init__(
        self,
        device_step: DeviceStep,
        analog_channels: list[int],
        sample_rate: int,
    ):
        super().__init__(device_step)
        self._analog_channels = analog_channels
        self.sample_rate = sample_rate
        self.segment_size = get_segment_sample_size_from_time(device_step.duration, sample_rate)

        self._get_awg_data(device_step, analog_channels)

    def _get_awg_data(self, device_step: DeviceStep, analog_channels: list[int]):
        times = np.arange(self.segment_size) / self.sample_rate
        analog_func_1 = device_step.analog_functions.get(analog_channels[0], AnalogEmpty())
        self.analog_data_1 = analog_func_1.output(times)
        analog_func_2 = device_step.analog_functions.get(analog_channels[1], AnalogEmpty())
        self.analog_data_2 = analog_func_2.output(times)

    def __eq__(self, other: "SDG6000XSegment") -> bool:
        if self._device_step != other._device_step:
            return False
        if self.sample_rate != other.sample_rate:
            return False
        if len(self._analog_channels) != len(other._analog_channels):
            return False
        for kk in range(len(self._analog_channels)):
            if self._analog_channels[kk] != other._analog_channels[kk]:
                return False
        return True


class SDG6000XSegmenter(Segmenter):
    """
    Converts steps to :class:`SDG6000XSegment` objects.

    Supports Siglent SDG6000X series AWG.

    Args:
        analog_channels (list[int]): Analog channels of the device.
        digital_channels (list[int]): Digital channels of the device.
        sample_rate (int): Sample rate of the AWG. Minimum 1 uS/s. Maximum 300 MS/s.

    Attributes:
        _device_steps (list[DeviceStep]):
            See :meth:`set_steps`, device steps scheduled on this device.
            All unique device_steps are saved in it.
        _sequence_to_device_steps_map (dict[int, list[int]]):
            See :meth:`set_steps`, mapping from sequence indices to device step indices.
        _segments (list[AWG710Segment]): Unique segments from :attr:`_device_steps`.
        _device_step_to_segment_map (dict[int, int]):
            Mapping from device step indices to segment indices.
        _sequence_to_segments_map (dict[int, list[int]]):
            Mapping from sequence indices to segment indices. The keys are the segment indices
            in the scheduled order. The values are segment indices in the execution order
            of a sequence.
    """

    def __init__(
        self,
        analog_channels: list[int],
        digital_channels: list[int],
        sample_rate: int = DEFAULT_SAMPLE_RATE,
    ):
        if len(analog_channels) != ANALOG_CHANNELS:
            raise ValueError(f"The number of analog channels must be {ANALOG_CHANNELS}.")
        if len(digital_channels) != DIGITAL_CHANNELS:
            raise ValueError(f"The number of digital channels must be {DIGITAL_CHANNELS}.")
        super().__init__(analog_channels, digital_channels)
        self._sample_rate = sample_rate
        self._analog_channels = analog_channels
        self._segments: list[SDG6000XSegment] = []

    def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
        super().set_steps(steps, sequence_to_steps_map)
        self._device_steps_to_segments()
        self._get_sequence_to_segments_map()

    def _device_steps_to_segments(self):
        """
        Converts device steps to segments.

        If a segment has been defined before in :attr:`_segments`, reuse it.
        """
        new_segments: list[SDG6000XSegment] = []
        self._device_step_to_segment_map: dict[int, int] = {}
        for device_step_index, device_step in enumerate(self._device_steps):
            segment = SDG6000XSegment(device_step, self._analog_channels, self._sample_rate)
            try:
                segment_new_saved_index = new_segments.index(segment)
                self._device_step_to_segment_map[device_step_index] = segment_new_saved_index
            except ValueError:
                try:
                    segment_saved_index = self._segments.index(segment)
                    segment = self._segments[segment_saved_index]
                except ValueError:
                    pass
                new_segments.append(segment)
                self._device_step_to_segment_map[device_step_index] = len(new_segments) - 1
        self._segments = new_segments

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
        segments: list[SDG6000XSegment] = []
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
