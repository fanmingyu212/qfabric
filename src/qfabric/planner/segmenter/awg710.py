from typing import Any

import numpy as np

from qfabric.planner.segmenter import Segment, Segmenter
from qfabric.sequence.function import AnalogEmpty, DigitalEmpty
from qfabric.sequence.step import DeviceStep, Step

SAMPLE_RATE = int(4e9)
SAMPLE_TIME = 1 / SAMPLE_RATE

DIGITAL_CHANNELS = 2
ANALOG_CHANNELS = 1
MIN_SAMPLES_PER_SEGMENT_BLOCK = 960
MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK = 4
MAX_SAMPLES_PER_SEGMENT_BLOCK = 16200000
MAX_SEQUENCE_STEPS = 8000
MAX_LOOP_COUNT = 65536


def get_segment_sample_size_from_time(nominal_segment_time: float) -> int:
    """
    Gets the minimum allowed sample size given the segment duration.

    Args:
        nominal_segment_time (float): Desired duration of a segment.

    Returns:
        int:
            Minimum number of samples of a segment
            longer or equal to the desired duration.
    """
    nominal_sample_size = int(nominal_segment_time * SAMPLE_RATE)
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
    samples_to_append = MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK - (
        nominal_sample_size % MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK
    )
    if samples_to_append == MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK:
        samples_to_append = 0
    actual_sample_size = nominal_sample_size + samples_to_append
    if actual_sample_size < MIN_SAMPLES_PER_SEGMENT_BLOCK:
        actual_sample_size = MIN_SAMPLES_PER_SEGMENT_BLOCK
    if actual_sample_size > MAX_SAMPLES_PER_SEGMENT_BLOCK:
        raise ValueError(f"Sample size cannot exceed {MAX_SAMPLES_PER_SEGMENT_BLOCK}.")
    return actual_sample_size


class AWG710Segment(Segment):
    """
    Represents a device step for the Tektronix AWG710 AWG.

    Args:
        device_step (DeviceStep): Device step containing AWG functions on a device.
        analog_channel (int): Analog channel of the device.
        digital_channels (list[int]): Digital channels of the device.
    """

    def __init__(
        self,
        device_step: DeviceStep,
        analog_channel: int,
        digital_channels: list[int],
    ):
        super().__init__(device_step)
        self._analog_channel = analog_channel
        self._digital_channels = digital_channels
        self.sample_rate = SAMPLE_RATE
        self.segment_size = get_segment_sample_size_from_time(device_step.duration)

        self._get_awg_data(device_step, analog_channel, digital_channels)

    def _get_awg_data(
        self, device_step: DeviceStep, analog_channel: int, digital_channels: list[int]
    ):
        times = np.arange(self.segment_size) * SAMPLE_TIME
        analog_func = device_step.analog_functions.get(analog_channel, AnalogEmpty())
        self.analog_data = analog_func.output(times)

        digital_func = device_step.digital_functions.get(digital_channels[0], DigitalEmpty())
        self.digital_1 = digital_func.output(times)
        digital_func = device_step.digital_functions.get(digital_channels[1], DigitalEmpty())
        self.digital_2 = digital_func.output(times)

    def __eq__(self, other: "AWG710Segment") -> bool:
        if self._device_step != other._device_step:
            return False
        if self.sample_rate != other.sample_rate:
            return False
        if self._analog_channel != other._analog_channel:
            return False
        if self._digital_channels != other._digital_channels:
            return False
        return True


class AWG710Segmenter(Segmenter):
    """
    Converts steps to :class:`AWG710Segment` objects.

    Supports the Tektronix AWG710 AWG.

    Args:
        device_step (DeviceStep): Device step containing AWG functions on a device.
        analog_channels: (list[int]): Analog channels of the device.
        digital_chnanels: (list[int]): Digital channels of the device.

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

    def __init__(self, analog_channels: list[int], digital_channels: list[int]):
        if len(analog_channels) != ANALOG_CHANNELS:
            raise ValueError(f"The number of analog channels must be {ANALOG_CHANNELS}.")
        if len(digital_channels) != DIGITAL_CHANNELS:
            raise ValueError(f"The number of digital channels must be {DIGITAL_CHANNELS}.")
        super().__init__(analog_channels, digital_channels)
        self._analog_channel = analog_channels[0]
        self._segments: list[AWG710Segment] = []

    def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
        super().set_steps(steps, sequence_to_steps_map)
        self._device_steps_to_segments()
        self._get_sequence_to_segments_map()

    def _device_steps_to_segments(self):
        """
        Converts device steps to segments.

        If a segment has been defined before in :attr:`_segments`, reuse it.
        """
        new_segments: list[AWG710Segment] = []
        self._device_step_to_segment_map: dict[int, int] = {}
        for device_step_index, device_step in enumerate(self._device_steps):
            segment = AWG710Segment(device_step, self._analog_channel, self._digital_channels)
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
        segments: list[AWG710Segment] = []
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
