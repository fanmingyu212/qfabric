from typing import Any

import numpy as np
import numpy.typing as npt

from qfabric.planner.segmenter import Segment, Segmenter
from qfabric.sequence.function import AnalogEmpty, DigitalEmpty
from qfabric.sequence.step import DeviceStep, Step

SAMPLE_RATE = int(625e6)
SAMPLE_TIME = 1 / SAMPLE_RATE

DIGITAL_CHANNELS = 3
ANALOG_CHANNELS = 4
MAX_SEGMENT_BLOCK_COUNT = int(64e3)
RESERVED_SEGMENT_BLOCK_COUNT = 1  # for a shared empty segment.
MIN_SAMPLES_PER_SEGMENT_BLOCK = 96  # for 4 channels
MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK = 32  # for 4 channels
MAX_SEQUENCE_STEPS = 4096
MAX_LOOP_COUNT = int(1e6 - 1)
TOTAL_MEM_SIZE = int(2e9) // ANALOG_CHANNELS  # 2 Gsamples per all channels.

SEG_CHUNK_SIZE = 65536

VOLTAGE_RANGE = 2.5
VOLTAGE_STEPS = 2**16


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
    return actual_sample_size


def get_max_segment_size_and_count(segment_count: int) -> tuple[int, int]:
    """
    Gets the maximum segment size and maximum number of segments.

    Args:
        segment_count (int): Number of segments needed.

    Returns:
        tuple[int, int]:
            Maximum size of each segment, aximum number of segments
            without reducing the segment size.
    """
    segment_count += RESERVED_SEGMENT_BLOCK_COUNT
    segment_count_for_memory_use = np.power(2, int(np.ceil(np.log2(segment_count))))
    max_nominal_segment_size = TOTAL_MEM_SIZE // segment_count_for_memory_use
    max_segment_size = get_segment_sample_size(max_nominal_segment_size)
    return max_segment_size, segment_count_for_memory_use


def voltages_to_awg_data(analog_voltages: npt.NDArray[np.float64]) -> npt.NDArray[np.int16]:
    """
    Converts voltages in V to AWG data of a channel.

    Args:
        analog_voltages (npt.NDArray[np.float64]): AWG output in V.

    Returns:
        npt.NDArray[np.int16]: AWG raw data.
    """
    analog_int: npt.NDArray[np.int16] = np.round(
        (analog_voltages / VOLTAGE_RANGE) * VOLTAGE_STEPS
    ).astype(np.int16)
    return analog_int


def add_digital_to_awg_data(
    awg_data: npt.NDArray[np.int16], digital: npt.NDArray[np.bool]
) -> npt.NDArray[np.int16]:
    """
    Adds digital output information to AWG analog data.

    Digital data of a channel uses the most significant bit of the AWG data of a channel.

    Args:
        awg_data (npt.NDArray[np.int16]): AWG raw data.
        digital (npt.NDArray[np.bool]): Digital data.

    Returns:
        awg_data (npt.NDArray[np.int16]): AWG raw data with digital data in it.
    """
    return np.bitwise_or(
        np.right_shift(awg_data.view(np.uint16), 1),
        np.left_shift(digital.astype(np.uint16), 15),
    ).astype(np.int16)


class SegmentBlock:
    """
    A block of a segment that corresponds to a AWG memory block

    As the M4i6622 device has uniform sized memory blocks,
    it is often helpful to divide :class:`M4i6622Segment` into smaller
    units to ensure efficient use of the AWG memory.

    Args:
        segment (M4i6622Segment): Segment to divide into blocks.
        start_index (int): Start sample (time) index.
        stop_index (int): Stop sample (time) index.

    Attributes:
        awg_data (npt.NDArray[np.int16]): AWG data compatible with its memory format.
    """

    def __init__(self, segment: "M4i6622Segment", start_index: int, stop_index: int):
        self.awg_data = segment.awg_data[start_index:stop_index].flatten()

    def __eq__(self, other: "SegmentBlock") -> bool:
        return (self.awg_data.shape == other.awg_data.shape) and np.all(
            self.awg_data == other.awg_data
        )


class EmptySegmentBlock(SegmentBlock):
    """Empty segment block.

    Args:
        num_samples (int): Number of samples for each channel.
    """

    def __init__(self, num_samples: int):
        self.awg_data = np.zeros(num_samples * ANALOG_CHANNELS, dtype=np.int16)


class M4i6622Segment(Segment):
    """
    Performs segmentation of a device step data for the M4i6622 AWG.

    Args:
        device_step (DeviceStep): Device step containing AWG functions on a device.
        analog_channels: (list[int]): Analog channels of the device.
        digital_analog_map: (dict[int, int]):
            Maps digital channels to analog channels storing the digital data.

    Attributes:
        max_segment_block_size (int):
            Last maximum block size used to divide the segment into blocks.
        segment_size (int): Number of samples in this segment.
        is_empty (bool): If this segment has no nonzero analog or digital data.
        awg_data (npt.NDArray[np.int16]):
            AWG data in 2D array. Axis 0 is sample indices, axis 1 is channel indices.
        _constant_sample_region_indices (list[tuple[int, int]]):
            Start and stop indices for regions with constant AWG data.
            Here constant AWG data means that for each digital or analog channel,
            the output value does not change in this region.
            Such regions may be replaced by a shorter block repeated a few times.

        segment_blocks (list[SegmentBlock]): Distinct blocks for this segment.
        segment_block_indices_and_repeats (list[tuple[int, int]]):
            Block indices and repeats for this segment. Index -1 refers to a maximum length
            :class:`EmptySegmentBlock`.
    """

    def __init__(
        self,
        device_step: DeviceStep,
        analog_channels: list[int],
        digital_analog_map: dict[int, int],
    ):
        super().__init__(device_step)
        self._analog_channels = analog_channels
        self._digital_analog_map = digital_analog_map
        self.max_segment_block_size = None
        self.segment_size = get_segment_sample_size_from_time(device_step.duration)
        self.is_empty = False

        # check if there is no function defined in this segment.
        if len(device_step.analog_functions) + len(device_step.digital_functions) == 0:
            self.awg_data = None
            self.is_empty = True
        else:
            awg_data = self._get_awg_data(device_step, analog_channels, digital_analog_map)
            # check if awg data is all zero.
            if np.all(awg_data == 0):
                self.awg_data = None
                self.is_empty = True
            else:
                self.awg_data = awg_data
                # attempts to find regions with constant awg data.
                # TODO: these regions can be replaced by shorter repeating segment blocks
                # to reduce memory use.
                self._constant_sample_region_indices = self._find_constant_blocks(awg_data)

    def _get_awg_data(
        self,
        device_step: DeviceStep,
        analog_channels: list[int],
        digital_analog_map: dict[int, int],
    ) -> npt.NDArray[np.int16]:
        times = np.arange(self.segment_size) * SAMPLE_TIME
        analog_awg_data = {}
        for analog_channel in analog_channels:
            analog_func = device_step.analog_functions.get(analog_channel, AnalogEmpty())
            analog_awg_data_channel = voltages_to_awg_data(analog_func.output(times, time_offset=0))
            analog_awg_data[analog_channel] = analog_awg_data_channel
        for digital_channel in digital_analog_map:
            analog_channel = digital_analog_map[digital_channel]
            digital_func = device_step.digital_functions.get(digital_channel, DigitalEmpty())
            digital_awg_data = digital_func.output(times)

            # digital data uses one bit in a selected AWG channel.
            analog_awg_data[analog_channel] = add_digital_to_awg_data(
                analog_awg_data[analog_channel], digital_awg_data
            )
        awg_data = []
        for analog_channel in analog_channels:
            awg_data.append(analog_awg_data[analog_channel])
        awg_data = np.transpose(awg_data)
        return awg_data

    def _find_constant_blocks(self, awg_data: npt.NDArray[np.int16]) -> list[tuple[int, int]]:
        """
        Finds blocks of constant AWG data.

        Ignores constant AWG data blocks with less than `SEG_CHUNK_SIZE` points.

        Currently the blocks are underestimated. The left and right bounds may not reach the end
        of the constant region. Number of segment_blocks needed may be further reduced by obtaining
        the correct bounds.

        Args:
            awg_data (npt.NDArray[np.int16]):
                AWG data in 2D array. Axis 0 is data index, and axis 1 is AWG channel index.

        Returns:
            list[tuple[int, int]]: list of (start, stop) indices of constant blocks.
        """
        length = len(awg_data)
        if length < SEG_CHUNK_SIZE:
            return []

        # finds regions where the AWG data is the same for start sample of two consecutive blocks.
        previous_row: npt.NDArray[np.int16] = awg_data[0]
        possible_equal_block_start_indices: list[int] = []
        for check_index in range(
            SEG_CHUNK_SIZE, length - MIN_SAMPLES_PER_SEGMENT_BLOCK, SEG_CHUNK_SIZE
        ):
            if np.all(previous_row == awg_data[check_index]):
                possible_equal_block_start_indices.append(check_index - SEG_CHUNK_SIZE)
            previous_row = awg_data[check_index]

        # confirms if the previously filtered blocks have the same AWG data.
        confirmed_equal_block_block_indices: list[int] = []
        for check_index in possible_equal_block_start_indices:
            block = awg_data[check_index : check_index + SEG_CHUNK_SIZE]
            if np.all(block == block[0]):
                confirmed_equal_block_block_indices.append(check_index // SEG_CHUNK_SIZE)

        # finds the start and stop indices of consecutive AWG data regions.
        consecutive_block_regions = self._get_consecutive_regions(
            confirmed_equal_block_block_indices
        )
        constant_sample_region_indices: list[tuple[int, int]] = []
        for start, stop in consecutive_block_regions:
            if stop > start:
                constant_sample_region_indices.append(
                    (start * SEG_CHUNK_SIZE, (stop + 1) * SEG_CHUNK_SIZE)
                )
        return constant_sample_region_indices

    def _get_consecutive_regions(self, indices: list[int]) -> list[tuple[int, int]]:
        """
        Groups consecutive values in a list.

        Examples:
            >>> self._get_consecutive_regions([1, 2, 4, 5, 6, 8])
            [(1, 2), (4, 6), (8, 8)]
        """
        if len(indices) == 0:
            return []
        regions = []
        start = prev = indices[0]
        for x in indices[1:]:
            if x == prev + 1:
                prev = x
            else:
                regions.append((start, prev))
                start = prev = x
        regions.append((start, prev))
        return regions

    def __eq__(self, other: "M4i6622Segment") -> bool:
        if self._device_step != other._device_step:
            return False
        if self._analog_channels != other._analog_channels:
            return False
        if self._digital_analog_map != other._digital_analog_map:
            return False
        return True

    def estimate_segment_blocks_needed(self, max_segment_block_size: int) -> int:
        """
        Conservatively estimates the number of segment blocks needed for this segment.

        Args:
            max_segment_block_size (int): Maximum length of a segment block to be saved in the AWG.

        Returns:
            int: Maximum number of segment blocks needed.
        """
        # case 1: segment is empty.
        if self.is_empty:
            # the device always keep a segment block of all zeros to be shared among all segments.
            if self.segment_size % max_segment_block_size >= MIN_SAMPLES_PER_SEGMENT_BLOCK:
                # if the remainder is longer than the minimum samples allowed
                # the remainder can be fit in a single sequence.
                return 1
            else:
                # if the remainder is shorter than the minimum samples allowed
                # the last two segment blocks both need to be saved.
                return 2

        constant_region_lengths = [
            stop - start for start, stop in self._constant_sample_region_indices
        ]
        min_valid_constant_region_length = max_segment_block_size * 3
        # case 2: no long constant regions.
        # the segment is simply segmented in equal steps.
        if (
            len(constant_region_lengths) == 0
            or np.max(constant_region_lengths) < min_valid_constant_region_length
        ):
            return (self.segment_size + max_segment_block_size - 1) // max_segment_block_size

        # case 3: long constant regions available.
        segment_block_divide_indices = list(range(0, self.segment_size, max_segment_block_size))
        constant_region_repeat_counters: dict[int, int] = {}
        for const_index in range(len(self._constant_sample_region_indices)):
            constant_region_repeat_counters[const_index] = 0

        for kk in range(len(segment_block_divide_indices) - 1):
            start = segment_block_divide_indices[kk]
            stop = segment_block_divide_indices[kk + 1]
            const_index = self._check_segment_block_in_constant_region(start, stop)
            if const_index is not None:
                constant_region_repeat_counters[const_index] += 1

        num_of_segment_blocks = (
            self.segment_size + max_segment_block_size - 1
        ) // max_segment_block_size

        for const_index in range(len(self._constant_sample_region_indices)):
            if constant_region_repeat_counters[const_index] > 1:
                num_of_segment_blocks -= constant_region_repeat_counters[const_index] - 1

        # if the last segment is too short, the previous one must be reduced in size.
        # if the previous segment is part of a constant region, it cannot be reused.
        if self.segment_size % max_segment_block_size < MIN_SAMPLES_PER_SEGMENT_BLOCK:
            const_index_last_long_segment_block = self._check_segment_block_in_constant_region(
                segment_block_divide_indices[-2], segment_block_divide_indices[-1]
            )
            if (
                const_index_last_long_segment_block is not None
                and constant_region_repeat_counters[const_index_last_long_segment_block] > 1
            ):
                num_of_segment_blocks += 1
        return num_of_segment_blocks

    def _check_segment_block_in_constant_region(self, start_index: int, stop_index: int) -> int:
        """
        Check if the segment block start and stop indices are in a constant region.
        """
        for const_index, (const_start, const_stop) in enumerate(
            self._constant_sample_region_indices
        ):
            if start_index >= const_start and stop_index <= const_stop:
                return const_index
        return None

    def create_segment_blocks(self, max_segment_block_size: int):
        """
        Creates the segment_blocks.

        Defines :attr:`segment_blocks` as distinct segment_blocks of this segment.

        Defines :attr:`segment_block_indices_and_repeats` as segment_block indices and repeats
        to execute this segment. Index -1 refers to an empty segment block of maximum length.

        Args:
            max_segment_block_size (int): Maximum length of a segment_block to be saved in the AWG.
        """
        if max_segment_block_size == self.max_segment_block_size:
            return
        self.max_segment_block_size = max_segment_block_size
        self.segment_blocks: list[SegmentBlock] = []
        self.segment_block_indices_and_repeats: list[tuple[int, int]] = []

        if self.is_empty:
            # empty segment
            repeats = self.segment_size // max_segment_block_size
            remainder = self.segment_size % max_segment_block_size
            if remainder < MIN_SAMPLES_PER_SEGMENT_BLOCK:
                # reminder is shorter than the minimum length of a segment block.
                repeats -= 1
                remainder += max_segment_block_size
            if repeats > 0:
                segment_block = EmptySegmentBlock(max_segment_block_size)
                segment_block_index = self._check_and_add_segment_block(segment_block)
                self.segment_block_indices_and_repeats.append((segment_block_index, repeats))

            start_index = max_segment_block_size * repeats
            if remainder > max_segment_block_size:
                # if remainder is slightly larger than the maximum segment block size,
                # adds a segment block slightly shorter than the maximum size.
                stop_index = self.segment_size - MIN_SAMPLES_PER_SEGMENT_BLOCK
                segment_block = EmptySegmentBlock(stop_index - start_index)
                segment_block_index = self._check_and_add_segment_block(segment_block)
                self.segment_block_indices_and_repeats.append((segment_block_index, 1))
                remainder = MIN_SAMPLES_PER_SEGMENT_BLOCK

            # puts the remainder in another segment block.
            segment_block = EmptySegmentBlock(remainder)
            segment_block_index = self._check_and_add_segment_block(segment_block)
            self.segment_block_indices_and_repeats.append((segment_block_index, 1))
        else:
            segment_block_divide_indices = list(range(0, self.segment_size, max_segment_block_size))
            if segment_block_divide_indices[-1] != self.segment_size:
                segment_block_divide_indices.append(self.segment_size)
            if self.segment_size - segment_block_divide_indices[-2] < MIN_SAMPLES_PER_SEGMENT_BLOCK:
                segment_block_divide_indices[-2] = self.segment_size - MIN_SAMPLES_PER_SEGMENT_BLOCK
            constant_region_segment_block_index: dict[int, int] = {}
            for const_index in range(len(self._constant_sample_region_indices)):
                constant_region_segment_block_index[const_index] = -1

            for kk in range(len(segment_block_divide_indices) - 1):
                start = segment_block_divide_indices[kk]
                stop = segment_block_divide_indices[kk + 1]
                const_index = self._check_segment_block_in_constant_region(start, stop)
                if (
                    const_index is None
                    or constant_region_segment_block_index[const_index] == -1
                    or stop - start != max_segment_block_size
                ):
                    # if the block is not in a constant region
                    # or the constant region has no block defined
                    # or the block length is shorter than the maximum block size.
                    segment_block = SegmentBlock(self, start, stop)
                    segment_block_index = self._check_and_add_segment_block(segment_block)
                    if const_index is not None:
                        constant_region_segment_block_index[const_index] = segment_block_index
                else:
                    # if a segment in the current constant region of equal length is defined.
                    segment_block_index = constant_region_segment_block_index[const_index]
                self.segment_block_indices_and_repeats.append((segment_block_index, 1))

            # puts the remainder in another segment_block.
            if stop != self.segment_size:
                segment_block = SegmentBlock(self, stop, self.segment_size)
                segment_block_index = self._check_and_add_segment_block(segment_block)
                self.segment_block_indices_and_repeats.append((segment_block_index, 1))

        # combine consecutive repeats of the same segment_block.
        condensed_indices_and_repeats: list[tuple[int, int]] = []
        last_segment_block_index = None
        last_segment_block_repeat = 0
        for segment_block_index, repeat in self.segment_block_indices_and_repeats:
            if last_segment_block_index is None:
                last_segment_block_index = segment_block_index
            elif segment_block_index != last_segment_block_index:
                condensed_indices_and_repeats.append(
                    (last_segment_block_index, last_segment_block_repeat)
                )
                last_segment_block_index = segment_block_index
                last_segment_block_repeat = 0
            last_segment_block_repeat += repeat
        condensed_indices_and_repeats.append((last_segment_block_index, last_segment_block_repeat))
        self.segment_block_indices_and_repeats = condensed_indices_and_repeats

    def _check_and_add_segment_block(self, segment_block: SegmentBlock) -> int:
        """
        Adds a segment_block to :attr:`segment_blocks`.

        If the segment_block is already in :attr:`segment_blocks`, skips adding it.

        Args:
            segment_block (SegmentBlock): Segment_block to add.

        Returns:
            int: Index of the segment_block in :attr:`segment_blocks`.
        """
        for segment_block_index, segment_block_saved in enumerate(self.segment_blocks):
            if segment_block == segment_block_saved:
                return segment_block_index
        self.segment_blocks.append(segment_block)
        return len(self.segment_blocks) - 1


class M4i6622Segmenter(Segmenter):
    """
    Converts steps to :class:`M4i6622Segment` objects.

    Supports the Spectrum M4i6622 AWG.

    Assumptions:
        * The AWG sample rate is fixed at 625 Msps.
        * All four AWG channels of the device is used.

    Args:
        device_step (DeviceStep): Device step containing AWG functions on a device.
        analog_channels: (list[int]): Analog channels of the device.
        digital_analog_map: (dict[int, int]):
            Maps digital channels to analog channels storing the digital data.

    Attributes:
        _device_steps (list[DeviceStep]):
            See :meth:`set_steps`, device steps scheduled on this device.
            All unique device_steps are saved in it.
        _sequence_to_device_steps_map (dict[int, list[int]]):
            See :meth:`set_steps`, mapping from sequence indices to device step indices.
        _segments (list[M4i6622Segment]): Unique segments from :attr:`_device_steps`.
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
        analog_channels_to_store_digital_data: list[int],
    ):
        if len(analog_channels) != ANALOG_CHANNELS:
            raise ValueError(f"The number of analog channels must be {ANALOG_CHANNELS}.")
        if len(digital_channels) != DIGITAL_CHANNELS:
            raise ValueError(f"The number of digital channels must be {DIGITAL_CHANNELS}.")
        super().__init__(analog_channels, digital_channels)
        digital_analog_map = dict(zip(digital_channels, analog_channels_to_store_digital_data))
        if set(digital_channels) != set(list(digital_analog_map)):
            raise ValueError("All digital channels must be defined in the digital to analog map.")
        if not set(list(digital_analog_map.values())).issubset(set(analog_channels)):
            raise ValueError("The digital to analog map must use the analog channels defined.")
        self._digital_analog_map = digital_analog_map
        self._segments: list[M4i6622Segment] = []

    def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
        super().set_steps(steps, sequence_to_steps_map)
        self._device_steps_to_segments()
        self._get_sequence_to_segments_map()

    def _device_steps_to_segments(self):
        """
        Converts device steps to segments.

        If a segment has been defined before in :attr:`_segments`, reuse it.
        """
        new_segments: list[M4i6622Segment] = []
        self._device_step_to_segment_map: dict[int, int] = {}
        for device_step_index, device_step in enumerate(self._device_steps):
            segment = M4i6622Segment(device_step, self._analog_channels, self._digital_analog_map)
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

    def _check_awg_memory_limit(self, sequence_indices: list[int]) -> list[int]:
        """
        Recursively checks which sequences can be programmed.

        Also determines the maximum segment block size.

        Args:
            sequence_indices (list[int]): Desired sequence indices to program.

        Returns:
            list[int]: Sequence indices that can be fitted in the AWG memory.
        """
        # gets the segment indices given the sequence indices
        segment_indices: list[int] = []
        for sequence_index in sequence_indices:
            segment_indices.extend(self._sequence_to_segments_map[sequence_index])
        segment_indices = list(dict.fromkeys(segment_indices))

        # gets sample sizes of the segments.
        segment_sizes: list[int] = []
        for segment_index in segment_indices:
            segment = self._segments[segment_index]
            if segment.is_empty:
                segment_sizes.append(0)
            else:
                segment_sizes.append(segment.segment_size)
        segment_block_count = len(segment_sizes)
        max_segment_block_size, max_segment_block_count = get_max_segment_size_and_count(
            segment_block_count
        )

        while max_segment_block_count < MAX_SEGMENT_BLOCK_COUNT:
            memory_check_succeeded = True
            total_segment_block_count = RESERVED_SEGMENT_BLOCK_COUNT
            for segment_index in range(len(segment_sizes)):
                segment = self._segments[segment_index]
                total_segment_block_count += segment.estimate_segment_blocks_needed(
                    max_segment_block_size
                )
                if total_segment_block_count > max_segment_block_count:
                    # more blocks needed than memory size.
                    memory_check_succeeded = False
                    break
            if not memory_check_succeeded:
                # reduce the block size by half and try again.
                max_segment_block_size, max_segment_block_count = get_max_segment_size_and_count(
                    max_segment_block_count * 2
                )
            else:
                self.valid_max_segment_block_size = max_segment_block_size
                self.valid_max_segment_block_count = max_segment_block_count
                return sequence_indices

        # if not succeeded, check if less sequences can be programmed.
        if len(sequence_indices) > 1:
            return self._check_awg_memory_limit(sequence_indices[:-1])
        else:
            raise RuntimeError("Sequence is too long to be programmed.")

    def _get_divided_segments(
        self, sequence_indices: list[int]
    ) -> tuple[list[M4i6622Segment], dict[int, int]]:
        """
        Gets segments needed for the desired sequences and divide them into blocks.

        Args:
            sequence_indices (list[int]): Sequence indices to be programmed.

        Returns:
            tuple[list[M4i6622Segment], dict[int, int]]:
                list of segments to be programmed, mapping from step indices to segment indices.
        """
        segment_indices: list[int] = []
        device_step_indices: list[int] = []
        for sequence_index in sequence_indices:
            segment_indices.extend(self._sequence_to_segments_map[sequence_index])
            device_step_indices.extend(self._sequence_to_device_steps_map[sequence_index])
        segment_indices = list(dict.fromkeys(segment_indices))
        device_step_indices = list(dict.fromkeys(device_step_indices))

        segments: list[M4i6622Segment] = []
        step_to_segment_map: dict[int, int] = {}
        for segment_index in segment_indices:
            segment = self._segments[segment_index]
            segment.create_segment_blocks(self.valid_max_segment_block_size)
            segments.append(segment)
        for device_step_index in device_step_indices:
            # step and device_step indices are the same.
            step_to_segment_map[device_step_index] = segments.index(
                self._segments[self._device_step_to_segment_map[device_step_index]]
            )
        return segments, step_to_segment_map

    def get_awg_memory_data(
        self, sequence_indices: list[int]
    ) -> tuple[dict[str, Any], dict[int, int], list[int]]:
        sequence_indices_to_program = self._check_awg_memory_limit(sequence_indices)
        segments, step_to_segment_map = self._get_divided_segments(sequence_indices_to_program)
        awg_data = {
            "segments": segments,
            "max_segment_block_size": self.valid_max_segment_block_size,
            "max_num_of_blocks": self.valid_max_segment_block_count,
        }
        return awg_data, step_to_segment_map, sequence_indices_to_program
