from typing import Any

from qfabric.planner.segmenter.m4i6622 import (
    ANALOG_CHANNELS,
    MAX_LOOP_COUNT,
    M4i6622Segment,
    M4i6622Segmenter,
    SegmentBlock,
)
from qfabric.programmer.device import Device
from qfabric.programmer.driver.m4i6622 import M4i6622Driver


class M4i6622Device(Device):
    """
    Programming interface of the Spectrum M4i6622 AWG.

    Args:
        segmenter (M4i6622Segmenter): Segmenter for this AWG device.
        resource (str): Resource name of the device.
        principal_device (bool): Whether the device is a principal device (controlling other AWGs).
        **kwargs:
            See :class:`~qfabric.programmer.driver.m4i6622.M4i6622Driver` for
            optional keyword arguments.
    """

    def __init__(
        self, segmenter: M4i6622Segmenter, resource: str, principal_device: bool, **kwargs
    ):
        super().__init__(segmenter, resource, principal_device)

        # gets the ttl mapping to awg channels from the segmenter definition.
        ttl_to_awg_map: dict[int, int] = {}
        for ttl_index, digital_channel_index in enumerate(segmenter._digital_channels):
            analog_channel_index = segmenter._digital_analog_map[digital_channel_index]
            awg_index = segmenter._analog_channels.index(analog_channel_index)
            ttl_to_awg_map[ttl_index] = awg_index
        self._driver = M4i6622Driver(ttl_to_awg_map=ttl_to_awg_map, **kwargs)

        self._max_num_of_blocks: int = None
        self._max_sample_size_per_block: int = None
        self._segment_to_blocks_map: dict[int, list[int]] = {}
        self._programmed_segments: list[M4i6622Segment] = []

    def program_memory(self, instructions: dict[str, Any]):
        segments: list[M4i6622Segment] = instructions["segments"]
        if self._max_sample_size_per_block != instructions["max_segment_block_size"]:
            # if the maximum block size is changed, all programmed segments are useless.
            self._max_sample_size_per_block = instructions["max_segment_block_size"]
            self._max_num_of_blocks = instructions["max_num_of_segment_blocks"]
            self._segment_to_blocks_map = {}
            self._programmed_segments = []
            self._set_maximum_number_of_blocks(self._max_num_of_blocks)
        else:
            self._cleanup_programmed_segments(segment)

        # find the block indices available to use.
        available_block_indices = set(range(self._max_num_of_blocks))
        for used_block_indices in self._segment_to_blocks_map.values():
            available_block_indices -= set(used_block_indices)
        available_block_indices = sorted(list(available_block_indices))

        new_segment_to_blocks_map: dict[int, list[int]] = {}
        for segment_index, segment in enumerate(segments):
            # try to check if the segment is already programmed.
            try:
                segment_saved_index = self._programmed_segments.index(segment)
                new_segment_to_blocks_map[segment_index] = self._segment_to_blocks_map[
                    segment_saved_index
                ]
            except ValueError:
                new_segment_to_blocks_map[segment_index] = []
                for segment_block in segment.segment_blocks:
                    block_index = available_block_indices[0]
                    self._write_segment_block(block_index, segment_block)
                    new_segment_to_blocks_map[segment_index].append(block_index)
                    available_block_indices = available_block_indices[1:]
        self._segment_to_blocks_map = new_segment_to_blocks_map

    def _set_maximum_number_of_blocks(self, num_of_blocks: int):
        self._driver._set_sequence_max_segments(num_of_blocks)

    def _write_segment_block(self, block_index: int, segment_block: SegmentBlock):
        self._driver._set_sequence_write_segment(block_index)
        size = len(segment_block.awg_data) // ANALOG_CHANNELS
        self._driver._set_sequence_write_segment_size(size)
        data_length = self._driver._define_transfer_buffer(segment_block.awg_data)
        self._driver._set_data_ready_to_transfer(data_length)
        self._driver._start_dma_transfer()
        self._driver._wait_dma_transfer()

    def _cleanup_programmed_segments(self, new_segments: list[M4i6622Segment]):
        """
        Remove all unused segments that are already programmed.
        """
        programmed_segments: list[M4i6622Segment] = []
        for segment_index, segment in enumerate(self._programmed_segments):
            if segment in new_segments:
                programmed_segments.append(segment)
            else:
                del self._segment_to_blocks_map[segment_index]
        self._programmed_segments = programmed_segments

    def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
        sequence_steps: list[tuple[int, int]] = []
        for segment_index, segment_repeats in segment_indices_and_repeats:
            segment = self._programmed_segments[segment_index]

            if len(segment.segment_block_indices_and_repeats) == 1:
                segment_block_index, segment_block_repeats = (
                    segment.segment_block_indices_and_repeats[0]
                )
                total_loops = segment_repeats * segment_block_repeats
                block_index = self._segment_to_blocks_map[segment_index][segment_block_index]
                sequence_steps.extend(
                    self._get_sequence_steps_for_N_repeats(block_index, total_loops)
                )
            else:
                for kk in range(segment_repeats):
                    for (
                        segment_block_index,
                        segment_block_repeats,
                    ) in segment.segment_block_indices_and_repeats:
                        block_index = self._segment_to_blocks_map[segment_index][
                            segment_block_index
                        ]
                        sequence_steps.extend(
                            self._get_sequence_steps_for_N_repeats(
                                block_index, segment_block_repeats
                            )
                        )
        self._write_sequence_steps(sequence_steps)

    def _get_sequence_steps_for_N_repeats(
        self, block_index: int, total_loops: int
    ) -> list[tuple[int, int]]:
        steps: list[tuple[int, int]] = [(block_index, MAX_LOOP_COUNT)] * (
            total_loops // MAX_LOOP_COUNT
        )
        remainder = total_loops % MAX_LOOP_COUNT
        if remainder != 0:
            steps.append((block_index, remainder))
        return steps

    def _write_sequence_steps(self, sequence_steps: list[tuple[int, int]]):
        step_number = 0
        total_steps = len(sequence_steps)
        for block_number, block_repeats in sequence_steps:
            if step_number == total_steps - 1:
                end = "end_sequence"
            else:
                edn = "end_loop"
            self._driver._set_segment_step_memory(
                step_number, block_number, step_number + 1, block_repeats, end=end
            )
            step_number += 1

    def start(self):
        self._driver._set_segment_start_step(0)
        self._driver._start()
        self._driver._enable_triggers()

    def wait_until_complete(self):
        self._driver._wait_for_complete()

    def stop(self):
        self._driver._disable_triggers()
        self._driver._stop()

    def setup_external_trigger(self):
        self._driver._setup_external_trigger()

    def setup_software_trigger(self):
        self._driver._setup_software_trigger()
