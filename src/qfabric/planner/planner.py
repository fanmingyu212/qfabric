from collections import deque
from typing import Any, Callable

from qfabric.planner.segmenter import Segmenter
from qfabric.sequence.sequence import Sequence
from qfabric.sequence.step import Step


class Planner:
    """
    Sequence planner and AWG data generator.

    Args:
        segmenters (list[Segmenter]):
            List of AWG segmenters. Each device corresponds to a segmenter.

    Attributes:
        _sequences (dict[int, Sequence]):
            :class:`Sequence` objects that are currently scheduled.
        _scheduled_sequences (deque[int]): Sequence indices in the scheduled order.
        _prepared_sequences (dict[int, list[int]]):
            Sequence indices where their segments are programmed on each of the AWG device.
            The dict keys are AWG device indices.

        _steps (list[Step]): list of nonduplicate steps in all scheduled sequences.
        _sequence_to_steps_map (dict[int, list[int]]):
            Mapping from sequence indices to step indices in :attr:`_steps`.
        _step_to_segment_maps (dict[int, dict[int, int]]):
            Mapping from steps to segments for each AWG device.
            The first dict layer represents different AWG devices, and the second dict
            layer is mapping from step indices in :attr:`_steps` to segment indices
            sent to the AWG programmer device.
    """

    def __init__(self, segmenters: list[Segmenter]):
        self._segmenters = segmenters

        self._sequence_counter = 0
        self._sequences: dict[int, Sequence] = {}
        self._scheduled_sequences: deque[int] = deque([])
        self._prepared_sequences: dict[int, list[int]] = {}
        for device_index in range(len(self._segmenters)):
            self._prepared_sequences[device_index] = []

        self._steps: list[Step] = []
        self._sequence_to_steps_map: dict[int, list[int]] = {}
        self._step_to_segment_maps: dict[int, dict[int, int]] = {}

        self._needs_setup = False

    def register_program_single_device(
        self,
        function_program_memory_single_device: Callable[[int, Any], None],
        function_program_segment_step_single_device: Callable[[int, list[tuple[int, int]]], None],
    ):
        """
        Registers callback functions for programming the AWG memory and segment steps.

        Args:
            function_program_memory_single_device (Callable[[int, Any], None]):
                Function (*device_index*, *awg_data*) -> None.
            function_program_segment_step_single_device (Callable[[int, list[tuple[int, int]]], None]):
                Function (*device_index*, *list_of_segment_indices_and_repeats*) -> None.
        """
        self._program_memory_single_device = function_program_memory_single_device
        self._program_segment_step_single_device = function_program_segment_step_single_device

    def schedule(self, sequences: Sequence | list[Sequence], repeats: int = 1) -> list[int]:
        """
        Schedules sequences with repeats.

        Args:
            sequences (Sequence | list[Sequence]): A single sequence or a list of sequences.
            repeats (int): Number of repeats for *sequences*. Default 1.

        Returns:
            list[int]: List of sequence indices scheduled.
        """
        if isinstance(sequences, Sequence):
            sequences = [sequences]
        scheduled_sequences: deque[int] = deque([])
        for sequence in sequences:
            # if a sequence is already saved in `_sequences`, do not duplicate it.
            try:
                sequence_index = list(self._sequences.values()).index(sequence)
            except ValueError:
                self._sequences[self._sequence_counter] = sequence
                sequence_index = self._sequence_counter
                self._sequence_counter += 1
                # if a new sequence is added, `setup` must be called before
                # the new sequence can be executed.
                self._needs_setup = True
            scheduled_sequences.append(sequence_index)
        newly_scheduled = scheduled_sequences * repeats
        self._scheduled_sequences += newly_scheduled
        return newly_scheduled

    @property
    def scheduled_sequence_indices(self) -> deque[int]:
        """
        Gets indices of all scheduled sequences.
        """
        return self._scheduled_sequences

    def setup(self, program_single_sequence_only: bool = False):
        """
        Prepares execution of the sequences.

        This must be called before calling :attr:`get_next_sequence_info`.
        It sends *steps* in all scheduled sequences to segmenters to produce AWG segments.

        Args:
            program_single_sequence_only (bool):
                Whether to only program the next sequence. If True, attempts to program
                as many sequences as possible to fit in the AWG memory.
        """
        self._single_sequence_only = program_single_sequence_only
        self._get_steps_and_device_steps()
        self._needs_setup = False

    def _get_steps_and_device_steps(self):
        """
        Gets non-duplicate steps and calls `set_steps` for each segmenter.
        """
        self._steps, self._sequence_to_steps_map = self._deduplicate_steps()
        for segmenter in self._segmenters:
            segmenter.set_steps(self._steps, self._sequence_to_steps_map)

    def _deduplicate_steps(self) -> tuple[list[Step], dict[int, list[int]]]:
        """
        Gets non-duplicate steps in all scheduled sequences.
        """
        dedup_steps: list[Step] = []
        dedup_step_map: dict[int, list[int]] = {}
        dedup_sequence_indices: list[int] = list(dict.fromkeys(self._scheduled_sequences))
        for sequence_index in dedup_sequence_indices:
            sequence = self._sequences[sequence_index]
            dedup_step_map[sequence_index] = []
            for step in sequence.get_steps():
                try:
                    step_index = dedup_steps.index(step)
                except ValueError:
                    dedup_steps.append(step)
                    step_index = len(dedup_steps) - 1
                dedup_step_map[sequence_index].append(step_index)
        return dedup_steps, dedup_step_map

    def program_next_sequence(
        self,
    ) -> Sequence:
        """
        Programs the AWG memory and segment orders and repeats for the next scheduled sequence.

        Returns:
            Sequence: next scheduled sequence object.
        """
        if self._needs_setup:
            raise RuntimeError(
                "There are scheduled sequences that are not set up. Call `setup` first."
            )

        # gets the next scheduled sequence index.
        sequence_index = self._scheduled_sequences[0]
        # programs the segment steps.
        self._program_awg_memory(sequence_index)

        for segmenter_index, segmenter in enumerate(self._segmenters):
            segment_indices_and_repeats: list[tuple[int, int]] = []
            step_repeats = self._sequences[sequence_index].get_repeats()
            for step_order, step_repeat in enumerate(step_repeats):
                segment_indices_and_repeats.append(
                    (
                        self._step_to_segment_maps[segmenter_index][
                            self._sequence_to_steps_map[sequence_index][step_order]
                        ],
                        step_repeat,
                    )
                )
            # programs the segment steps.
            self._program_segment_step_single_device(segmenter_index, segment_indices_and_repeats)

        # removes the first element of scheduled sequences.
        self._scheduled_sequences.popleft()

        # removes the sequence if the it is not in the scheduled sequences.
        if sequence_index not in self._scheduled_sequences:
            sequence = self._sequences.pop(sequence_index)
            for segmenter_index in self._prepared_sequences:
                self._prepared_sequences[segmenter_index].remove(sequence_index)
        else:
            sequence = self._sequences[sequence_index]
        return sequence

    def _program_awg_memory(self, sequence_index_skip_if_programmed: int = None) -> dict[int, Any]:
        """
        Programs memory of each AWG.

        Args:
            sequence_index_skip_if_programmed (int):
                If this sequence index has been programmed in an AWG, it can skip programming.
                If None, programming is done on each AWG device.
        """
        if not self._single_sequence_only:
            sequence_indices_to_prepare = list(dict.fromkeys(self._scheduled_sequences))
        else:
            sequence_indices_to_prepare = [self._scheduled_sequences[0]]

        for segmenter_index, segmenter in enumerate(self._segmenters):
            if (
                sequence_index_skip_if_programmed is not None
                and sequence_index_skip_if_programmed in self._prepared_sequences[segmenter_index]
            ):
                # if the required sequence is already programmed on this device.
                continue
            awg_data, step_to_segment_map, sequence_prepared = segmenter.get_awg_memory_data(
                sequence_indices_to_prepare
            )
            self._program_memory_single_device(segmenter_index, awg_data)
            self._step_to_segment_maps[segmenter_index] = step_to_segment_map

            # constructs a new list to prevent several segmenters sharing the same list object.
            self._prepared_sequences[segmenter_index] = list(sequence_prepared)
