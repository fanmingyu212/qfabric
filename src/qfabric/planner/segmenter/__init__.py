from abc import abstractmethod
from typing import Any

from qfabric.sequence.step import DeviceStep, Step


class Segment:
    """
    Base class representing AWG data of a DeviceStep.

    Args:
        device_step (DeviceStep): device step containing AWG functions on a single device.
    """

    def __init__(self, device_step: DeviceStep):
        self._device_step = device_step


class Segmenter:
    """
    Base class for converting from :class:`Step` to :class:`Segment` on a AWG device.

    It converts from pulse sequence steps (analog and digital functions) to AWG-specific data.
    Data stored in the :class:`Segment` objects should be compatible with the AWG used,
    respecting all specifications and constraints of the AWG.

    Each different AWG model (or even the same AWG model with very different usage) generally
    needs a subclass of :class:`Segmenter` converting :class:`Step` objects to :class:`Segment`
    objects. See :class:`~qfabric.planner.segmenter.mock.MockSegmenter`
    and :class:`~qfabric.planner.segmenter.m4i6622.M4i6622Segmenter` for examples.

    Each subclass of :class:`Segmenter` needs to override :meth:`set_steps` and
    :meth:`get_awg_memory_data`.

    Args:
        analog_channel (list[int]): Analog channel indices.
        digital_channels (list[int]): Digital channel indices.

    Attributes:
        trigger_device (bool): Whether this AWG is used to trigger other AWGs.
        _device_steps (list[DeviceStep]):
            See :meth:`set_steps`, device steps scheduled on this device.
            All unique device_steps are saved in it.
        _sequence_to_device_steps_map (dict[int, list[int]]):
            See :meth:`set_steps`, mapping from sequence indices to device step indices.
    """

    def __init__(self, analog_channels: list[int], digital_channels: list[int]):
        self._analog_channels = analog_channels
        self._digital_channels = digital_channels
        self.trigger_device = False

    @abstractmethod
    def set_steps(self, steps: list[Step], sequence_to_steps_map: dict[int, list[int]]):
        """
        Gets a list of device steps and a list of segments from a list of steps.

        This function converts *steps* to :attr:`_device_steps`, which is a list of
        :class:`~qfabric.sequence.step.DeviceStep` with the same length and order as *steps*.
        It also stores the map from sequence indices to lists of *steps* indices,
        *sequence_to_steps_map*, in :attr:`_sequence_to_device_steps_map`.

        Args:
            steps (list[Step]): All steps in sequences that are scheduled.
            sequence_to_steps_map (dict[int, list[int]]):
                Mapping from sequence indices to lists to step indices in *steps*.
                Keys should be in the order of sequence scheduling.
                Values should be in the order of step executation in each sequence.
        """
        self._device_steps: list[DeviceStep] = self._get_single_device_device_steps(steps)
        self._sequence_to_device_steps_map = sequence_to_steps_map

    def _get_single_device_device_steps(self, steps: list[Step]) -> list[DeviceStep]:
        """
        Converts a list of :class:`Step` to a list of :class:`DeviceStep` for this AWG device.
        """
        device_steps: list[DeviceStep] = []
        for step in steps:
            device_steps.append(
                step.get_functions_on_device(self._analog_channels, self._digital_channels)
            )
        return device_steps

    @abstractmethod
    def get_awg_memory_data(
        self, sequence_indices: list[int]
    ) -> tuple[Any, dict[int, int], list[int]]:
        """
        Gets AWG programming information given sequences that need to be programmed.

        Args:
            sequence_indices (list[int]):
                Sequence indices to be programmed. The indices must match the indices
                from :attr:`_sequence_to_device_steps_map`.

        Returns:
            (Any, dict[int, int], list[int]):

            (*awg_data*, *step_to_segment_map*, *sequence_indices_programmed*)
                *awg_data*: Data sent to the programming device
                (:class:`~qfabric.programmer.device.Device`) to program the segments.
                The data usually contains a list of :class:`~qfabric.planner.segmenter.Segment`
                objects, representing the segments to be programmed.

                *step_to_segment_map*: Mapping from step indices to segment indices. Steps
                are indexed with :attr:`_device_steps` and segments are indexed with the
                list of segments returned in *awg_data*.

                *sequence_indices_programmed*: List of sequence indexed programmed.
        """
