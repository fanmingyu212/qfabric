from abc import abstractmethod
from typing import Any

from qfabric.planner.segmenter import Segmenter


class Device:
    """
    Base class of the AWG device interface.

    Each AWG model should subclass :class:`Device` and override :meth:`program_memory`,
    :meth:`program_segment_steps`, :meth:`start`, :meth:`wait_until_complete`, :meth:`stop`,
    :meth:`setup_external_trigger`, and :meth:`setup_software_trigger`.

    Args:
        segmenter (Segmenter): Segmenter for this AWG device.
        resource (str): Resource name of the device.
    """

    def __init__(self, segmenter: Segmenter, resource: str):
        self._segmenter = segmenter
        self._resource = resource
        self.is_principal_device = self._segmenter.trigger_device

    @abstractmethod
    def program_memory(self, instructions: Any):
        """
        Programs the memory.

        Override this function to define how the AWG memory is programmed.

        Args:
            instructions (Any): The instruction format should be in agreement with the segmenter.
        """

    @abstractmethod
    def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
        """
        Programs the segment steps.

        Override this function to define how the AWG segment step is programmed.

        Args:
            segment_indices_and_repeats (list[tuple[int, int]]):
                List of (segment_index, segment_repeat).
        """

    @abstractmethod
    def start(self):
        """
        Starts the currently programmed sequence.
        """

    @abstractmethod
    def wait_until_complete(self):
        """
        Waits until the currently running sequence is finished.
        """

    @abstractmethod
    def stop(self):
        """
        Stops the currently running sequence.
        """

    @abstractmethod
    def setup_external_trigger(self):
        """
        Sets the device to use external trigger.
        """

    @abstractmethod
    def setup_software_trigger(self):
        """
        Sets the device to use software trigger.
        """
