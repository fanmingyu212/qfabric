from typing import Any

from qfabric.programmer.device import Device


class Programmer:
    """
    Programming interface to all AWGs.

    Args:
        devices (list[Device]): AWG devices.

    Attributes:
        devices (list[Device]): AWG devices.
    """

    def __init__(self, devices: list[Device]):
        self.devices = devices

        has_principal_device = False
        for device in self.devices:
            if device.is_principal_device and has_principal_device:
                raise ValueError("Cannot have two principal devices.")
            elif device.is_principal_device:
                has_principal_device = True

    def program_memory_single_device(self, device_index: int, instructions: Any):
        """
        Program the memory of a single AWG device given instructions.

        Args:
            device_index (int): Index of the AWG device.
            instructions (Any): Instructions to program the AWG memory.
        """
        self.devices[device_index].program_memory(instructions)

    def program_segment_step_single_device(
        self, device_index: int, segment_indices_and_repeats: list[tuple[int, int]]
    ):
        """
        Program the segment steps of a single AWG device given instructions.

        Args:
            device_index (int): Index of the AWG device.
            segment_indices_and_repeats (list[tuple[int, int]]):
                Segment indices and segment repeats in the order of execution.
        """
        self.devices[device_index].program_segment_steps(segment_indices_and_repeats)

    def run(self, wait_for_finish: bool):
        """
        Executes the currently loaded sequence.

        Args:
            wait_for_finish (bool):
                Wait for the currently programmed sequence to run before return.
                Otherwise, call :meth:`wait_until_complete` and :meth:`stop` to wait and stop.
        """
        principal_device = None
        # if there is a principal device (device triggering other AWGs), turn it on the last.
        for device in self.devices:
            if device.is_principal_device:
                principal_device = device
            else:
                device.start()
        if principal_device is not None:
            principal_device.start()
        if wait_for_finish:
            self.wait_until_complete()
            self.stop()

    def wait_until_complete(self):
        """
        Waits for all devices to finish.
        """
        for device in self.devices:
            device.wait_until_complete()

    def stop(self):
        """
        Stops all devices.
        """
        for device in self.devices:
            device.stop()

    def set_principal_device_trigger(self, external: bool):
        """
        Sets trigger mode of the principal device.

        Args:
            external (bool): Uses external trigger.
        """
        for device in self.devices:
            if device.is_principal_device:
                if external:
                    device.setup_external_trigger()
                else:
                    device.setup_software_trigger()
