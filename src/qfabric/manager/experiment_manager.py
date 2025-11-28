import importlib

from qfabric.manager.config import load_hardware_config
from qfabric.planner.planner import Planner
from qfabric.planner.segmenter import Segmenter
from qfabric.programmer.device import Device
from qfabric.programmer.programmer import Programmer
from qfabric.sequence.sequence import Sequence


def _dynamic_import(module_path: str, class_name: str):
    """Import a class via module path and class name."""
    module = importlib.import_module(module_path)
    cls = getattr(module, class_name)
    return cls


class ExperimentManager:
    """
    Experiment sequence manager.

    This is the public interface to control AWG devices.

    If controlling individual AWGs is needed, use :attr:`programmer`.

    Args:
        config_path (str): Path to the config file.

    Attributes:
        planner (Planner): Sequence scheduling object.
        programmer (Programmer): AWG programming interface.
    """

    def __init__(self, config_path: str):
        self._config = load_hardware_config(config_path)
        self._load_awgs()

    def _load_awgs(self):
        segmenters: list[Segmenter] = []
        devices: list[Device] = []
        for awg in self._config.awgs:
            segmenter_class = _dynamic_import(awg.segmenter_module, awg.segmenter_class)
            segmenter = segmenter_class(**awg.segmenter_config.model_dump())
            segmenters.append(segmenter)
            device_class = _dynamic_import(awg.device_module, awg.device_class)
            device = device_class(segmenter=segmenter, **awg.device_config.model_dump())
            devices.append(device)
        self.planner = Planner(segmenters)
        self.programmer = Programmer(devices)

        # links the memory and segment step programming functions of the device
        # to the experiment planner.
        self.planner.register_program_single_device(
            self.programmer.program_memory_single_device,
            self.programmer.program_segment_step_single_device,
        )

    def schedule(self, sequences: Sequence | list[Sequence], repeats: int = 1) -> list[int]:
        """
        Schedules sequences with repeats.

        Args:
            sequences (Sequence | list[Sequence]): A single sequence or a list of sequences.
            repeats (int): Number of repeats for *sequences*. Default 1.

        Returns:
            list[int]: List of sequence indices scheduled.
        """
        return self.planner.schedule(sequences, repeats)

    def setup(self, program_single_sequence_only: bool = False):
        """
        Prepares execution of the sequences.

        This must be called before calling :attr:`program_next_sequence`.
        It sends *steps* in all scheduled sequences to segmenters to produce AWG segments.

        Args:
            program_single_sequence_only (bool):
                Whether to only program the next sequence. If True, attempts to program
                as many sequences as possible to fit in the AWG memory.
        """
        self.planner.setup(program_single_sequence_only)

    @property
    def scheduled_sequence_indices(self) -> list[int]:
        """
        Returns:
            list[int]: List of indices of scheduled sequences.
        """
        return list(self.planner.scheduled_sequence_indices)

    def program_next_sequence(self) -> Sequence:
        """
        Programs the AWG memory and segment orders and repeats for the next scheduled sequence.

        Returns:
            Sequence: next scheduled sequence object.
        """
        sequence = self.planner.program_next_sequence()
        return sequence

    def run(self, wait_for_finish: bool = True):
        """
        Runs the next sequence.

        Args:
            wait_for_finish (bool): Whether to wait for sequence to finish.
        """
        self.programmer.run(wait_for_finish)

    def wait_until_complete(self):
        """
        Waits for the sequence to finish.
        """
        self.programmer.wait_until_complete()

    def stop(self):
        """
        Stops the sequence.
        """
        self.programmer.stop()

    def set_principal_device_trigger(self, external: bool):
        """
        Sets the principal AWG device trigger mode.

        Args:
            external (bool): If False, software trigger is used.
        """
        self.programmer.set_principal_device_trigger(external)
