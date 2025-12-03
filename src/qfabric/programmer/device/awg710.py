import time
from datetime import datetime
from typing import Any

import numpy as np

from qfabric.planner.segmenter.awg710 import AWG710Segment, AWG710Segmenter
from qfabric.programmer.device import Device
from qfabric.programmer.driver.awg710 import AWG710Driver


class AWG710Device(Device):
    """
    Programming interface of the Tektronix AWG710 AWG.

    Args:
        segmenter (AWG710Segmenter): Segmenter for this AWG device.
        resource (str): Resource name of the device.
        **kwargs:
            See :class:`~qfabric.programmer.driver.awg710.AWG710Driver` for
            optional keyword arguments.
    """

    def __init__(self, segmenter: AWG710Segmenter, resource: str, **kwargs):
        super().__init__(segmenter, resource)
        self._driver = AWG710Driver(resource, self.is_principal_device, **kwargs)
        self._file_folder: str = None

    def program_memory(self, instructions: dict[str, Any]):
        segments: list[AWG710Segment] = instructions["segments"]
        self._file_folder = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
        self._driver.cd_ftp("/")
        self._driver.mkdir(self._file_folder)
        self._driver.cd_ftp("/" + self._file_folder)

        for segment_index, segment in enumerate(segments):
            filename = f"pattern_{segment_index}.PAT"
            self._driver.create_pattern_file(
                filename,
                segment.analog_data,
                segment.digital_1,
                segment.digital_2,
                segment.sample_rate,
            )
        filename = "start.PAT"
        self._driver.create_pattern_file(
            filename,
            np.zeros(960, dtype=float),
            np.zeros(960, dtype=bool),
            np.zeros(960, dtype=bool),
            segment.sample_rate,
        )

    def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
        sequence_steps: list[tuple[str, int, int]] = [("start.PAT", 1, 2)]
        for step_index, (segment_index, repeat) in enumerate(segment_indices_and_repeats):
            sequence_steps.append([f"pattern_{segment_index}.PAT", repeat, step_index + 3])
        sequence_steps[-1] = (sequence_steps[-1][0], sequence_steps[-1][1], 0)

        self._driver.cd_ftp("/" + self._file_folder)
        self._driver.create_sequence_file("sequence.SEQ", sequence_steps)
        self._driver.cd_telnet("/" + self._file_folder)
        self._driver.set_waveform_file("sequence.SEQ")

    def start(self):
        self._driver.start()
        self._driver.trigger_now()

    def wait_until_complete(self):
        while self._driver.get_run_state() == 2:
            time.sleep(1e-3)

    def stop(self):
        self._driver.stop()

    def setup_external_trigger(self):
        self._driver.set_trigger_source(external=True)

    def setup_software_trigger(self):
        """
        This still uses the external trigger.

        The force trigger works for external trigger.
        If set to internal trigger, it runs a periodic trigger internally.
        """
        self._driver.set_trigger_source(external=True)
