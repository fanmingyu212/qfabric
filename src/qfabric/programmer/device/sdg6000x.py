from typing import Any

from qfabric.planner.segmenter.sdg6000x import SDG6000XSegment, SDG6000XSegmenter
from qfabric.programmer.device import Device
from qfabric.programmer.driver.sdg6000x import SDG6000XDriver


class SDG6000XDevice(Device):
    """
    Programming interface of the Siglent SDG6000X series AWG.

    Each segment step needs an external trigger.
    The output of each segment step starts 2 us after the trigger.
    Use another pulse generator to generate pulses to trigger the segment steps.
    The spacing between the pulses from the pulse generators needs to be ~5 us longer
    thant the actual step length to reliably output the correct sequence.

    Args:
        segmenter (SDG6000XSegmenter): Segmenter for this AWG device.
        resource (str): Resource name of the device.
        **kwargs:
            See :class:`~qfabric.programmer.driver.sdg6000x.SDG6000XDriver` for
            optional keyword arguments.
    """

    def __init__(self, segmenter: SDG6000XSegmenter, resource: str, **kwargs):
        super().__init__(segmenter, resource)
        self._driver = SDG6000XDriver(resource, **kwargs)

    def program_memory(self, instructions: dict[str, Any]):
        self.stop()
        segments: list[SDG6000XSegment] = instructions["segments"]

        sample_rate = None
        for segment_index, segment in enumerate(segments):
            for channel in [1, 2]:
                if channel == 1:
                    voltages = segment.analog_data_1
                else:
                    voltages = segment.analog_data_2
                if sample_rate is None:
                    sample_rate = segment.sample_rate
                else:
                    if sample_rate != segment.sample_rate:
                        raise ValueError("Variable sample rate in a sequence is not allowed.")
                filename = f"custom{segment_index}ch{channel}.bin"
                self._driver.transfer_waveform(filename, voltages)
        if sample_rate is not None:
            for channel in [1, 2]:
                self._driver.set_sample_rate(channel, sample_rate)
        self._driver.opc()

    def program_segment_steps(self, segment_indices_and_repeats: list[tuple[int, int]]):
        self.stop()
        for channel in [1, 2]:
            self._driver.clear_segments(channel)
        for step_index, (segment_index, repeats) in enumerate(segment_indices_and_repeats):
            step_index += 1
            for channel in [1, 2]:
                if step_index > 1:
                    self._driver.add_segment(channel)
                    self._driver.set_segment_goto(channel, step_index - 1, step_index)
                self._driver.set_segment_waveform(
                    channel, step_index, f"custom{segment_index}ch{channel}.bin"
                )
                self._driver.set_segment_amplitude(channel, step_index, 10)
                self._driver.set_segment_offset(channel, step_index, 0)
                self._driver.set_segment_repeats(channel, step_index, repeats)
        self._driver.opc()

    def start(self):
        """This function does not start the output. It will wait for the trigger before output."""
        self._driver.set_run_state(1, True)
        self._driver.opc()
        self._driver.set_run_state(2, True)
        self._driver.opc()
        self._driver.set_output_state(1, True)
        self._driver.opc()
        self._driver.set_output_state(2, True)
        self._driver.opc()

    def wait_until_complete(self):
        """This device does not have a method to query whether the output has completed."""
        raise NotImplementedError()

    def stop(self):
        self._driver.set_output_state(1, False)
        self._driver.opc()
        self._driver.set_output_state(2, False)
        self._driver.opc()
        self._driver.set_run_state(1, False)
        self._driver.opc()
        self._driver.set_run_state(2, False)
        self._driver.opc()

    def setup_external_trigger(self):
        """The current code only supports external trigger."""
        pass

    def setup_software_trigger(self):
        raise NotImplementedError("This device does not support software trigger.")
