from typing import Literal

import numpy as np

# Requires https://github.com/SpectrumInstrumentation/spcm
import spcm
from spcm.classes_sequence import Segment

CHANNEL_TYPE = Literal[0, 1, 2, 3]
MAX_SAMPLE_RATE = 625000000


class M4i6622Driver:
    """
    Driver for the M4i6622 arbitrary waveform generator.
    """

    def __init__(
        self,
        resource: str,
        digital_to_analog_map: dict[int, int],
        external_clock_frequency: int = 10000000,
        principal_card: bool = False,
    ):
        self._principal_card = principal_card
        self._hcard = spcm.Card(resource).open()
        self._hcard.reset()
        self._bytes_per_sample = self._hcard.bytes_per_sample()

        clock = spcm.Clock(self._hcard)
        if external_clock_frequency is not None:
            clock.mode(spcm.SPC_CM_EXTREFCLOCK)
            clock.reference_clock(external_clock_frequency)
        else:
            clock.mode(spcm.SPC_CM_INTPLL)

        self._sample_rate = clock.sample_rate(clock.max_sample_rate())

        self._trigger = spcm.Trigger(self._hcard)
        self._setup_external_trigger()
        self._trigger.and_mask(spcm.SPC_TMASK_NONE)
        self._trigger.termination(1)  # 1: 50 ohm, 0: high impedance.
        self._trigger.ext0_level0(750)
        self._trigger.ext0_level1(1500)
        self._trigger.ext0_mode(spcm.SPC_TM_POS)
        if principal_card:
            self._setup_software_trigger()

        channels = spcm.Channels(self._hcard)
        channels.channels_enable(enable_all=True)  # all channels are used.
        channels.enable()  # turn on the amplifier for each channel.
        channels.amp(2500)  # max amplitude is 2.5 V.
        channels.filter(0)  # turn off analog filter.

        for digital_ch in [0, 1, 2]:
            multi_io = spcm.MultiPurposeIO(self._hcard, digital_ch)
            mode = spcm.SPCM_XMODE_DIGOUT
            mode = mode | getattr(
                spcm, f"SPCM_XMODE_DIGOUTSRC_CH{digital_to_analog_map[digital_ch]}"
            )
            mode = mode | spcm.SPCM_XMODE_DIGOUTSRC_BIT15
            multi_io.x_mode(mode)
        self._sequence = spcm.Sequence(self._hcard)
        self._current_segment_index = None

    # sequence functions.
    def _set_segment_step_memory(
        self,
        step_number: int,
        segment_number: int,
        next_step: int,
        loops: int,
        end: Literal["end_loop", "end_loop_on_trig", "end_sequence"] = "end_loop",
    ):
        """Sets the parameters for a segment step.

        For some reason the last "end_sequence" step never outputs.
        Adding a short, empty step at the end solves the problem.
        """
        if end == "end_loop":
            end = spcm.SPCSEQ_ENDLOOPALWAYS
        elif end == "end_loop_on_trig":
            end = spcm.SPCSEQ_ENDLOOPONTRIG
        elif end == "end_sequence":
            end = spcm.SPCSEQ_END
        self._sequence.step_memory(step_number, next_step, segment_number, loops, end)

    def _get_segment_start_step(self) -> int:
        return self._sequence.start_step()

    def _set_segment_start_step(self, start_step_number: int):
        self._sequence.start_step(start_step_number)

    def _get_segment_current_step(self) -> int:
        return self._sequence.current_step()

    def _set_sequence_max_segments(self, segments: int):
        self._sequence.max_segments(segments)

    def _set_segment_data(self, segment_number: int, sample_size: int, data: np.ndarray) -> int:
        segment = Segment(self._hcard, segment_number, data, sample_size)
        segment.update()

    # trigger functions.
    def _setup_external_trigger(self):
        self._trigger.or_mask(spcm.SPC_TMASK_EXT0)

    def _setup_software_trigger(self):
        self._trigger.or_mask(spcm.SPC_TMASK_SOFTWARE)

    def _enable_triggers(self):
        """Enables detecting triggers."""
        self._trigger.enable()

    def _force_trigger(self):
        """Sends a software trigger."""
        self._trigger.force()

    def _disable_triggers(self):
        """Disables detecting triggers."""
        self._trigger.disable()

    # card control
    def _write_setup(self):
        """Writes the setup without starting the card."""
        self._hcard.write_setup()

    def _start(self):
        """Writes the setup and starts the card."""
        self._hcard.start()

    def _stop(self):
        """Stops the current run of the card."""
        self._hcard.stop()

    def _wait_for_trigger(self):
        """Waits until the first trigger event has been detected by the card."""
        self._hcard.cmd(spcm.M2CMD_CARD_WAITTRIGGER)

    def _wait_for_complete(self):
        """Waits until the card has completed the current run."""
        self._hcard.cmd(spcm.M2CMD_CARD_WAITREADY)
