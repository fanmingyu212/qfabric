from typing import Literal

import numpy as np

import qfabric.programmer.driver.m4i6622.pyspcm as pyspcm
from qfabric.programmer.driver.m4i6622.spcm_tools import pvAllocMemPageAligned

CHANNEL_TYPE = Literal[0, 1, 2, 3]
MAX_SAMPLE_RATE = 625000000


class M4i6622Driver:
    """
    Driver for the M4i6622 arbitrary waveform generator.
    """

    def __init__(
        self,
        resource: str,
        ttl_to_awg_map: dict[int, int],
        external_clock_frequency: int = 10000000,
        principal_card: bool = False,
    ):
        self._principal_card = principal_card
        self._hcard = pyspcm.spcm_hOpen(pyspcm.create_string_buffer(resource.encode("ascii")))
        if self._hcard is None:
            raise Exception(f"No card is found at {resource}.")
        self._aligned_buffer = None
        self._reset()
        self._bytes_per_sample = self._get_bytes_per_sample()

        if external_clock_frequency is not None:
            self._set_clock_mode("external_reference")
            self._set_external_clock_frequency(external_clock_frequency)
        else:
            self._set_clock_mode("internal_pll")

        self._set_sample_rate()
        self._sample_rate = self._get_sample_rate()

        if principal_card:
            self._setup_software_trigger()
        else:
            self._setup_external_trigger()
        self._set_trigger_and_mask(pyspcm.SPC_TMASK_NONE)
        self._set_ext0_trigger_impedance(is_50_ohm=True)
        self._set_ext0_trigger_level(level=750)
        self._set_ext0_trigger_rising_edge()

        self._awg_channels = list(range(4))
        self._ttl_channels = list(range(3))
        self._select_channels(self._awg_channels)

        for awg_channel in self._awg_channels:
            self._enable_channel(awg_channel)
            self._set_channel_amplitude(awg_channel)
            self._set_channel_filter(awg_channel)

        for ttl_channel in [0, 1, 2]:
            mode = pyspcm.SPCM_XMODE_DIGOUT

            mode = mode | getattr(pyspcm, f"SPCM_XMODE_DIGOUTSRC_CH{ttl_to_awg_map[ttl_channel]}")

            mode = mode | pyspcm.SPCM_XMODE_DIGOUTSRC_BIT15
            self._set_multi_purpose_io_mode(ttl_channel, mode)

    # device methods
    def _get_bytes_per_sample(self) -> int:
        value = pyspcm.int32(0)
        ret = pyspcm.spcm_dwGetParam_i32(
            self._hcard, pyspcm.SPC_MIINST_BYTESPERSAMPLE, pyspcm.byref(value)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get bytes per sample failed with code {ret}.")
        return value.value

    def _select_channels(self, channels: list[CHANNEL_TYPE]):
        if len(channels) == 3:
            raise ValueError("Cannot enable 3 channels. Enable 4 channels instead.")
        value = 0
        for channel in channels:
            value = value | getattr(pyspcm, f"CHANNEL{channel}")
        ret = pyspcm.spcm_dwSetParam_i64(self._hcard, pyspcm.SPC_CHENABLE, value)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Select channels failed with code {ret}.")

    def _enable_channel(self, channel: CHANNEL_TYPE, enabled: bool = True):
        if enabled:
            value = pyspcm.int64(1)
        else:
            value = pyspcm.int64(0)
        ret = pyspcm.spcm_dwSetParam_i64(
            self._hcard, getattr(pyspcm, f"SPC_ENABLEOUT{channel}"), value
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Enable channel failed with code {ret}.")

    def _set_channel_amplitude(self, channel: CHANNEL_TYPE, amplitude_mV: int = 2500):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard,
            getattr(pyspcm, f"SPC_AMP{channel}"),
            pyspcm.int32(amplitude_mV),
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set channel amplitude failed with code {ret}.")

    def _set_channel_filter(self, channel: CHANNEL_TYPE, filter: int = 0):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, getattr(pyspcm, f"SPC_FILTER{channel}"), pyspcm.int32(filter)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set channel filter failed with code {ret}.")

    def _set_mode(
        self,
        mode: Literal[
            "single",
            "multiple",
            "single_restart",
            "sequence",
        ],
    ):
        """See replay modes in the manual. Not all modes are implemented here."""
        if mode == "single":
            value = pyspcm.SPC_REP_STD_SINGLE
        elif mode == "multiple":
            value = pyspcm.SPC_REP_STD_MULTI
        elif mode == "single_restart":
            value = pyspcm.SPC_REP_STD_SINGLERESTART
        elif mode == "sequence":
            value = pyspcm.SPC_REP_STD_SEQUENCE
        else:
            raise ValueError(f"The replay mode {mode} is invalid or not implemented.")
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_CARDMODE, value)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set mode failed with code {ret}.")

    def _reset(self):
        """Resets the card."""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_RESET)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Reset failed with code {ret}.")

    def _write_setup(self):
        """Writes the setup without starting the card."""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_WRITESETUP
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Write setup failed with code {ret}.")

    def _start(self):
        """Writes the setup and starts the card."""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_START)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Start card failed with code {ret}.")

    def _enable_triggers(self):
        """Enables detecting triggers."""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_ENABLETRIGGER
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Enable triggers failed with code {ret}.")

    def _force_trigger(self):
        """Sends a software trigger."""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_FORCETRIGGER
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Force trigger failed with code {ret}.")

    def _disable_triggers(self):
        """Disables detecting triggers."""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_DISABLETRIGGER
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Disable triggers failed with code {ret}.")

    def _stop(self):
        """Stops the current run of the card."""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_STOP)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Stop card failed with code {ret}.")

    def _wait_for_trigger(self):
        """Waits until the first trigger event has been detected by the card."""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_WAITTRIGGER
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Wait for trigger failed with code {ret}.")

    def _wait_for_complete(self):
        """Waits until the card has completed the current run."""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_CARD_WAITREADY)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Wait for complete failed with code {ret}.")

    def _define_transfer_buffer(self, data: np.ndarray, transfer_offset: int = 0) -> int:
        self._aligned_buffer = pvAllocMemPageAligned(len(data) * self._bytes_per_sample)
        # this variable must maintain a reference after exit.
        data = data.astype(np.int16)
        pyspcm.memmove(self._aligned_buffer, data.ctypes.data, 2 * len(data))
        ret = pyspcm.spcm_dwDefTransfer_i64(
            self._hcard,
            pyspcm.SPCM_BUF_DATA,
            pyspcm.SPCM_DIR_PCTOCARD,
            pyspcm.uint32(0),
            self._aligned_buffer,
            pyspcm.uint64(transfer_offset),
            pyspcm.uint64(len(self._aligned_buffer)),
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Define transfer buffer failed with code {ret}.")
        return len(self._aligned_buffer)

    def _get_data_ready_to_transfer(self) -> int:
        value = pyspcm.int32(0)
        ret = pyspcm.spcm_dwGetParam_i32(
            self._hcard, pyspcm.SPC_DATA_AVAIL_USER_LEN, pyspcm.byref(value)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get data ready to transfer failed with code {ret}.")
        return value.value

    def _set_data_ready_to_transfer(self, data_bytes: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_DATA_AVAIL_CARD_LEN, pyspcm.int32(data_bytes)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set data ready to transfer failed with code {ret}.")

    def _start_dma_transfer(self, wait: bool = False):
        if wait:
            value = pyspcm.M2CMD_DATA_STARTDMA | pyspcm.M2CMD_DATA_WAITDMA
        else:
            value = pyspcm.M2CMD_DATA_STARTDMA
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, value)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Start DMA transfer failed with code {ret}.")

    def _wait_dma_transfer(self):
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_DATA_WAITDMA)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Wait DMA transfer failed with code {ret}.")

    def _stop_dma_transfer(self):
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_M2CMD, pyspcm.M2CMD_DATA_STOPDMA)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Stop DMA transfer failed with code {ret}.")

    def _set_memory_size(self, samples_per_channel: int):
        ret = pyspcm.spcm_dwSetParam_i64(
            self._hcard, pyspcm.SPC_MEMSIZE, pyspcm.int64(samples_per_channel)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set memory size failed with code {ret}.")

    def _set_segment_size(self, samples_per_segment: int):
        ret = pyspcm.spcm_dwSetParam_i64(
            self._hcard, pyspcm.SPC_SEGMENTSIZE, pyspcm.int64(samples_per_segment)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set segment size failed with code {ret}.")

    def _set_number_of_loops(self, loops: int):
        ret = pyspcm.spcm_dwSetParam_i64(self._hcard, pyspcm.SPC_LOOPS, pyspcm.int64(loops))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set number of loops failed with code {ret}.")

    def _set_clock_mode(
        self,
        clock_mode: Literal["internal_pll", "quartz2", "external_reference", "pxi_reference"],
    ):
        if clock_mode == "internal_pll":
            value = pyspcm.SPC_CM_INTPLL
        elif clock_mode == "quartz2":
            value = pyspcm.SPC_CM_QUARTZ2
        elif clock_mode == "external_reference":
            value = pyspcm.SPC_CM_EXTREFCLOCK
        elif clock_mode == "pxi_reference":
            value = pyspcm.SPC_CM_PXIREFCLOCK
        else:
            raise ValueError(f"Clock mode {clock_mode} is not valid.")
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_CLOCKMODE, value)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set clock mode failed with code {ret}.")

    def _get_sample_rate(self) -> int:
        value = pyspcm.int64(0)
        ret = pyspcm.spcm_dwGetParam_i64(self._hcard, pyspcm.SPC_SAMPLERATE, pyspcm.byref(value))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get sample rate failed with code {ret}.")
        return value.value

    def _set_sample_rate(self, sample_rate: int = MAX_SAMPLE_RATE):
        ret = pyspcm.spcm_dwSetParam_i64(
            self._hcard, pyspcm.SPC_SAMPLERATE, pyspcm.int64(sample_rate)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set sample rate failed with code {ret}.")

    def _set_external_clock_frequency(self, clock_frequency: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_REFERENCECLOCK, pyspcm.int32(clock_frequency)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set external clock frequency failed with code {ret}.")

    def _set_trigger_or_mask(self, trigger_mask: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_TRIG_ORMASK, pyspcm.int32(trigger_mask)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger OR mask failed with code {ret}.")

    def _set_trigger_and_mask(self, trigger_mask: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_TRIG_ANDMASK, pyspcm.int32(trigger_mask)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger AND mask failed with code {ret}.")

    def _set_ext0_trigger_impedance(self, is_50_ohm=True):
        """Set the trigger impedance"""
        if is_50_ohm:
            ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_TERM, pyspcm.int32(1))
        else:
            ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_TERM, pyspcm.int32(0))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger impedance failed with code {ret}.")

    def _set_ext0_trigger_level(self, level):
        """Set the trigger level"""
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_TRIG_EXT0_LEVEL0, pyspcm.int32(level)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger level failed with code {ret}.")

    def _set_ext0_trigger_rising_edge(self):
        """Set trigger edge to rising or falling"""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_EXT0_MODE, pyspcm.SPC_TM_POS)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger edge to rising failed with code {ret}.")

    def _set_ext0_trigger_falling_edge(self):
        """Set trigger edge to falling"""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_EXT0_MODE, pyspcm.SPC_TM_NEG)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger edge to falling failed with code {ret}.")

    def _set_ext0_trigger_level_high(self):
        """Trigger detection for high levels (signal above level 0)"""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_EXT0_MODE, pyspcm.SPC_TM_HIGH)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger level to high failed with code {ret}.")

    def _set_ext0_trigger_level_low(self):
        """Trigger detection for low levels (signal below level 0)"""
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_EXT0_MODE, pyspcm.SPC_TM_LOW)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger level to low failed with code {ret}.")

    def _set_trigger_termination(self, use_50_ohm):
        if use_50_ohm:
            value = 1
        else:
            value = 0
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPC_TRIG_TERM, value)
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set trigger termination failed with code {ret}.")

    def _set_multi_purpose_io_mode(self, line_number: Literal[0, 1, 2], mode: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard,
            getattr(pyspcm, f"SPCM_X{line_number}_MODE"),
            pyspcm.int32(mode),
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set multiple purpose io mode failed with code {ret}.")

    def _get_multi_purpose_io_output(self) -> tuple[bool, bool, bool]:
        mode = pyspcm.int32(0)
        ret = pyspcm.spcm_dwGetParam_i32(self._hcard, pyspcm.SPCM_XX_ASYNCIO, pyspcm.byref(mode))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get multi purpose io output failed with code {ret}.")
        return (mode.value & 1 != 0, mode.value & 2 != 0, mode.value & 4 != 0)

    def _set_multi_purpose_io_output(self, x0_state: bool, x1_state: bool, x2_state: bool):
        mode = 0
        if x0_state:
            mode += 1
        if x1_state:
            mode += 2
        if x2_state:
            mode += 4
        ret = pyspcm.spcm_dwSetParam_i32(self._hcard, pyspcm.SPCM_XX_ASYNCIO, pyspcm.int32(mode))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set multi purpose io output failed with code {ret}.")

    def _set_sequence_max_segments(self, segments: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_MAXSEGMENTS, pyspcm.int32(segments)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set sequence max segments failed with code {ret}.")

    def _set_sequence_write_segment(self, segment_number: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_WRITESEGMENT, pyspcm.int32(segment_number)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set sequence write segment failed with code {ret}.")

    def _set_sequence_write_segment_size(self, segment_size: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_SEGMENTSIZE, pyspcm.int32(segment_size)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set sequence write segment size failed with code {ret}.")

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
        register = pyspcm.SPC_SEQMODE_STEPMEM0 + step_number
        if end == "end_loop":
            end = pyspcm.SPCSEQ_ENDLOOPALWAYS
        elif end == "end_loop_on_trig":
            end = pyspcm.SPCSEQ_ENDLOOPONTRIG
        elif end == "end_sequence":
            end = pyspcm.SPCSEQ_END
        value = (end << 32) | (loops << 32) | (next_step << 16) | segment_number
        ret = pyspcm.spcm_dwSetParam_i64(self._hcard, register, pyspcm.int64(value))
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set segment step memory failed with code {ret}.")

    def _get_segment_start_step(self) -> int:
        value = pyspcm.int32(0)
        ret = pyspcm.spcm_dwGetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_STARTSTEP, pyspcm.byref(value)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get segment start step failed with code {ret}.")
        return value.value

    def _set_segment_start_step(self, start_step_number: int):
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_STARTSTEP, pyspcm.int32(start_step_number)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Set segment start step failed with code {ret}.")

    def _get_segment_current_step(self) -> int:
        value = pyspcm.int32(0)
        ret = pyspcm.spcm_dwSetParam_i32(
            self._hcard, pyspcm.SPC_SEQMODE_STATUS, pyspcm.byref(value)
        )
        if ret != pyspcm.ERR_OK:
            raise Exception(f"Get segment current step failed with code {ret}.")
        return value.value

    def _get_error_information(self) -> tuple[int, int, str]:
        error_register = pyspcm.uint32(0)
        error_code = pyspcm.int32(0)
        text = pyspcm.create_string_buffer(1000)
        ret = pyspcm.spcm_dwGetErrorInfo_i32(
            self._hcard, pyspcm.byref(error_register), pyspcm.byref(error_code), pyspcm.byref(text)
        )
        return (error_register.value, error_code.value, text.value.decode("utf-8"))

    def _setup_external_trigger(self):
        self._set_trigger_or_mask(pyspcm.SPC_TMASK_EXT0)

    def _setup_software_trigger(self):
        self._set_trigger_or_mask(pyspcm.SPC_TMASK_SOFTWARE)
