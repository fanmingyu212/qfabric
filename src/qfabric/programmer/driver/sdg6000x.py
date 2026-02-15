import socket
from typing import Literal

import numpy as np

PORT = 5025


class SDG6000XDriver:
    """
    Current it only supports external triggers.
    Internal triggers do not allow synchronous operation of the two channels.

    Each segment step needs an external trigger.
    The output of each segment step starts 2 us after the trigger.
    """

    def __init__(self, address: str):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.s.settimeout(30)
        self.s.connect((address, PORT))
        idn = self.idn()
        if "SDG6" not in idn:
            raise Exception(f"{idn} is not a valid identification.")

        for channel in [1, 2]:
            self.set_sequence_output(channel, True)
            self.set_output_load_highZ(channel, False)
            self.set_amplitude_offset(channel, 0)
            self.set_amplitude_scale(channel, 1)

            self.set_run_state(channel, False)
            self.set_run_mode(channel, False)
            self.set_trigger_external(channel, True)
            self.set_trigger_delay(channel, 2e-6)  # minimum 1.536 us.
            self.set_output_state(channel, True)

    def _query(self, command: str, ask: bool = False) -> str:
        command += "\n"
        self.s.sendall(command.encode("latin1"))
        if ask:
            return self.s.recv(4096).decode("latin1").strip()

    def idn(self) -> str:
        return self._query("*IDN?", ask=True)

    def opc(self) -> int:
        """Some commands require OPC after before running another command.

        For example, between two `set_output_state` commands, OPC is needed for reliable operation.
        """
        return int(self._query("*OPC?", ask=True))

    def set_sequence_output(self, channel: int, state: bool):
        """Enters the sequence mode."""
        if state:
            self._query(f":C{channel}:SEQ ON")
        else:
            self._query(f":C{channel}:SEQ OFF")

    def get_sequence_output(self, channel: int) -> bool:
        return self._query(f":C{channel}:SEQ?", ask=True) == "ON"

    def set_sample_rate(self, channel: int, sample_rate: float):
        """Run state must be False before changing sample rate."""
        self._query(f":C{channel}:SEQ:SRAT {sample_rate}")

    def get_sample_rate(self, channel: int) -> float:
        return float(self._query(f":C{channel}:SEQ:SRAT?", ask=True))

    def set_run_state(self, channel: int, run: bool):
        if run:
            self._query(f":C{channel}:SEQ:STAT RUN")
        else:
            self._query(f":C{channel}:SEQ:STAT STOP")

    def get_run_state(self, channel: int) -> bool:
        """Documentation is incorrect. It return "Runing" when it is running."""
        return self._query(f":C{channel}:SEQ:STAT?", ask=True).lower().startswith("run")

    def set_amplitude_scale(self, channel: int, scale: float):
        self._query(f":C{channel}:SEQ:SCAL {scale * 100}")

    def get_amplitude_scale(self, channel: int) -> float:
        return float(self._query(f":C{channel}:SEQ:SCAL?", ask=True)) / 100

    def set_amplitude_offset(self, channel: int, offset: float):
        self._query(f":C{channel}:SEQ:OFF {offset}")

    def get_amplitude_offset(self, channel: int) -> float:
        return float(self._query(f":C{channel}:SEQ:OFF?", ask=True))

    def set_run_mode(self, channel: int, continuous: bool):
        if continuous:
            self._query(f":C{channel}:SEQ:RMOD CONT")
        else:
            self._query(f":C{channel}:SEQ:RMOD STEP")

    def get_run_mode(self, channel: int) -> bool:
        return self._query(f":C{channel}:SEQ:RMOD?", ask=True) == "CONT"

    def set_start_segment_number(self, channel: int, segment_number: int):
        self._query(f":C{channel}:SEQ:STARTN {segment_number}")

    def get_start_segment_number(self, channel: int) -> int:
        return int(self._query(f":C{channel}:SEQ:STARTN?", ask=True))

    def set_interpolation(
        self, channel: int, interpolation: Literal["LINE", "HOLD", "SINC", "SINC13", "SINC27"]
    ):
        self._query(f":C{channel}:SEQ:INTP {interpolation}")

    def get_interpolation(self, channel: int) -> str:
        return self._query(f":C{channel}:SEQ:INTP?", ask=True)

    def set_trigger_external(self, channel: int, external: bool):
        if external:
            self._query(f":C{channel}:SEQ:TRIG:SOUR EXT")
        else:
            self._query(f":C{channel}:SEQ:TRIG:SOUR MAN")

    def get_trigger_external(self, channel: int) -> bool:
        return self._query(f":C{channel}:SEQ:TRIG:SOUR?", ask=True) == "EXT"

    def set_trigger_slope_positive(self, channel: int, positive: bool):
        if positive:
            self._query(f":C{channel}:SEQ:TRIG:SLOP RISE")
        else:
            self._query(f":C{channel}:SEQ:TRIG:SLOP FALL")

    def get_trigger_slope_positive(self, channel: int) -> bool:
        return self._query(f":C{channel}:SEQ:TRIG:SLOP?", ask=True) == "RISe"

    def software_trigger(self, channel: int):
        self._query(f":C{channel}:SEQ:TRIG:TRIG")

    def set_trigger_hold(self, channel: int, hold_type: Literal["END", "MID", "START"]):
        """In the Chinese language manual it says that it sets the output voltage when not outputting."""
        self._query(f":C{channel}:SEQ:TRIG:HOLD {hold_type}")

    def get_trigger_hold(self, channel: int) -> str:
        return self._query(f":C{channel}:SEQ:TRIG:HOLD?", ask=True)

    def set_trigger_delay(self, channel: int, delay_time: float):
        self._query(f":C{channel}:SEQ:TRIG:DELAY {delay_time}")

    def get_trigger_delay(self, channel: int) -> float:
        return float(self._query(f":C{channel}:SEQ:TRIG:DELAY?", ask=True))

    def set_trigger_out(self, channel: int, out_type: Literal["UP", "DOWN", "OFF"]):
        self._query(f":C{channel}:SEQ:TRIG:OUT {out_type}")

    def get_trigger_out(self, channel: int) -> str:
        return self._query(f":C{channel}:SEQ:TRIG:OUT?", ask=True)

    def set_increasing(self, channel: int, mode: Literal["INT", "ZERO", "HLAS", "DUPL"]):
        """How does it oversamples the sequence."""
        self._query(f":C{channel}:SEQ:INCR {mode}")

    def get_increasing(self, channel: int) -> str:
        return self._query(f":C{channel}:SEQ:INCR?", ask=True)

    def set_decreasing(self, channel: int, mode: Literal["DECI", "CTAI", "CHEa"]):
        """How does it undersamples the sequence."""
        self._query(f":C{channel}:SEQ:DECR {mode}")

    def get_decreasing(self, channel: int) -> str:
        return self._query(f":C{channel}:SEQ:DECR?", ask=True)

    def add_segment(self, channel: int):
        self._query(f":C{channel}:SEQ:SEGM:Add")

    def insert_segment(self, channel: int, insert_after: int):
        self._query(f":C{channel}:SEQ:SEGM{insert_after}:INSE")

    def delete_segment(self, channel: int, index: int):
        self._query(f":C{channel}:SEQ:SEGM{index}:DELE")

    def clear_segments(self, channel: int):
        self._query(f":C{channel}:SEQ:SEGM:Clear")

    def set_segment_goto(self, channel: int, from_index: int, goto_index: int):
        self._query(f":C{channel}:SEQ:SEGM{from_index}:GOTO {goto_index}")

    def get_segment_goto(self, channel: int, from_index: int) -> int:
        return int(self._query(f":C{channel}:SEQ:SEGM{from_index}:GOTO?", ask=True))

    def set_segment_length(self, channel: int, index: int, length: int):
        self._query(f":C{channel}:SEQ:SEGM{index}:LENG {length}")

    def get_segment_length(self, channel: int, index: int) -> int:
        return int(self._query(f":C{channel}:SEQ:SEGM{index}:LENG?", ask=True))

    def set_segment_repeats(self, channel: int, index: int, count: int):
        self._query(f":C{channel}:SEQ:SEGM{index}:REP:COUN {count}")

    def get_segment_repeats(self, channel: int, index: int) -> int:
        return int(self._query(f":C{channel}:SEQ:SEGM{index}:REP:COUN?", ask=True))

    def set_segment_waveform(self, channel: int, index: int, waveform: str):
        self._query(f':C{channel}:SEQ:SEGM{index}:WAV "Local/{waveform}"')

    def get_segment_waveform(self, channel: int, index: int) -> str:
        return self._query(f":C{channel}:SEQ:SEGM{index}:WAV?", ask=True)

    def set_segment_amplitude(self, channel: int, index: int, amplitude_Vpp: float):
        self._query(f":C{channel}:SEQ:SEGM{index}:AMP {amplitude_Vpp}")

    def get_segment_amplitude(self, channel: int, index: int) -> float:
        return float(self._query(f":C{channel}:SEQ:SEGM{index}:AMP?", ask=True))

    def set_segment_offset(self, channel: int, index: int, offset: float):
        self._query(f":C{channel}:SEQ:SEGM{index}:OFF {offset}")

    def get_segment_offset(self, channel: int, index: int) -> float:
        return float(self._query(f":C{channel}:SEQ:SEGM{index}:OFF?", ask=True))

    def set_output_state(self, channel: int, state: bool):
        if state:
            self._query(f":C{channel}:OUTP ON")
        else:
            self._query(f":C{channel}:OUTP OFF")

    def get_output_state(self, channel: int) -> bool:
        ret = self._query(f":C{channel}:OUTP?", ask=True)
        data = ret.split(" ")[-1]
        return data.split(",")[0] == "ON"

    def set_output_load_highZ(self, channel: int, high_Z: bool):
        if high_Z:
            self._query(f":C{channel}:OUTP LOAD,HZ")
        else:
            self._query(f":C{channel}:OUTP LOAD,50")

    def get_output_load_highZ(self, channel: int) -> bool:
        ret = self._query(f":C{channel}:OUTP?", ask=True)
        data = ret.split(" ")[-1]
        return data.split(",")[2] == "HZ"

    def set_output_polarity(self, channel: int, normal: bool):
        if normal:
            self._query(f":C{channel}:OUTP PLRT,NOR")
        else:
            self._query(f":C{channel}:OUTP PLRT,INVT")

    def get_output_polarity(self, channel: int) -> bool:
        ret = self._query(f":C{channel}:OUTP")
        data = ret.split(" ")[-1]
        return data.split(",")[4] == "NOR"

    def set_waveform_data(self, channel: int, name: str, waveform_bytes: bytes):
        byte_str = f"b'0x{waveform_bytes.hex()}"
        self._query(f':C{channel}:MVDT WVNM,"Local/{name}",WAVEDATA,{byte_str}')

    def transfer_waveform(self, filename: str, voltages: list[float]):
        waveform_str = self._waveform_voltages_to_bytes(voltages).decode("latin1")
        waveform_length = str(len(waveform_str))
        waveform_length_length = str(len(waveform_length))
        waveform = f"#{waveform_length_length}{waveform_length}{waveform_str}"
        self._query(f'MMEM:TRAN "Local/{filename}",{waveform}')

    def _waveform_voltages_to_bytes(self, voltages: list[float]) -> bytes:
        voltage_amplitude_V = 5  # 10 Vpp at 50 ohm
        voltage_amplitude_machine_unit = 32768
        voltage_machine_units = (
            np.array(voltages) / voltage_amplitude_V * voltage_amplitude_machine_unit
        )
        voltage_machine_units = voltage_machine_units.astype(int)
        voltage_machine_units[voltage_machine_units > 32767] = 32767
        voltage_machine_units[voltage_machine_units < -32768] = -32768
        return voltage_machine_units.astype("<i2").tobytes()

    def close(self):
        self.s.close()
