from typing import Literal

import numpy as np


class AWG710Commands:
    def __init__(self): ...

    def device_identification(self) -> str:
        return "*IDN?"

    def get_jump_mode_sequence_enhanced(self) -> str:
        return ":AWGC:ENH:SEQ?"

    def set_jump_mode_sequence_enhanced(self, mode: Literal["LOGIC", "TABLE", "SOFTWARE"]) -> str:
        return f":AWGC:ENH:SEQ {mode}"

    def generate_trigger_event_logic_jump(self) -> str:
        return ":AWGC:EVEN"

    def generate_trigger_event_software_jump(self) -> str:
        return ":AWGC:EVEN:SOFT"

    def generate_trigger_event_table_jump(self) -> str:
        return ":AWGC:EVEN:TABL"

    def get_func_gen_frequency(self) -> str:
        return ":AWGC:FG:FREQ?"

    def set_func_gen_frequency(self, frequency: float) -> str:
        if frequency < 1 or frequency > 400e6:
            raise ValueError("Frequency must be between 1 Hz to 400 MHz.")
        return f":AWGC:FG:FREQ {frequency}Hz"

    def get_func_gen_function(self) -> str:
        return ":AWGC:FG:FUNC?"

    def set_func_gen_function(
        self, shape: Literal["SIN", "TRI", "SQU", "RAMP", "PULS", "DC"]
    ) -> str:
        return f":AWGC:FG:FUNC {shape}"

    def get_func_gen_polarity(self) -> str:
        return ":AWGC:FG:POL?"

    def set_func_gen_polarity(self, polarity: Literal["POSITIVE", "NEGATIVE"]) -> str:
        return f":AWGC:FG:POL {polarity}"

    def get_func_gen_duty_cycle(self) -> str:
        return ":AWGC:FG:PULS:DCYC?"

    def set_func_gen_duty_cycle(self, duty_cycle: float) -> str:
        if duty_cycle < 0 or duty_cycle > 1:
            raise ValueError("Duty cycle must be between 0 to 1.")
        duty_cycle_percent = np.round(duty_cycle * 100, 1)
        return f":AWGC:FG:PULS:DCYC {duty_cycle_percent}"

    def get_func_gen_mode(self) -> str:
        return ":AWGC:FG?"

    def set_func_gen_mode(self, state: bool) -> str:
        if state:
            return ":AWGC:FG ON"
        else:
            return ":AWGC:FG OFF"

    def get_func_gen_amplitude(self) -> str:
        return ":AWGC:FG:VOLT?"

    def set_func_gen_amplitude(self, amplitude_Vpp: float) -> str:
        if amplitude_Vpp < 0.02 or amplitude_Vpp > 2:
            raise ValueError("Amplitude must be between 20 mVpp to 2 Vpp.")
        return f":AWGC:FG:VOLT {amplitude_Vpp}"

    def get_func_gen_offset(self) -> str:
        return ":AWGC:FG:VOLT:OFFS?"

    def set_func_gen_offset(self, offset: float) -> str:
        if offset < -0.5 or offset > 0.5:
            raise ValueError("Offset must be between -0.5 V to 0.5 V.")
        return f":AWGC:FG:VOLT:OFFS {offset}"

    def get_waveform_mix_mode(self) -> str:
        return ":AWGC:MIX?"

    def set_waveform_mix_mode(self, state: bool) -> str:
        if state:
            return ":AWGC:MIX ON"
        else:
            return ":AWGC:MIX OFF"

    def get_run_mode(self) -> str:
        return ":AWGC:RMOD?"

    def set_run_mode(self, mode: Literal["CONT", "TRIG", "GAT", "ENH"]) -> str:
        return f":AWGC:RMOD {mode}"

    def get_run_state(self) -> str:
        return ":AWGC:RST?"

    def start_output(self) -> str:
        return ":AWGC:RUN"

    def stop_output(self) -> str:
        return ":AWGC:STOP"

    def restore_settings(
        self, file_name: str, storage: Literal["MAIN", "FLOP", "NET1", "NET2", "NET3"] = "MAIN"
    ) -> str:
        return f':AWGC:SRES "{file_name}","{storage}"'

    def save_settings(
        self, file_name: str, storage: Literal["MAIN", "FLOP", "NET1", "NET2", "NET3"] = "MAIN"
    ) -> str:
        return f':AWGC:SSAV "{file_name}","{storage}"'

    def clear_event_registers_and_queues(self) -> str:
        return "*CLS"

    def set_display_state(self, state: bool) -> str:
        if state:
            return ":DISP:ENAB ON"
        else:
            return ":DISP:ENAB OFF"

    def set_display_hilight_color(self, color: Literal[0, 1, 2, 3, 4, 5, 6, 7]) -> str:
        return f":DISP:HIL:COL {color}"

    def catalog(self, storage: Literal["MAIN", "FLOP", "NET1", "NET2", "NET3"] = "MAIN") -> str:
        return f':MMEM:CAT? "{storage}"'

    def cd(self, directory: str) -> str:
        return f':MMEM:CDIR "{directory}"'

    def copy_file(self, file_source: str, file_destination: str) -> str:
        return f':MMEM:COPY "{file_source}","{file_destination}"'

    def get_file_data(self, file_name: str) -> str:
        return f':MMEM:DATA "{file_name}"?'

    def set_file_data(self, file_name: str, data: str) -> str:
        return f':MMEM:DATA "{file_name}",{data}'

    def delete_file(self, file_name: str) -> str:
        return f':MMEM:DEL "{file_name}"?'

    def mkdir(self, directory: str) -> str:
        return f':MMEM:MDIR "{directory}"'

    def move_file(self, file_source: str, file_destination: str) -> str:
        return f':MMEM:MOVE "{file_source}","{file_destination}"'

    def select_default_storage(
        self, storage: Literal["MAIN", "FLOP", "NET1", "NET2", "NET3"] = "MAIN"
    ) -> str:
        return f':MMEM:MSIS "{storage}"'

    def get_output_filter_low_pass_frequency(self) -> str:
        return ":OUTP:FILT:FREQ?"

    def set_output_filter_low_pass_frequency(
        self, value: Literal["20MHz", "50MHz", "100MHz", "200MHz", "INFINITY"]
    ) -> str:
        return f":OUTP:FILT:FREQ {value}"

    def get_inverted_output_state(self) -> str:
        return f":OUTP:IST?"

    def set_inverted_output_state(self, state: bool) -> str:
        if state:
            return f":OUTP:IST ON"
        else:
            return f":OUTP:IST OFF"

    def get_output_state(self) -> str:
        return f":OUTP:STAT?"

    def set_output_state(self, state: bool) -> str:
        if state:
            return f":OUTP:STAT ON"
        else:
            return f":OUTP:STAT OFF"

    def get_waveform_sample_frequency(self) -> str:
        return ":SOUR:FREQ?"

    def set_waveform_sample_frequency(self, frequency: float) -> str:
        if frequency < 50e3 or frequency > 4e9:
            raise ValueError("The sample frequency must be between 50 kHz to 4 GHz.")
        return f":SOUR:FREQ {frequency}Hz"

    def get_waveform_file(self) -> str:
        return ":SOUR:FUNC:USER?"

    def set_waveform_file(
        self, file_name: str, storage: Literal["MAIN", "FLOP", "NET1", "NET2", "NET3"] = "MAIN"
    ) -> str:
        return f':SOUR:FUNC:USER "{file_name}","{storage}"'

    def get_marker_high_voltage(self, channel: Literal[1, 2]) -> str:
        return f":SOUR:MARK{channel}:VOLT:HIGH?"

    def set_marker_high_voltage(self, channel: Literal[1, 2], voltage: float) -> str:
        if voltage < -1.1 or voltage > 3.0:
            raise ValueError("Voltage must be between -1.1 V to 3.0 V.")
        return f":SOUR:MARK{channel}:VOLT:HIGH {voltage}"

    def get_marker_low_voltage(self, channel: Literal[1, 2]) -> str:
        return f":SOUR:MARK{channel}:VOLT:LOW?"

    def set_marker_low_voltage(self, channel: Literal[1, 2], voltage: float) -> str:
        if voltage < -1.1 or voltage > 3.0:
            raise ValueError("Voltage must be between -1.1 V to 3.0 V.")
        return f":SOUR:MARK{channel}:VOLT:LOW {voltage}"

    def get_oscillator_reference(self) -> str:
        return f":SOUR:ROSC:SOUR?"

    def set_oscillator_reference(self, reference: Literal["INTERNAL", "EXTERNAL"]) -> str:
        return f":SOUR:ROSC:SOUR {reference}"

    def get_output_amplitude(self) -> str:
        return ":SOUR:VOLT:AMPL?"

    def set_output_amplitude(self, amplitude_Vpp: float) -> str:
        if amplitude_Vpp < 0.02 or amplitude_Vpp > 2:
            raise ValueError("Amplitude must be between 20 mVpp to 2 Vpp.")
        return f":SOUR:VOLT:AMPL {amplitude_Vpp}"

    def get_output_offset(self) -> str:
        return ":SOUR:VOLT:OFFS?"

    def set_output_offset(self, offset: float) -> str:
        if offset < -0.5 or offset > 0.5:
            raise ValueError("Offset must be between -0.5 V to 0.5 V.")
        return f":SOUR:VOLT:OFFS {offset}"

    def trigger_immediate(self) -> str:
        return ":TRIG"

    def get_trigger_impedance(self) -> str:
        return ":TRIG:IMP?"

    def set_trigger_impedance(self, impedance: Literal[50, 1000]) -> str:
        return f":TRIG:IMP {impedance}"

    def get_trigger_level(self) -> str:
        return ":TRIG:LEV?"

    def set_trigger_level(self, level: float) -> str:
        if level < -5 or level > 5:
            raise ValueError("Trigger level must be between -5 V to 5 V.")
        return f":TRIG:LEV {level}"

    def get_trigger_polarity(self) -> str:
        return ":TRIG:POL?"

    def set_trigger_polarity(self, polarity: Literal["POSITIVE", "NEGATIVE"]) -> str:
        return f":TRIG:POL {polarity}"

    def get_trigger_slope(self) -> str:
        return ":TRIG:SLOP?"

    def set_trigger_slope(self, slope: Literal["POSITIVE", "NEGATIVE"]) -> str:
        return f":TRIG:SLOP {slope}"

    def get_trigger_source(self) -> str:
        return ":TRIG:SOUR?"

    def set_trigger_source(self, source: Literal["INTERNAL", "EXTERNAL"]) -> str:
        return f":TRIG:SOUR {source}"

    def get_internal_trigger_period(self) -> str:
        return ":TRIG:TIM?"

    def set_internal_trigger_period(self, period: float) -> str:
        if period < 1e-3 or period > 10:
            raise ValueError("Trigger period must be between 1 ms to 10 s.")
        return f":TRIG:TIM {period}"
