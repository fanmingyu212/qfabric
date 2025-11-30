# This driver requires the `telnetlib3` package.
import io
import os
from ftplib import FTP

import numpy as np
import numpy.typing as npt
from telnetlib3 import Telnet

from qfabric.programmer.driver.awg710.commands import AWG710Commands
from qfabric.programmer.driver.awg710.data import PatternData, SequenceData

PORT = 4000


class AWG710Driver:
    """
    Telnet / FTP-based driver for the Tektronix AWG710 arbitrary waveform generator.

    Only implements features related to the sequence / waveform mode.
    Does not support using the AWG as a function generator.

    Always fix the output amplitude to 1 V and offset to 0 V.

    Assumes that the device is connected to the same network as the control computer
    and FTP is enabled on the device.

    This driver only uses the main internal storage of the AWG.
    """

    def __init__(
        self,
        address: str,
        principal_card: bool = False,
        external_reference: bool = False,
        waveform_sample_freq: float = 4e9,
    ):
        self._address = address
        self._principal_card = principal_card
        self._control = Telnet(address, PORT)
        self._ftp = FTP(address)
        self._ftp.login()

        self._commands = AWG710Commands()

        self._startup(not external_reference, self._principal_card, waveform_sample_freq)

    def _send_or_query(self, commands: str | list[str]) -> list[str]:
        if not isinstance(commands, list):
            commands = [commands]
        command_str = ";".join(commands)
        self._control.write(command_str)
        query_indices = [commands.index(command) for command in commands if command.endswith("?")]
        if len(query_indices) > 0:
            response = self._control.read_all()
            messages = response.split(";")
            if len(messages) != len(query_indices):
                raise RuntimeError(
                    f"Got {len(messages)} of responses "
                    f"while expecting {len(query_indices)} responses "
                    f"for command {command_str}"
                )
            return messages
        else:
            return []

    def _startup(
        self, internal_reference: bool, internal_trigger: bool, waveform_sample_frequency: float
    ):
        if internal_trigger:
            trigger_source = "INTERNEL"
        else:
            trigger_source = "EXTERNAL"
        if internal_reference:
            reference_source = "INTERNEL"
        else:
            reference_source = "EXTERNAL"
        self._send_or_query(
            [
                self._commands.set_run_mode("ENH"),
                self._commands.stop_output(),
                self._commands.select_default_storage(),
                self._commands.set_output_amplitude(2),
                self._commands.set_output_offset(0),
                self._commands.set_trigger_source(trigger_source),
                self._commands.set_oscillator_reference(reference_source),
                self._commands.set_waveform_sample_frequency(waveform_sample_frequency),
                self._commands.set_output_state(True),
            ]
        )

    def set_trigger_source(self, external: bool):
        if external:
            trigger_source = "EXTERNAL"
        else:
            trigger_source = "INTERNEL"
        self._send_or_query(self._commands.set_trigger_source(trigger_source))

    def trigger_now(self):
        self._send_or_query(self._commands.trigger_immediate())

    def get_run_state(self) -> int:
        """
        Gets the current run state.

        Returns:
            int: 0 - stopped, 1 - waiting for trigger, 2 - running.
        """
        val = self._send_or_query(self._commands.get_run_state())
        return int(val)

    def start(self):
        self._send_or_query(self._commands.start_output())

    def stop(self):
        self._send_or_query(self._commands.stop_output())

    def cd_telnet(self, path: str):
        self._send_or_query(self._commands.cd(path))

    def set_waveform_file(self, file_name: str):
        self._send_or_query(self._commands.set_waveform_file(file_name))

    def mkdir(self, directory: str):
        self._ftp.mkd(directory)

    def rmdir(self, directory: str):
        self._ftp.rmd(directory)

    def delete(self, file_name: str):
        self._ftp.delete(file_name)

    def cd_ftp(self, path: str):
        self._ftp.cwd(path)

    def _create_file(self, file_name: str, data: bytes):
        self._ftp.storbinary(f"STOR {file_name}", io.BytesIO(data))

    def create_pattern_file(
        self,
        file_name: str,
        analog_data: npt.NDArray[np.float64],
        digital_1: npt.NDArray[np.bool],
        digital_2: npt.NDArray[np.bool],
        sample_rate: float,
    ):
        pattern = PatternData(analog_data, digital_1, digital_2, sample_rate)
        self._create_file(file_name, pattern.data)

    def create_sequence_file(self, file_name: str, step_info: list[tuple[str, int, int]]):
        lines = []
        for name, repeat, next_step in step_info:
            lines.append(name, repeat, False, next_step)
        sequence = SequenceData(lines)
        self._create_file(file_name, sequence.data)

    def _remove_ftp_dir_recursive(self, path: str):
        for name, properties in self._ftp.mlsd(path=path):
            if name in (".", ".."):
                continue

            full_path = os.path.join(path, name).replace("\\", "/")

            if properties.get("type") == "file":
                self._ftp.delete(full_path)
            elif properties.get("type") == "dir":
                self._remove_ftp_dir_recursive(full_path)
        self._ftp.rmd(path)

    def remove_all_files_in_directory(self, directory: str = "/", skip_ask: bool = False):
        if directory == "/":
            if not skip_ask:
                input_val = input("Confirm to delete all data? (y/N)")
                if input_val != "y" and input_val != "Y":
                    print("File deletion cancelled.")
                    return
        self._remove_ftp_dir_recursive(directory)
