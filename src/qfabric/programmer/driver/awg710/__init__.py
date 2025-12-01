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
        principal_card: bool,
        ftp_user: str,
        ftp_password: str,
        external_reference: bool = False,
        waveform_sample_freq: float = 4e9,
    ):
        self._address = address
        self._principal_card = principal_card
        self._control = Telnet(address, PORT)
        self._ftp = FTP(address)
        self._ftp.login(user=ftp_user, passwd=ftp_password)

        self._commands = AWG710Commands()

        self._startup(not external_reference, waveform_sample_freq)

    def _send_or_query(self, commands: str | list[str]) -> list[str]:
        if not isinstance(commands, list):
            commands = [commands]
        command_str = ";".join(commands) + "\n"
        self._control.write(command_str.encode("ascii"))
        query_indices = [commands.index(command) for command in commands if command.endswith("?")]
        response = self._control.read_until(b"\n", timeout=1)
        messages = response.decode("ascii").split(";")
        if len(messages) != len(query_indices) and messages != [""]:
            raise RuntimeError(
                f"Got {len(messages)} of responses "
                f"while expecting {len(query_indices)} responses "
                f"for command {command_str}"
            )
        
        self._get_next_error()
        return messages

    def _get_next_error(self):
        self._control.write("*ESR?\n".encode("ascii"))
        self._control.read_until(b"\n", timeout=1)
        self._control.write(
            (self._commands.get_next_error() + "\n").encode("ascii")
        )
        result = self._control.read_until(b"\n", timeout=1).decode("ascii")
        if int(result.split(",")[0]) != 0:
            raise RuntimeError(f"Failed with error: {result}")

    def _startup(
        self, internal_reference: bool, waveform_sample_frequency: float
    ):
        self._send_or_query("*CLS")
        if internal_reference:
            reference_source = "INTERNAL"
        else:
            reference_source = "EXTERNAL"
        self._send_or_query(
            [
                self._commands.set_run_mode("ENH"),
                self._commands.stop(),
                self._commands.select_default_storage(),
                self._commands.set_output_amplitude(2),
                self._commands.set_output_offset(0),
                self._commands.set_trigger_source("EXTERNAL"),
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
        return int(val[0])

    def start(self):
        self._send_or_query(self._commands.start())

    def stop(self):
        self._send_or_query(self._commands.stop())

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
        trigger = True
        for name, repeat, next_step in step_info:
            lines.append((name, repeat, trigger, next_step))
            trigger = False
        sequence = SequenceData(lines)
        self._create_file(file_name, sequence.data)

    def remove_all_files_in_directory(self, directory: str = "/", skip_ask: bool = False):
        if directory == "/":
            if not skip_ask:
                input_val = input("Confirm to delete all data? (y/N)")
                if input_val != "y" and input_val != "Y":
                    print("File deletion cancelled.")
                    return
        self._remove_ftp_dir_recursive(directory)

    def _remove_ftp_dir_recursive(self, path: str):
        local_dirs: list[str] = []
        local_files: list[str] = []
        def worker(line):
            is_directory = line[0] == "d"
            filename = line.split(" ")[-1]
            if filename in (".", ".."):
                return
            full_path = os.path.join(path, filename).replace("\\", "/")
            if is_directory:
                local_dirs.append(full_path)
            else:
                local_files.append(full_path)
        
        self._ftp.dir(path, worker)
        for filename in local_files:
            self.delete(filename)
        for dir in local_dirs:
            self._remove_ftp_dir_recursive(dir)
        if path != "/":
            self.rmdir(path)
