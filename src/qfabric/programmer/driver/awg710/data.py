import numpy as np
import numpy.typing as npt


class PatternData:
    def __init__(
        self,
        analog_data: npt.NDArray[np.float64],
        digital_1: npt.NDArray[np.bool],
        digital_2: npt.NDArray[np.bool],
        sample_rate: float,
        amplitude: float = 1,  # default full-range 2 Vpp
    ):
        if not (len(analog_data) == len(digital_1) == len(digital_2)):
            raise ValueError("analog_data, digital_1, and digital_2 must have the same length.")

        header = "MAGIC 2000\r\n"
        num_bytes = str(len(analog_data) * 2)
        num_digits = str(len(num_bytes))
        self.data = (header + "#" + num_digits + num_bytes).encode("ascii")

        analog_8bit = np.clip(np.round((analog_data + amplitude) / (2 * amplitude) * 255), 0, 255).astype(np.uint16)
        words = analog_8bit << 2
        words |= (digital_1 & 1).astype(np.uint16) << 13
        words |= (digital_2 & 1).astype(np.uint16) << 14
        self.data += np.asarray(words, dtype="<u2").tobytes()

        sample_rate = str(int(sample_rate))
        self.data += f"CLOCK {sample_rate}\r\n".encode("ascii")


class SequenceData:
    """
    Builds the content of a sequence definition file.

    Does not support logic or table jumps. Does not support the strobe signal.

    Args:
        lines (list[tuple[str, int, bool, int]]):
            (file_name, repeats, wait_for_trigger, next_line_number).
    """

    def __init__(self, lines: list[tuple[str, int, bool, int]]):
        data = "MAGIC 3002A\r\n"
        num_lines = len(lines)
        data += f"LINES {num_lines}\r\n"
        for file_name, repeats, wait_for_trigger, next_line_number in lines:
            data += f'"{file_name}","",{repeats},{int(wait_for_trigger)},0,0,{next_line_number}\r\n'
        self.data = data.encode("ascii")
