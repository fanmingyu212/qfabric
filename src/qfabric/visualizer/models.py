from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from qfabric.sequence.function import AnalogFunction, DigitalFunction, Function
from qfabric.sequence.sequence import Sequence
from qfabric.sequence.step import Step


def _value_to_str_with_prefix(value: float) -> str:
    if isinstance(value, bool):
        return str(value)
    if not np.isscalar(value):
        return str(value)
    if value == 0:
        return "0"
    value_sign = value / abs(value)
    value = abs(value)
    if value < 1e-7:
        text = f"{value * 1e9}n"
    elif value < 1e-4:
        text = f"{value * 1e6}µ"
    elif value < 1e-1:
        text = f"{value * 1e3}m"
    elif value < 1e2:
        text = f"{value}"
    elif value < 1e5:
        text = f"{value / 1e3}k"
    elif value < 1e8:
        text = f"{value / 1e6}M"
    elif value < 1e11:
        text = f"{value / 1e9}G"
    else:
        text = f"{value / 1e12}T"
    if value_sign < 0:
        text = f"-{text}"
    return text


@dataclass
class FunctionBlockModel:
    func: Function
    x_index: int
    y_index: int
    y_label: str
    start_time: float
    duration: float
    repeat: int

    def __init__(
        self,
        function: Function,
        x_index: int,
        y_index: int,
        y_label: str,
        start_time: float,
        duration: float,
        repeat: int,
    ):
        self.func = function
        self.x_index = x_index
        self.y_index = y_index
        self.y_label = y_label
        self.start_time = start_time
        self.duration = duration
        self.repeat = repeat

    @property
    def parameters(self) -> dict[str, Any]:
        func_fields = fields(self.func)
        params = {"pulse start time": self.start_time}
        for field in func_fields:
            if field.compare:
                params[field.name] = getattr(self.func, field.name)
        return params

    @property
    def name(self) -> str:
        return self.func.__class__.__name__

    @property
    def parameters_description(self) -> str:
        description = ""
        for name in self.parameters:
            value = self.parameters[name]
            description += f"{name} = {_value_to_str_with_prefix(value)}<br>"
        return description


@dataclass
class AnalogFunctionBlockModel(FunctionBlockModel):
    func: AnalogFunction
    x_index: int
    y_index: int
    y_label: str
    start_time: float
    duration: float
    repeat: int


@dataclass
class DigitalFunctionBlockModel(FunctionBlockModel):
    func: DigitalFunction
    x_index: int
    y_index: int
    y_label: str
    start_time: float
    duration: float
    repeat: int


@dataclass
class StepModel:
    name: str
    index: int
    start_time: float
    duration: float
    repeat: int
    analog_functions: dict[int, AnalogFunctionBlockModel]
    digital_functions: dict[int, DigitalFunctionBlockModel]

    def __init__(
        self,
        step: Step,
        index: int,
        start_time: float,
        repeat: int,
        analog_chs: list[int],
        digital_chs: list[int],
    ):
        self.name = step.name
        self.index = index
        self.start_time = start_time
        self.duration = step.duration
        self.repeat = repeat

        self.analog_functions = {}
        for analog_ch, analog_function in step.analog_functions.items():
            y_index = -analog_chs.index(analog_ch)
            y_label = f"Analog CH{analog_ch}"
            self.analog_functions[analog_ch] = AnalogFunctionBlockModel(
                analog_function,
                self.index,
                y_index,
                y_label,
                self.start_time,
                self.duration,
                self.repeat,
            )

        self.digital_functions = {}
        for digital_ch, digital_function in step.digital_functions.items():
            y_index = -digital_chs.index(digital_ch) - len(analog_chs)
            y_label = f"Digital CH{digital_ch}"
            self.digital_functions[digital_ch] = DigitalFunctionBlockModel(
                digital_function,
                self.index,
                y_index,
                y_label,
                self.start_time,
                self.duration,
                self.repeat,
            )


@dataclass
class SequenceModel:
    steps: list[StepModel]
    channel_names: list[str]

    def __init__(self, sequence: Sequence):
        steps = sequence.get_steps()
        repeats = sequence.get_repeats()

        analog_chs = set()
        digital_chs = set()
        for step in steps:
            analog_chs = analog_chs | set(step.analog_functions)
            digital_chs = digital_chs | set(step.digital_functions)
        analog_chs = sorted(list(analog_chs))
        digital_chs = sorted(list(digital_chs))
        self.channel_names: list[str] = []
        for analog_ch in analog_chs:
            self.channel_names.append(f"Analog CH{analog_ch}")
        for digital_ch in digital_chs:
            self.channel_names.append(f"Digital CH{digital_ch}")

        self.steps: list[StepModel] = []
        start_time = 0
        for index in range(len(steps)):
            self.steps.append(
                StepModel(steps[index], index, start_time, repeats[index], analog_chs, digital_chs)
            )
            start_time += steps[index].duration * repeats[index]

    @property
    def functions(self) -> list[FunctionBlockModel]:
        values: list[FunctionBlockModel] = []
        for step in self.steps:
            values.extend(step.analog_functions.values())
            values.extend(step.digital_functions.values())
        return values

    @property
    def step_labels(self) -> list[str]:
        values: list[str] = []
        for step in self.steps:
            name = step.name
            duration = _value_to_str_with_prefix(step.duration) + "s"
            label = name + "\n"
            if step.repeat > 1:
                total_duration = _value_to_str_with_prefix(step.duration * step.repeat) + "s"
                label += f"{duration} \u00d7 {step.repeat}\n({total_duration})"
            else:
                label += duration
            values.append(label)
        return values
