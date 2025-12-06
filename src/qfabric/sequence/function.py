from abc import abstractmethod
from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import numpy.typing as npt

from qfabric._util import dynamic_import


@dataclass
class Function:
    """
    Base class for all analog or digital functions.

    All subclasses are automatically decorated by :deco:`dataclass`. All attributes of these
    subclasses that affect the output of the function should be defined as `field`
    (annotated instance variables), see below for an example.

    .. code-block:: python

        class FunctionWithFrequency(AnalogFunction):
            frequency: float

            def __init__(self, frequency: float):
                self.frequency = frequency

    Without defining these attributes as fields, the AWG memory loaded may not be correct.
    If cache / state attributes are needed (that do not affect the output of the function),
    these attributes can be defined as fields with `compare=False`, see below.

    .. code-block:: python

        from dataclasses import field

        class FunctionWithTimings(DigitalFunction):
            time_1: float
            time_2: float
            time_3: float
            _total_time: float = field(compare=False)  # cache. Does not affect output.

            def __init__(self, time_1: float, time_2: float, time_3: float):
                self.time_1 = time_1
                self.time_2 = time_2
                self.time_3 = time_3
                # adds times and save in a field without comparison.
                self._total_time = time_1 + time_2 + time_3

            @property
            def min_duration(self) -> float:
                # does not need to add up times here.
                return self._total_time

    All subclasses used as actual functions should override :meth:`min_duration`.
    """

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        dataclass(cls, init=False)

    def __setattr__(self, name: str, value: Any):
        # Allow dataclass internals
        if name.startswith("__"):
            return super().__setattr__(name, value)

        # During __init__, dataclass may set attributes before fields() is fully ready,
        # so catch exceptions and just set.
        try:
            field_names = {f.name for f in fields(self)}
        except TypeError:
            return super().__setattr__(name, value)

        if name not in field_names:
            raise AttributeError(
                f"{type(self).__name__}: '{name}' is not a declared field. "
                "Declare it with as a field (with type annotation) if it affects the output. "
                "Define it as a field with `compare=False` if it does not affect the output."
            )

        return super().__setattr__(name, value)

    @property
    @abstractmethod
    def min_duration(self) -> float:
        """Minimum duration of this function."""

    def to_dict(self) -> dict[str, Any]:
        """
        Dict representation of the function without guarantee of reproduction.

        This can be serialized to JSON for saving.
        If the class definition is changed or removed, the output of this function
        may not be reproduced after saving and loading.

        Returns:
            dict[str, Any]: dict representation of the step.
        """
        value = {}
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        value["fields"] = {}
        for field in fields(self):
            value["fields"][field.name] = getattr(self, field.name)
        return value


class AnalogFunction(Function):
    """
    Base class for all analog functions.

    All subclasses should override :meth:`output`.
    """

    @abstractmethod
    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        """
        Outputs of the analog function.

        Args:
            times (npt.NDArray[np.float64]): Times to evaluate the function at.
            time_offset (float):
                Time offset to keep phase coherence condition for :class:`AnalogSequence`.
                For an oscillating signal, the phase calculation should use timestamps
                as `times + time_offset`. See :class:`~qfabric.sequence.basic_functions.SineWave`
                as an example.

        Returns:
            npt.NDArray[np.float64]: output.
        """


class AnalogEmpty(AnalogFunction):
    """
    Analog function with zero output.
    """

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        return np.zeros(len(times), dtype=np.float64)

    @property
    def min_duration(self):
        return 0


class DigitalFunction(Function):
    """
    Base class for all digital functions.

    All subclasses should override :meth:`output`.
    """

    @abstractmethod
    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.bool]:
        """
        Outputs of the digital function.

        Args:
            times (npt.NDArray[np.float64]): Times to evaluate the function at.

        Returns:
            npt.NDArray[np.bool]: output.
        """


class DigitalEmpty(DigitalFunction):
    """
    Digital function with zero output.
    """

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.bool]:
        return np.zeros(len(times), dtype=np.bool)

    @property
    def min_duration(self):
        return 0


class AnalogSum(AnalogFunction):
    """
    Sum of multiple analog functions.

    Args:
        function_1 (AnalogFunction): Function 1 to be summed.
        *functions (AnalogFunction): Other functions to be summed.
    """

    functions: list[AnalogFunction]

    def __init__(
        self,
        function_1: AnalogFunction,
        *functions: AnalogFunction,
    ):
        self.functions: list[AnalogFunction] = [function_1] + list(functions)

    @property
    def min_duration(self) -> float:
        value = 0
        for function in self.functions:
            if value < function.min_duration:
                value = function.min_duration
        return value

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        result = np.zeros(len(times), dtype=float)
        for function in self.functions:
            result += function.output(times, time_offset)
        return result

    def to_dict(self) -> dict[str, Any]:
        value = {}
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        value["fields"] = {}
        for field in fields(self):
            if field.name != "functions":
                value["fields"][field.name] = getattr(self, field.name)
            else:
                value["fields"]["functions"] = []
                for function in self.functions:
                    value["fields"]["functions"].append(function.to_dict())
        return value


class AnalogProduct(AnalogFunction):
    """
    Product of multiple analog functions.

    Args:
        function_1 (AnalogFunction): Function 1 to be multiplied.
        *functions (AnalogFunction): Other functions to be multiplied.
    """

    functions: list[AnalogFunction]

    def __init__(
        self,
        function_1: AnalogFunction,
        *functions: AnalogFunction,
    ):
        self.functions: list[AnalogFunction] = [function_1] + list(functions)

    @property
    def min_duration(self) -> float:
        value = 0
        for function in self.functions:
            if value < function.min_duration:
                value = function.min_duration
        return value

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        result = np.ones(len(times), dtype=float)
        for function in self.functions:
            result *= function.output(times, time_offset)
        return result

    def to_dict(self) -> dict[str, Any]:
        value = {}
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        value["fields"] = {}
        for field in fields(self):
            if field.name != "functions":
                value["fields"][field.name] = getattr(self, field.name)
            else:
                value["fields"]["functions"] = []
                for function in self.functions:
                    value["fields"]["functions"].append(function.to_dict())
        return value


class AnalogSequence(AnalogFunction):
    """
    Sequence of analog functions.

    This is useful to build longer steps containing multiple analog pulses on the same channel,
    with optional phase coherence between the pulses.
    """

    functions: list[AnalogFunction]
    start_times: list[float]
    durations: list[float]
    use_coherent_phases: list[bool]

    def __init__(self):
        self.functions: list[AnalogFunction] = []
        self.start_times: list[float] = []
        self.durations: list[float] = []
        self.use_coherent_phases: list[bool] = []

    def add_function(
        self,
        function: AnalogFunction,
        delay_time_after_previous: float = 0,
        duration: float = None,
        coherent_phase: bool = False,
    ):
        """
        Appends an analog function.

        Args:
            function (AnalogFunction): Analog function to be appended.
            delay_time_after_previous (float): Delay time after the previous function. Default 0.
            duration (float):
                Duration of this function. If None, the minimum function duration is used.
            coherent_phase (bool):
                If True, the phase of the function is calculated relative to the start of this
                function sequence. Otherwise, the phase of the function is calculated relative to
                the start of this appended function.
        """
        if delay_time_after_previous < 0:
            raise ValueError("Delay time after the previous function cannot be less than 0.")
        start_time = self.min_duration + delay_time_after_previous
        if duration is None:
            duration = function.min_duration
            if np.isclose(duration, 0, atol=1e-12):
                raise ValueError(
                    f"Duration must be defined as the function {function} does not have a well-defined duration."
                )
        elif duration < 0:
            raise ValueError("Duration cannot be less than 0 s.")

        self.functions.append(function)
        self.start_times.append(start_time)
        self.durations.append(duration)
        self.use_coherent_phases.append(coherent_phase)

    @property
    def min_duration(self) -> float:
        value = 0
        if len(self.start_times) > 0:
            value = self.start_times[-1] + self.durations[-1]
        return value

    def output(
        self, times: npt.NDArray[np.float64], time_offset: float = 0
    ) -> npt.NDArray[np.float64]:
        outputs = np.zeros(len(times), dtype=float)
        for kk in range(len(self.functions)):
            function = self.functions[kk].output
            start_time = self.start_times[kk]
            stop_time = start_time + self.durations[kk]
            func_mask = (times >= start_time) & (times < stop_time)
            if self.use_coherent_phases[kk]:
                time_offset_this_pulse = time_offset + start_time
            else:
                time_offset_this_pulse = 0
            outputs[func_mask] = function(times[func_mask] - start_time, time_offset_this_pulse)
        return outputs

    def to_dict(self) -> dict[str, Any]:
        value = {}
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        value["fields"] = {}
        for field in fields(self):
            if field.name != "functions":
                value["fields"][field.name] = getattr(self, field.name)
            else:
                value["fields"]["functions"] = []
                for function in self.functions:
                    value["fields"]["functions"].append(function.to_dict())
        return value


class DigitalSequence(DigitalFunction):
    """
    Sequence of digital functions.

    This is useful to build longer steps containing multiple digital pulses on the same channel.
    """

    default_on: bool
    functions: list[AnalogFunction]
    start_times: list[float]
    durations: list[float]

    def __init__(self, default_on: bool = False):
        self.default_on: bool = default_on
        self.functions: list[DigitalFunction] = []
        self.start_times: list[float] = []
        self.durations: list[float] = []

    def add_function(
        self,
        function: DigitalFunction,
        delay_time_after_previous: float = 0,
        duration: float = None,
    ):
        """
        Appends a digital function.

        Args:
            function (DigitalFunction): Digital function to be appended.
            delay_time_after_previous (float): Delay time after the previous function. Default 0.
            duration (float):
                Duration of this function. If None, the minimum function duration is used.
        """
        if delay_time_after_previous < 0:
            raise ValueError("Delay time after the previous function cannot be less than 0 s.")
        start_time = self.min_duration + delay_time_after_previous
        if duration is None:
            duration = function.min_duration
            if np.isclose(duration, 0, atol=1e-12):
                raise ValueError(
                    f"Duration must be defined as the function {function} does not have a well-defined duration."
                )
        elif duration < 0:
            raise ValueError("Duration cannot be less than 0 s.")
        self.functions.append(function)
        self.start_times.append(start_time)
        self.durations.append(duration)

    @property
    def min_duration(self) -> float:
        value = 0
        if len(self.start_times) > 0:
            value = self.start_times[-1] + self.durations[-1]
        return value

    def output(self, times: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
        outputs = np.zeros(len(times), dtype=bool)
        for kk in range(len(self.functions)):
            function = self.functions[kk].output
            start_time = self.start_times[kk]
            stop_time = start_time + self.durations[kk]
            func_mask = (times >= start_time) & (times < stop_time)
            outputs[func_mask] = function(times[func_mask] - start_time)
        return outputs

    def to_dict(self) -> dict[str, Any]:
        value = {}
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        value["fields"] = {}
        for field in fields(self):
            if field.name != "functions":
                value["fields"][field.name] = getattr(self, field.name)
            else:
                value["fields"]["functions"] = []
                for function in self.functions:
                    value["fields"]["functions"].append(function.to_dict())
        return value
