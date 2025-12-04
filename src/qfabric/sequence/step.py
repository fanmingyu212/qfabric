from dataclasses import dataclass
from typing import Any

from qfabric.sequence.basic_functions import DigitalOn
from qfabric.sequence.function import AnalogFunction, DigitalFunction


class Step:
    """
    Step in a pulse sequence.

    Step is the building block for a sequence, often consists one (or a few) logically-related
    pulses in a sequence. It is designed so that each step is replayed on the AWG exactly
    as defined with phase coherence. Due to AWG segment size limits, the actual duration of a
    step is often slightly longer than the defined duration. Therefore pulses spanning different
    steps may not have exact phase relations as defined.

    Args:
        name (str): Step name.
    """

    def __init__(self, name: str):
        self.name = name
        self.analog_functions: dict[int, AnalogFunction] = {}
        self.digital_functions: dict[int, DigitalFunction] = {}
        self._duration: float = None

    def __eq__(self, other: "Step") -> bool:
        if self.analog_functions != other.analog_functions:
            return False
        if self.digital_functions != other.digital_functions:
            return False
        if self.duration != other.duration:
            return False
        return True

    @property
    def duration(self) -> float:
        """
        Nominal duration of this step.

        If :attr:`duration` is not set, uses the longest function defined on all channels.
        """
        if self._duration is None:
            min_duration = 0
            for channel in self.analog_functions:
                if min_duration < self.analog_functions[channel].min_duration:
                    min_duration = self.analog_functions[channel].min_duration
            for channel in self.digital_functions:
                if min_duration < self.digital_functions[channel].min_duration:
                    min_duration = self.digital_functions[channel].min_duration
            return min_duration
        else:
            return self._duration

    @duration.setter
    def duration(self, value: float):
        for channel in self.analog_functions:
            if value < self.analog_functions[channel].min_duration:
                raise ValueError(
                    f"Step duration of {value} is shorter than the minimum "
                    f"duration of the function {self.analog_functions[channel]}."
                )
        for channel in self.digital_functions:
            if value < self.digital_functions[channel].min_duration:
                raise ValueError(
                    f"Step duration of {value} is shorter than the minimum "
                    f"duration of the function {self.digital_functions[channel]}."
                )
        self._duration = value

    def add_analog_function(self, channel: int, function: AnalogFunction):
        """
        Adds an analog function to a channel.

        Args:
            channel (int): Index of the analog channel.
            function (AnalogFunction): Analog function to use for this channel.
        """
        if channel in self.analog_functions:
            raise ValueError(f"Analog channel {channel} is already defined in step {self.name}.")
        if self._duration is not None and self._duration < function.min_duration:
            raise ValueError(
                f"Minimum duration of the function {function} is longer than the step duration of {self._duration}"
            )
        self.analog_functions[channel] = function

    def add_digital_function(self, channel: int, function: DigitalFunction):
        """
        Adds a digital function to a channel.

        Args:
            channel (int): Index of the digital channel.
            function (DigitalFunction): Digital function to use for this channel.
        """
        if channel in self.digital_functions:
            raise ValueError(f"Digital channel {channel} is already defined in step {self.name}.")
        if self._duration is not None and self._duration < function.min_duration:
            raise ValueError(
                f"Minimum duration of the function {function} is longer than the step duration of {self._duration}"
            )
        self.digital_functions[channel] = function

    def get_functions_on_device(
        self, analog_channels: list[int], digital_channels: list[int]
    ) -> "DeviceStep":
        """
        Splits functions on selected channels to form a device step.

        Used to isolate functions used on each of the AWG device.

        Args:
            analog_channels (list[int]): Analog channels of this device.
            digital_channels (list[int]): Digital channels of this device.

        Returns:
            DeviceStep: Device step containing only functions defined on these channels.
        """
        analog_functions = {}
        for channel in analog_channels:
            function = self.analog_functions.get(channel, None)
            if function is not None:
                analog_functions[channel] = function

        digital_functions = {}
        for channel in digital_channels:
            function = self.digital_functions.get(channel, None)
            if function is not None:
                digital_functions[channel] = function

        return DeviceStep(self.duration, analog_functions, digital_functions)

    def to_dict(self) -> dict[str, Any]:
        """
        Dict representation of the step, without function details.

        This can be serialized to JSON for saving.

        Returns:
            dict[str, Any]: dict representation of the step.
        """
        value = {}
        value["name"] = self.name
        value["duration"] = self.duration
        value["analog_functions"] = {}
        for channel in self.analog_functions:
            value["analog_functions"][channel] = self.analog_functions[channel].to_dict()
        value["digital_functions"] = {}
        for channel in self.digital_functions:
            value["digital_functions"][channel] = self.digital_functions[channel].to_dict()
        value["import"] = {"module": type(self).__module__, "name": type(self).__name__}
        return value


class StartStep(Step):
    """
    Pulse sequence start step.
    """

    def __init__(self, digital_channel_on: int = 0):
        super().__init__("__start__")
        self.duration = 10e-6
        if digital_channel_on is not None:
            self.add_digital_function(digital_channel_on, DigitalOn())


class StopStep(Step):
    """
    Pulse sequence stop step.
    """

    def __init__(self):
        super().__init__("__stop__")
        self.duration = 10e-6


class EmptyStep(Step):
    """
    Step with no function defined on any analog or digital channel.
    """

    def __init__(self, duration: float):
        super().__init__("empty")
        self.duration = duration


@dataclass
class DeviceStep:
    """
    Step with only functions on certain channels.

    This class is used to split channels in a :class:`Step`, spanning multiple AWG devices,
    to functions defined for channels on a single device.
    """

    duration: float
    analog_functions: dict[int, AnalogFunction]
    digital_functions: dict[int, DigitalFunction]

    def __init__(
        self,
        duration: float,
        analog_functions: dict[int, AnalogFunction],
        digital_functions: dict[int, DigitalFunction],
    ):
        self.duration = duration
        self.analog_functions = analog_functions
        self.digital_functions = digital_functions
