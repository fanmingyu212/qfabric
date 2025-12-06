from dataclasses import field

import numpy as np
import pytest

from qfabric import ConstantAnalog, DigitalPulse
from qfabric.sequence.basic_functions import DigitalOn
from qfabric.sequence.function import (
    AnalogEmpty,
    AnalogFunction,
    AnalogProduct,
    AnalogSequence,
    AnalogSum,
    DigitalEmpty,
    DigitalSequence,
)


def test_AnalogEmpty():
    f = AnalogEmpty()
    times = np.array([0, 1, 2], dtype=float)
    out = f.output(times)
    expected = np.zeros_like(times)
    np.testing.assert_allclose(out, expected)
    assert f.min_duration == 0


def test_DigitalEmpty():
    f = DigitalEmpty()
    times = np.array([0, 1, 2], dtype=float)
    out = f.output(times)
    expected = np.zeros(len(times), dtype=bool)
    np.testing.assert_array_equal(out, expected)
    assert f.min_duration == 0


def test_AnalogSum_basic():
    f1 = ConstantAnalog(1.0, stop_time=2.0)
    f2 = ConstantAnalog(2.0, stop_time=5.0)
    f3 = ConstantAnalog(-0.5, stop_time=1.0)

    f = AnalogSum(f1, f2, f3)

    # min_duration = max(min_durations)
    assert f.min_duration == 5.0

    times = np.array([0.0, 0.5, 1.5, 2.5, 4.5, 5.5])
    out = f.output(times)
    expected = [2.5, 2.5, 3, 2, 2, 0]
    np.testing.assert_allclose(out, expected)

    out = f.to_dict()
    val = {
        "import": {"module": "qfabric.sequence.function", "name": "AnalogSum"},
        "fields": {
            "functions": [
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 1.0, "start_time": None, "stop_time": 2.0},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 2.0, "start_time": None, "stop_time": 5.0},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": -0.5, "start_time": None, "stop_time": 1.0},
                },
            ]
        },
    }
    assert out == val


def test_AnalogProduct_basic():
    f1 = ConstantAnalog(3.0, stop_time=4)
    f2 = ConstantAnalog(2.0, stop_time=1)
    f3 = ConstantAnalog(-1.0, stop_time=5)

    f = AnalogProduct(f1, f2, f3)

    # min_duration = max
    assert f.min_duration == 5.0

    times = np.array([0.0, 0.5, 1.5, 2.5, 4.5, 5.5])
    out = f.output(times)
    expected = [-6, -6, 0, 0, 0, 0]
    np.testing.assert_allclose(out, expected)

    out = f.to_dict()
    val = {
        "import": {"module": "qfabric.sequence.function", "name": "AnalogProduct"},
        "fields": {
            "functions": [
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 3.0, "start_time": None, "stop_time": 4},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 2.0, "start_time": None, "stop_time": 1},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": -1.0, "start_time": None, "stop_time": 5},
                },
            ]
        },
    }
    assert out == val


def test_AnalogSequence_add_and_min_duration():
    seq = AnalogSequence()

    f1 = ConstantAnalog(1.0, stop_time=2)
    f2 = ConstantAnalog(2.0, stop_time=3)

    seq.add_function(f1)  # start=0, duration=2
    assert seq.min_duration == 2.0

    seq.add_function(f2, delay_time_after_previous=1.0)  # start=3, duration=3
    assert seq.min_duration == 6.0  # 3 + 3

    out = seq.to_dict()
    val = {
        "import": {"module": "qfabric.sequence.function", "name": "AnalogSequence"},
        "fields": {
            "functions": [
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 1.0, "start_time": None, "stop_time": 2},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "ConstantAnalog",
                    },
                    "fields": {"amplitude": 2.0, "start_time": None, "stop_time": 3},
                },
            ],
            "start_times": [0, 3.0],
            "durations": [2, 3],
            "use_coherent_phases": [False, False],
        },
    }
    assert out == val


def test_AnalogSequence_output_no_overlap():
    seq = AnalogSequence()

    f1 = ConstantAnalog(1.0, stop_time=2)
    f2 = ConstantAnalog(3.0, stop_time=1)

    seq.add_function(f1)  # [0,2)
    seq.add_function(f2, 1.0)  # start=3, duration=1 → [3,4)

    times = np.array([0, 1, 2, 3, 3.5, 5], float)
    out = seq.output(times)

    expected = np.array([1.0, 1.0, 0.0, 3.0, 3.0, 0.0])
    np.testing.assert_allclose(out, expected)


def test_AnalogSequence_coherent_phase():
    # Use functions whose output depends on time_offset
    class OffsetSensitive(AnalogFunction):
        @property
        def min_duration(self):
            return 1.0

        def output(self, times, time_offset=0):
            return times + time_offset * 2

    seq = AnalogSequence()

    f = OffsetSensitive()

    seq.add_function(f, duration=1.0, coherent_phase=False)
    seq.add_function(f, duration=1.0, coherent_phase=True)

    times = np.array([0.5, 1.5])  # first pulse at [0,1), second at [1,2)

    out = seq.output(times)

    # For t=0.5 → pulse 1 applies, time_offset = 0
    # For t=1.5 → pulse 2 applies, time_offset = start_time_of_pulse = 1.0
    expected = np.array([0.5 + 0.0, 0.5 + 2.0])  # first pulse  # second pulse (coherent)

    np.testing.assert_allclose(out, expected)


def test_AnalogSequence_errors():
    seq = AnalogSequence()
    f = ConstantAnalog(1.0)  # no well-defined duration

    # Zero-duration function without explicit duration → error
    with pytest.raises(ValueError):
        seq.add_function(f)

    # Negative delay
    with pytest.raises(ValueError):
        seq.add_function(ConstantAnalog(1.0, 1.0), delay_time_after_previous=-1)

    # Negative duration
    with pytest.raises(ValueError):
        seq.add_function(ConstantAnalog(1.0, 1.0), duration=-1)


def test_DigitalSequence_add_and_min_duration():
    seq = DigitalSequence(default_on=False)

    f1 = DigitalPulse(1, 2)
    f2 = DigitalPulse(0, 1)

    seq.add_function(f1)
    assert seq.min_duration == 2.0

    seq.add_function(f2, delay_time_after_previous=3.0)  # start=5, dur=1
    assert seq.min_duration == 6.0

    out = seq.to_dict()
    val = {
        "import": {"module": "qfabric.sequence.function", "name": "DigitalSequence"},
        "fields": {
            "default_on": False,
            "functions": [
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "DigitalPulse",
                    },
                    "fields": {"start_time": 1, "stop_time": 2, "default_off": True},
                },
                {
                    "import": {
                        "module": "qfabric.sequence.basic_functions",
                        "name": "DigitalPulse",
                    },
                    "fields": {"start_time": 0, "stop_time": 1, "default_off": True},
                },
            ],
            "start_times": [0, 5.0],
            "durations": [2, 1],
        },
    }
    assert out == val


def test_DigitalSequence_output():
    seq = DigitalSequence(default_on=False)

    f1 = DigitalPulse(1, 2)
    f2 = DigitalPulse(0, 1)

    seq.add_function(f1)  # [0,1)
    seq.add_function(f2, 1.0)  # [3,4)

    times = np.array([0, 0.5, 1.5, 2.5, 3.5, 5], float)
    out = seq.output(times)

    expected = np.array(
        [
            False,
            False,
            True,
            False,
            True,
            False,
        ]
    )

    np.testing.assert_array_equal(out, expected)


def test_DigitalSequence_errors():
    seq = DigitalSequence()
    f = DigitalPulse(0, 1)

    # No duration for zero-duration function
    with pytest.raises(ValueError):
        seq.add_function(DigitalOn())

    # Negative delay
    with pytest.raises(ValueError):
        seq.add_function(DigitalPulse(0, 1), delay_time_after_previous=-2)

    # Negative duration
    with pytest.raises(ValueError):
        seq.add_function(DigitalPulse(0, 1), duration=-1)


def test_Function_set_public_attributes_not_fields():
    class PublicAttributeFunction(AnalogFunction):
        def __init__(self):
            self.public_attribute = ""

    with pytest.raises(AttributeError):
        PublicAttributeFunction()


def test_Function_set_private_attributes_not_fields():
    class PrivateAttributeFunction(AnalogFunction):
        def __init__(self):
            self._private_attribute = ""

    with pytest.raises(AttributeError):
        PrivateAttributeFunction()


def test_Function_set_non_comparison_fields():
    class FunctionWithTimings(AnalogFunction):
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

    f = FunctionWithTimings(1, 2, 3)
    f_1 = FunctionWithTimings(1, 2, 3)
    f_1._total_time = 7  # only for testing
    # this works as _total_time is defined as a non-comparison field.
    assert f == f_1


def test_Function_inheritance_no_init():
    """Tests if dataclass(init=False) is set for subclasses of Function."""

    class Function1(AnalogFunction):
        frequency: float

        def __init__(self, frequency: float = 1):
            self.frequency = frequency

        @property
        def min_duration(self) -> float:
            return 0

    class Function2(Function1): ...

    f = Function2()
    assert f.frequency == 1
