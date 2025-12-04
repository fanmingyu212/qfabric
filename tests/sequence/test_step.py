import numpy as np
import pytest

from qfabric import (
    ConstantAnalog,
    DigitalOff,
    DigitalPulse,
    SineSweep,
    Step,
)
from qfabric.sequence.step import EmptyStep, StartStep, Step, StopStep


def test_Step():
    step = Step("test")
    step.add_analog_function(0, ConstantAnalog(1))
    with pytest.raises(ValueError):
        step.add_analog_function(0, ConstantAnalog(0.1))
    step.add_digital_function(1, DigitalPulse(10e-6, 20e-6))
    with pytest.raises(ValueError):
        step.add_digital_function(1, DigitalPulse(1e-3, 2e-3))
    np.testing.assert_allclose(step.duration, 20e-6)

    out = step.to_dict()
    val = {
        "name": "test",
        "duration": 2e-05,
        "analog_functions": {
            0: {
                "import": {"module": "qfabric.sequence.basic_functions", "name": "ConstantAnalog"},
                "fields": {"amplitude": 1, "start_time": None, "stop_time": None},
            }
        },
        "digital_functions": {
            1: {
                "import": {"module": "qfabric.sequence.basic_functions", "name": "DigitalPulse"},
                "fields": {"start_time": 1e-05, "stop_time": 2e-05, "default_off": True},
            }
        },
        "import": {"module": "qfabric.sequence.step", "name": "Step"},
    }
    assert out == val

    step_copy = Step("test_copy")
    step_copy.add_analog_function(0, ConstantAnalog(1))
    step_copy.add_digital_function(1, DigitalPulse(10e-6, 20e-6))
    assert step == step_copy

    step_diff_channel = Step("test_different_channel")
    step_diff_channel.add_analog_function(1, ConstantAnalog(1))
    step_diff_channel.add_digital_function(1, DigitalPulse(10e-6, 20e-6))
    assert step != step_diff_channel

    step_diff_amp = Step("test_different_amplitude")
    step_diff_amp.add_analog_function(0, ConstantAnalog(0.1))
    step_diff_amp.add_digital_function(1, DigitalPulse(10e-6, 20e-6))
    assert step != step_diff_amp

    step_diff_time = Step("test_different_time")
    step_diff_time.add_analog_function(0, ConstantAnalog(1))
    step_diff_time.add_digital_function(1, DigitalPulse(10e-6, 20e-6))
    step_diff_time.duration = 30e-6
    np.testing.assert_allclose(step_diff_time.duration, 30e-6)
    assert step != step_diff_time
    with pytest.raises(ValueError):
        step_diff_time.duration = 10e-6


def test_StartStep():
    start = StartStep()
    np.testing.assert_allclose(start.duration, 10e-6)
    assert len(start.analog_functions) == 0
    assert len(start.digital_functions) == 1


def test_StopStep():
    stop = StopStep()
    np.testing.assert_allclose(stop.duration, 10e-6)
    assert len(stop.analog_functions) == 0
    assert len(stop.digital_functions) == 0


def test_EmptyStep():
    empty = EmptyStep(100e-6)
    np.testing.assert_allclose(empty.duration, 100e-6)
    assert len(empty.analog_functions) == 0
    assert len(empty.digital_functions) == 0


def test_DeviceStep():
    step = Step("test")
    analog_0 = ConstantAnalog(1)
    analog_1 = ConstantAnalog(0.1)
    analog_4 = SineSweep(80e6, 100e6, 0.1, 0.1, 0, 20e-6)
    digial_1 = DigitalOff()
    digial_5 = DigitalPulse(10e-6, 15e-6)
    step.add_analog_function(0, analog_0)
    step.add_analog_function(1, analog_1)
    step.add_analog_function(4, analog_4)
    step.add_digital_function(1, digial_1)
    step.add_digital_function(5, digial_5)

    device_step_1 = step.get_functions_on_device([0, 1, 2, 3], [0, 1, 2])
    device_step_1_diff_order = step.get_functions_on_device([2, 3, 1, 0], [2, 0, 1])
    device_step_2 = step.get_functions_on_device([4, 5, 6, 7], [3, 4, 5])
    assert device_step_1 == device_step_1_diff_order
    assert device_step_1 != device_step_2
    assert device_step_1.analog_functions[0] == analog_0
    assert device_step_1.analog_functions[1] == analog_1
    assert device_step_2.analog_functions[4] == analog_4
    assert device_step_1.digital_functions[1] == digial_1
    assert device_step_2.digital_functions[5] == digial_5
