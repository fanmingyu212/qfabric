import numpy as np

from qfabric import DigitalOff, SineWave, Step
from qfabric.planner.segmenter import Segmenter


def test_Segmenter():
    segmenter = Segmenter([0, 1, 2, 3], [0, 1, 2])
    state_prep = Step("state_prep")
    state_prep.add_analog_function(1, SineWave(80e6, 1))
    state_prep.add_analog_function(5, SineWave(100e6, 1))
    state_prep.duration = 1e-3
    probe_1 = Step("probe")
    probe_1.add_analog_function(1, SineWave(80e6, 0.1))
    probe_1.add_digital_function(3, DigitalOff())
    probe_1.duration = 100e-6
    steps = [state_prep, probe_1]
    step_map = {
        (0, 0): 0,
        (0, 1): 1,
    }

    segmenter.set_steps(steps, step_map)
    out = segmenter._device_steps
    val = [
        state_prep.get_functions_on_device([0, 1, 2, 3], [0, 1, 2]),
        probe_1.get_functions_on_device([0, 1, 2, 3], [0, 1, 2]),
    ]
    np.testing.assert_equal(out, val)
