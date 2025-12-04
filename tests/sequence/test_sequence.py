import numpy as np

from qfabric import Sequence, Step
from qfabric.sequence.step import EmptyStep, StartStep, StopStep


def test_Sequence():
    sequence = Sequence()
    state_prep = Step("state_prep")
    state_prep.duration = 1e-3
    sequence.add_step(state_prep)
    probe = Step("probe")
    probe.duration = 100e-6
    sequence.add_step(probe, delay_time_after_previous=1e-3)
    measure = Step("measurement")
    measure.duration = 1e-3
    sequence.add_step(measure, repeats=10)
    np.testing.assert_allclose(sequence.nominal_duration, 12.1e-3)

    out = sequence.get_steps()
    val = [state_prep, EmptyStep(1e-3), probe, measure]
    np.testing.assert_equal(out, val)
    out = sequence.get_repeats()
    val = [1, 1, 1, 10]
    np.testing.assert_equal(out, val)

    sequence_1 = Sequence()
    state_prep_1 = Step("state_prep")
    state_prep_1.duration = 1e-3
    sequence_1.add_step(state_prep)
    probe_1 = Step("probe")
    probe_1.duration = 100e-6
    sequence_1.add_step(probe, delay_time_after_previous=1e-3)
    measure_1 = Step("measurement")
    measure_1.duration = 1e-3
    sequence_1.add_step(measure, repeats=10)
    assert sequence == sequence_1

    out = sequence.to_dict()
    val = {
        "repeats": [1, 1, 1, 10],
        "steps": [
            {
                "name": "state_prep",
                "duration": 0.001,
                "analog_functions": {},
                "digital_functions": {},
                "import": {"module": "qfabric.sequence.step", "name": "Step"},
            },
            {
                "name": "empty",
                "duration": 0.001,
                "analog_functions": {},
                "digital_functions": {},
                "import": {"module": "qfabric.sequence.step", "name": "EmptyStep"},
            },
            {
                "name": "probe",
                "duration": 0.0001,
                "analog_functions": {},
                "digital_functions": {},
                "import": {"module": "qfabric.sequence.step", "name": "Step"},
            },
            {
                "name": "measurement",
                "duration": 0.001,
                "analog_functions": {},
                "digital_functions": {},
                "import": {"module": "qfabric.sequence.step", "name": "Step"},
            },
        ],
        "import": {"module": "qfabric.sequence.sequence", "name": "Sequence"},
    }
    assert out == val
