import os

import pytest

from qfabric import (
    DigitalOn,
    DigitalPulse,
    ExperimentManager,
    LinearRamp,
    Sequence,
    SineWave,
    Step,
)


def test_experimental_manager():
    sequence = Sequence()
    step_1 = Step("step 1")
    step_1.add_analog_function(
        0, LinearRamp(start_amplitude=-1, stop_amplitude=1, start_time=0, stop_time=1)
    )
    step_1.add_analog_function(1, SineWave(10, 1))
    sequence.add_step(step_1)

    step_2 = Step("step 2")
    step_2.duration = 0.5
    step_2.add_digital_function(0, DigitalOn())
    sequence.add_step(step_2, delay_time_after_previous=0.5)

    step_3 = Step("step 3")
    step_3.add_analog_function(2, SineWave(10, 0.5))
    step_3.add_digital_function(1, DigitalPulse(0.25, 0.5))
    sequence.add_step(step_3, repeats=3)

    step_4 = Step("step 4")
    step_4.duration = 0.5
    step_4.add_digital_function(0, DigitalOn())
    step_4.add_analog_function(3, SineWave(20, 0.5))
    sequence.add_step(step_4)

    # test 1: single sequence
    script_dir = os.path.dirname(os.path.abspath(__file__))
    manager = ExperimentManager(os.path.join(script_dir, "mock_config.toml"))
    manager.schedule(sequence)
    manager.setup()
    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    manager.set_principal_device_trigger(external=True)
    assert manager.programmer.devices[0]._external_trigger == True
    assert manager.programmer.devices[1]._external_trigger == True

    manager.set_principal_device_trigger(external=False)
    assert manager.programmer.devices[0]._external_trigger == False
    assert manager.programmer.devices[1]._external_trigger == True

    assert manager.programmer.devices[0].is_principal_device == True

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 1), (4, 1)]
    assert out == val

    assert manager.programmer.devices[1].is_principal_device == False

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 3), (4, 1), (0, 1)]
    assert out == val

    sequence_new = Sequence()
    step_1 = Step("step 1")
    step_1.add_analog_function(
        0, LinearRamp(start_amplitude=-1, stop_amplitude=1, start_time=0, stop_time=1)
    )
    step_1.add_analog_function(1, SineWave(10, 1))
    sequence_new.add_step(step_1)

    step_2 = Step("step 2")
    step_2.duration = 0.5
    step_2.add_digital_function(0, DigitalOn())
    sequence_new.add_step(step_2, delay_time_after_previous=0.5)

    step_5 = Step("step 5")
    step_5.duration = 0.5
    step_5.add_analog_function(3, SineWave(20, 0.8))
    sequence_new.add_step(step_5)

    # test 2: 2 sequences, 2 repeats, program both sequences at once
    manager.schedule([sequence_new, sequence], 2)
    manager.setup()
    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 1), (0, 1)]
    assert out == val

    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (4, 3), (5, 1), (0, 1)]
    assert out == val

    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 1), (0, 1)]
    assert out == val

    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (4, 3), (5, 1), (0, 1)]
    assert out == val

    assert len(manager.scheduled_sequence_indices) == 0
    with pytest.raises(IndexError):
        manager.program_next_sequence()

    # test 3: 2 sequences, 2 repeats, program 1 sequence at once
    manager.schedule([sequence_new, sequence], 2)
    manager.setup(program_single_sequence_only=True)
    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 1), (0, 1)]
    assert out == val

    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 3), (4, 1), (0, 1)]
    assert out == val
    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 1), (0, 1)]
    assert out == val

    manager.program_next_sequence()
    manager.run(wait_for_finish=True)

    out = manager.programmer.devices[0]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (3, 1), (2, 3), (3, 1), (4, 1)]
    assert out == val

    out = manager.programmer.devices[1]._segment_indices_and_repeats
    val = [(0, 1), (1, 1), (2, 1), (2, 1), (3, 3), (4, 1), (0, 1)]
    assert out == val

    with pytest.raises(IndexError):
        manager.program_next_sequence()
