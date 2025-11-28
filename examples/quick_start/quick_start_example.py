import os

from qfabric import (
    AnalogSequence,
    ConstantAnalog,
    DigitalOn,
    ExperimentManager,
    LinearRamp,
    Sequence,
    SineWave,
    Step,
)

sequence = Sequence()
step_1 = Step("step 1")
step_1.add_analog_function(0, SineWave(frequency=1e5, amplitude=1))
analog_ch1 = AnalogSequence()
analog_ch1.add_function(
    LinearRamp(start_amplitude=0, stop_amplitude=1, start_time=0, stop_time=5e-5)
)
analog_ch1.add_function(
    LinearRamp(start_amplitude=1, stop_amplitude=0, start_time=0, stop_time=5e-5)
)
step_1.add_analog_function(1, analog_ch1)
sequence.add_step(step_1)

step_2 = Step("step 2")
step_2.duration = 5e-5
step_2.add_analog_function(1, ConstantAnalog(0.5))
step_2.add_digital_function(0, DigitalOn())
sequence.add_step(step_2, delay_time_after_previous=5e-5)

step_3 = Step("step 3")
step_3.duration = 5e-5
step_3.add_analog_function(0, SineWave(2e5, 1, start_time=2e-5))
sequence.add_step(step_3, repeats=5)


script_dir = os.path.dirname(os.path.abspath(__file__))
manager = ExperimentManager(os.path.join(script_dir, "config.toml"))
manager.schedule(sequence)
manager.setup()
while len(manager.scheduled_sequence_indices) > 0:
    manager.program_next_sequence()
    manager.run(wait_for_finish=True)
