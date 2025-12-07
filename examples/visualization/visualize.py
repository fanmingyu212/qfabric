from qfabric import (
    DigitalPulse,
    LinearRamp,
    Sequence,
    SineSweep,
    SineWave,
    Step,
    logic_sequence,
    timeline_sequence,
)

plot_logic = False  # change to True to see the plot in the logic mode.

sequence = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 5e-6
sequence.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-5))
step.add_digital_function(0, DigitalPulse(5e-6, 1e-5))
sequence.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-5))
step.add_digital_function(1, DigitalPulse(5e-6, 1e-5))
sequence.add_step(step, repeats=10)

# optional channel labels
analog_channels = {0: "AOM", 1: "EOM", 2: "laser mod"}
digital_channels = {0: "trigger_1", 1: "trigger_2"}

if plot_logic:
    logic_sequence(sequence, analog_map=analog_channels, digital_map=digital_channels)
else:
    timeline_sequence(sequence, analog_map=analog_channels, digital_map=digital_channels)
