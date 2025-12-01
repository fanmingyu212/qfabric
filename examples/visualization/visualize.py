from qfabric import DigitalPulse, LinearRamp, Sequence, SineSweep, SineWave, Step
from qfabric.visualizer.plot import logic_sequence, timeline_sequence

plot_logic = True

sequence = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 1e-3
sequence.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-3))
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence.add_step(step, repeats=10)

if plot_logic:
    logic_sequence(sequence)
else:
    timeline_sequence(sequence)
