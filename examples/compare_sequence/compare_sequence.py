# Example script showing how to compare sequences.

from pprint import pprint

from deepdiff import DeepDiff  # requires deepdiff.

from qfabric import DigitalPulse, LinearRamp, Sequence, SineSweep, SineWave, Step
from qfabric.visualizer.plot import logic_sequence, timeline_sequence

sequence_1 = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 1e-3
sequence_1.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-3))
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence_1.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence_1.add_step(step, repeats=10)

sequence_2 = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 1e-3
sequence_2.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-3))
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence_2.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence_2.add_step(step, repeats=10)

step = Step("extra")
step.duration = 1e-6
sequence_2.add_step(step)

print("For sequence with one extra step:")
pprint(DeepDiff(sequence_1, sequence_2))
print()

sequence_3 = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 1e-3
sequence_3.add_step(step)

step = Step("sweep")
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence_3.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence_3.add_step(step, repeats=10)

print("For sequence with one less function:")
pprint(DeepDiff(sequence_1, sequence_3))
print()

sequence_4 = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(100e6, 1))
step.duration = 1e-3
sequence_4.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-3))
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence_4.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence_4.add_step(step, repeats=9)

print("For sequence with different repeat for a step:")
pprint(DeepDiff(sequence_1, sequence_4))
print()

sequence_5 = Sequence()

step = Step("sine")
step.add_analog_function(0, SineWave(101e6, 1))
step.duration = 1e-3
sequence_5.add_step(step)

step = Step("sweep")
step.add_analog_function(1, SineSweep(90e6, 110e6, 1, 1, 0, 1e-3))
step.add_digital_function(0, DigitalPulse(5e-4, 1e-3))
sequence_5.add_step(step)

step = Step("ramp")
step.add_analog_function(2, LinearRamp(-1, 1, 0, 1e-3))
step.add_digital_function(1, DigitalPulse(5e-4, 1e-3))
sequence_5.add_step(step, repeats=10)

print("For sequence with different function parameter:")
pprint(DeepDiff(sequence_1, sequence_5))
print()
