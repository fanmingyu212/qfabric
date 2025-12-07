from qfabric import (
    ConstantAnalog,
    DigitalOn,
    DigitalPulse,
    Sequence,
    SineWave,
    Step,
)
from qfabric.planner.segmenter.awg710 import (
    MIN_SAMPLES_PER_SEGMENT_BLOCK,
    MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK,
    SAMPLE_RATE,
    AWG710Segment,
    AWG710Segmenter,
    get_segment_sample_size,
    get_segment_sample_size_from_time,
)
from qfabric.sequence.step import StartStep, StopStep


def test_get_segment_sample_size_from_time():
    t = 0
    out = get_segment_sample_size_from_time(t)
    assert out == MIN_SAMPLES_PER_SEGMENT_BLOCK

    t = 1e-6
    out = get_segment_sample_size_from_time(t)
    assert out % MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK == 0
    assert out >= int(t * SAMPLE_RATE)
    assert out < int(t * SAMPLE_RATE + MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK + 1)


def test_get_segment_sample_size():
    out = get_segment_sample_size(0)
    assert out == MIN_SAMPLES_PER_SEGMENT_BLOCK

    base = 1000
    out = get_segment_sample_size(base)
    assert out == base

    out = get_segment_sample_size(base + 1)
    assert out == base + MULTIPLE_SAMPLES_PER_SEGMENT_BLOCK


def test_AWG710Segment_basic():
    # one analog, two digital
    step = Step("test")
    step.add_analog_function(0, SineWave(50e6, 1, stop_time=1e-3))
    step.add_digital_function(0, DigitalOn())
    step.add_digital_function(1, DigitalPulse(1e-4, 2e-4))

    device_step = step.get_functions_on_device([0], [0, 1])
    seg = AWG710Segment(device_step, 0, [0, 1])

    assert seg.segment_size >= MIN_SAMPLES_PER_SEGMENT_BLOCK
    assert len(seg.analog_data) == seg.segment_size
    assert len(seg.digital_1) == seg.segment_size
    assert len(seg.digital_2) == seg.segment_size


def test_AWG710Segment_equality():
    s1 = Step("s1")
    s1.add_analog_function(0, ConstantAnalog(0.5))
    ds1 = s1.get_functions_on_device([0], [0, 1])

    s2 = Step("s2")
    s2.add_analog_function(1, ConstantAnalog(0.5))
    ds2 = s2.get_functions_on_device([0], [0, 1])

    seg1 = AWG710Segment(ds1, 0, [0, 1])
    seg2 = AWG710Segment(ds1, 0, [0, 1])
    seg3 = AWG710Segment(ds2, 0, [0, 1])

    assert seg1 == seg2
    assert seg1 != seg3  # different DeviceStep


def test_AWG710Segmenter_set_steps_and_map():
    sequence_1 = Sequence()
    sequence_2 = Sequence()

    # exact same first step
    state = Step("prep")
    state.add_analog_function(0, SineWave(20e6, 1))
    state.duration = 1e-3
    sequence_1.add_step(state)
    sequence_2.add_step(state)

    # different data for two sequences
    a1 = Step("drive")
    a1.add_analog_function(0, SineWave(40e6, 1))
    a1.duration = 2e-3
    sequence_1.add_step(a1)

    a2 = Step("drive")
    a2.add_analog_function(0, SineWave(45e6, 1))
    a2.duration = 2e-3
    sequence_2.add_step(a2)

    # shared final step
    meas = Step("meas")
    meas.add_analog_function(0, ConstantAnalog(0.2))
    meas.duration = 0.5e-3
    sequence_1.add_step(meas)
    sequence_2.add_step(meas)

    analog_channels = [0]
    digital_channels = [0, 1]
    segm = AWG710Segmenter(analog_channels, digital_channels)

    sequences = [sequence_1, sequence_2]

    # dedup steps as in m4i test structure
    dedup_steps = []
    sequence_to_steps_map = {}

    for si, seq in enumerate(sequences):
        steps = [StartStep(1e-5)] + seq.get_steps() + [StopStep(1e-5)]
        sequence_to_steps_map[si] = []
        for st in steps:
            try:
                idx = dedup_steps.index(st)
            except ValueError:
                dedup_steps.append(st)
                idx = len(dedup_steps) - 1
            sequence_to_steps_map[si].append(idx)

    segm.set_steps(dedup_steps, sequence_to_steps_map)

    # sequence→segment map should have same length as steps
    assert len(segm._sequence_to_segments_map[0]) == len(sequence_to_steps_map[0])
    assert len(segm._sequence_to_segments_map[1]) == len(sequence_to_steps_map[1])

    # device step count matches dedup count
    assert len(segm._device_steps) == len(dedup_steps)

    # segments created
    assert len(segm._segments) >= 1


def test_AWG710_get_awg_memory_data_structure():
    # simple short sequence
    seq = Sequence()
    st = Step("pulse")
    st.add_analog_function(0, ConstantAnalog(0.1))
    st.duration = 1e-4
    seq.add_step(st)

    analog_channels = [0]
    digital_channels = [0, 1]

    segm = AWG710Segmenter(analog_channels, digital_channels)

    dedup_steps = []
    sequence_to_steps_map = {}

    steps = [StartStep(1e-5)] + seq.get_steps() + [StopStep(1e-5)]
    sequence_to_steps_map[0] = []

    for step in steps:
        try:
            idx = dedup_steps.index(step)
        except ValueError:
            dedup_steps.append(step)
            idx = len(dedup_steps) - 1
        sequence_to_steps_map[0].append(idx)

    segm.set_steps(dedup_steps, sequence_to_steps_map)

    data, step_map, seqs = segm.get_awg_memory_data([0])

    # structural checks
    assert "segments" in data
    assert isinstance(data["segments"], list)
    assert len(seqs) == 1
    assert isinstance(step_map, dict)
    assert len(step_map) == len(sequence_to_steps_map[0])
