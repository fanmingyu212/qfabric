import math

import numpy as np

from qfabric import (
    ConstantAnalog,
    DigitalOff,
    DigitalOn,
    DigitalPulse,
    Sequence,
    SineWave,
    Step,
)
from qfabric.planner.segmenter.m4i6622 import (
    EmptySegmentBlock,
    M4i6622Segment,
    M4i6622Segmenter,
    add_digital_to_awg_data,
    get_max_segment_size_and_count,
    get_segment_sample_size,
    get_segment_sample_size_from_time,
    voltages_to_awg_data,
)


def test_get_segment_sample_size_from_time():
    nominal_segment_time = 0
    out = get_segment_sample_size_from_time(nominal_segment_time)
    val = 96
    assert out == val

    nominal_segment_time = 1e-6
    out = get_segment_sample_size_from_time(nominal_segment_time)
    val = 640
    assert out == val

    nominal_segment_time = 1.024e-6
    out = get_segment_sample_size_from_time(nominal_segment_time)
    val = 640
    assert out == val

    # a segment time just longer than 1.024 us + sample time.
    nominal_segment_time = math.nextafter(1.024e-6 + 1 / 625e6, math.inf)
    out = get_segment_sample_size_from_time(nominal_segment_time)
    val = 672
    assert out == val


def test_get_segment_sample_size():
    out = get_segment_sample_size(0)
    val = 96
    assert out == val

    out = get_segment_sample_size(1024)
    val = 1024
    assert out == val

    out = get_segment_sample_size(1025)
    val = 1056
    assert out == val


def test_get_max_segment_size_and_count():
    segment_count = 1
    out = get_max_segment_size_and_count(segment_count)
    val = (int(5e8) // 2, 2)
    assert out == val

    segment_count = 5
    out = get_max_segment_size_and_count(segment_count)
    val = (int(5e8) // 8, 8)
    assert out == val


def test_voltage_to_awg_data():
    times = np.arange(0, 1, 0.25)
    analog_voltages = np.sin(2 * np.pi * times)
    out = voltages_to_awg_data(analog_voltages)
    val = (np.array([0, 1, 0, -1]) / 2.5 * 2**16).astype(np.int16)
    np.testing.assert_almost_equal(out, val)


def test_add_digital_to_awg_data():
    awg_data = (np.array([1, 0, -1]) / 2.5 * 2**16).astype(np.int16)
    digital_1 = np.ones(3, dtype=bool)
    out_1 = add_digital_to_awg_data(awg_data, digital_1)
    val_1 = [-19661, -32768, -13107]
    np.testing.assert_almost_equal(out_1, val_1)

    digital_2 = np.zeros(3, dtype=bool)
    out_2 = add_digital_to_awg_data(awg_data, digital_2)
    val_2 = [13107, 0, 19661]
    np.testing.assert_almost_equal(out_2, val_2)
    np.testing.assert_almost_equal(out_1 - out_2, -32768)


def test_M4i6622Segment():
    step = Step("test")
    step.add_analog_function(0, SineWave(80.12e6, 1, stop_time=1e-3))
    step.add_analog_function(1, SineWave(89.238e6, 0.1, start_time=3e-3))
    step.add_analog_function(4, ConstantAnalog(0.1))
    step.add_digital_function(0, DigitalPulse(2.5e-3, 3.5e-3))
    step.add_digital_function(6, DigitalOn())
    step.add_digital_function(9, DigitalOff())

    # first awg: mixed constant and nonconstant regions.
    analog_channels = [0, 1, 2, 3]
    digital_channels = [0, 1, 2]
    digital_analog_map = {0: 1, 1: 2, 2: 3}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    assert segment.is_empty == False
    assert segment.segment_size == 2187520
    out = segment._constant_sample_region_indices
    val = [(655360, 1507328), (1572864, 1835008)]
    np.testing.assert_equal(out, val)

    out_num = segment.estimate_segment_blocks_needed(160000)
    val_num = 11
    np.testing.assert_equal(out_num, val_num)

    out_num = segment.estimate_segment_blocks_needed(60000)
    val_num = 22
    np.testing.assert_equal(out_num, val_num)

    segment.create_segment_blocks(60000)
    assert len(segment.segment_blocks) == 20
    val = [
        (0, 1),
        (1, 1),
        (2, 1),
        (3, 1),
        (4, 1),
        (5, 1),
        (6, 1),
        (7, 1),
        (8, 1),
        (9, 1),
        (10, 1),
        (11, 15),
        (12, 1),
        (13, 4),
        (14, 1),
        (15, 1),
        (16, 1),
        (17, 1),
        (18, 1),
        (19, 1),
    ]
    assert segment.segment_block_indices_and_repeats == val

    # second awg: all constant analog, no digital
    analog_channels = [4, 5, 6, 7]
    digital_channels = [3, 4, 5]
    digital_analog_map = {3: 5, 4: 6, 5: 7}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    assert segment.is_empty == False
    assert segment.segment_size == 2187520
    out = segment._constant_sample_region_indices
    val = [(0, 2162688)]
    np.testing.assert_equal(out, val)

    out_num = segment.estimate_segment_blocks_needed(160000)
    val_num = 2
    np.testing.assert_equal(out_num, val_num)

    segment.create_segment_blocks(160000)
    assert len(segment.segment_blocks) == 2
    val = [
        (0, 13),
        (1, 1),
    ]
    assert segment.segment_block_indices_and_repeats == val

    # third awg: all constant digital, no analog
    analog_channels = [8, 9, 10, 11]
    digital_channels = [6, 7, 8]
    digital_analog_map = {6: 9, 7: 10, 8: 11}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    assert segment.is_empty == False
    assert segment.segment_size == 2187520
    out = segment._constant_sample_region_indices
    val = [(0, 2162688)]
    np.testing.assert_equal(out, val)

    out_num = segment.estimate_segment_blocks_needed(160000)
    val_num = 2
    np.testing.assert_equal(out_num, val_num)

    segment.create_segment_blocks(160000)
    assert len(segment.segment_blocks) == 2
    val = [
        (0, 13),
        (1, 1),
    ]
    assert segment.segment_block_indices_and_repeats == val

    # fourth awg: empty, with a digital channel defined as zero
    analog_channels = [12, 13, 14, 15]
    digital_channels = [9, 10, 11]
    digital_analog_map = {9: 13, 10: 14, 11: 15}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    assert segment.is_empty == True
    assert segment.segment_size == 2187520

    out_num = segment.estimate_segment_blocks_needed(160000)
    val_num = 1
    np.testing.assert_equal(out_num, val_num)

    segment.create_segment_blocks(160000)
    assert len(segment.segment_blocks) == 2
    val = [
        (0, 13),
        (1, 1),
    ]
    assert segment.segment_block_indices_and_repeats == val

    # fifth awg: empty with no analog or digital channel defined.
    analog_channels = [16, 17, 18, 19]
    digital_channels = [12, 13, 14]
    digital_analog_map = {12: 17, 13: 18, 14: 19}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    assert segment.is_empty == True
    assert segment.segment_size == 2187520

    out_num = segment.estimate_segment_blocks_needed(160000)
    val_num = 1
    np.testing.assert_equal(out_num, val_num)

    segment.create_segment_blocks(160000)
    assert len(segment.segment_blocks) == 2
    val = [
        (0, 13),
        (1, 1),
    ]
    assert segment.segment_block_indices_and_repeats == val


def test_SegmentBlock():
    step = Step("test")
    step.add_analog_function(0, SineWave(80.12e6, 1, stop_time=1e-3))
    step.add_analog_function(1, SineWave(89.238e6, 0.1, start_time=3e-3))
    step.add_digital_function(0, DigitalPulse(2.5e-3, 3.5e-3))

    analog_channels = [0, 1, 2, 3]
    digital_channels = [0, 1, 2]
    digital_analog_map = {0: 1, 1: 2, 2: 3}
    device_step = step.get_functions_on_device(analog_channels, digital_channels)

    segment_1 = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    segment_1.create_segment_blocks(160000)
    segment_block_1 = segment_1.segment_blocks[0]
    segment_2 = M4i6622Segment(device_step, analog_channels, digital_analog_map)
    segment_2.create_segment_blocks(160000)
    segment_block_2 = segment_2.segment_blocks[0]
    assert segment_block_1 == segment_block_2
    assert segment_block_1 != segment_1.segment_blocks[1]


def test_EmptySegmentBlock():
    segment_block = EmptySegmentBlock(1000)
    np.testing.assert_equal(segment_block.awg_data, np.zeros(4000, dtype=np.int16))


def test_M4i6622Segmenter():
    sequence_1 = Sequence()
    sequence_2 = Sequence()
    # same for the two sequences.
    state_prep = Step("state_prep")
    state_prep.add_analog_function(0, SineWave(80.12e6, 1))
    state_prep.duration = 1e-3
    sequence_1.add_step(state_prep)
    sequence_2.add_step(state_prep)

    # different data for the two sequences on an untested AWG.
    rf_1 = Step("rf")
    rf_1.add_analog_function(4, SineWave(200e6, 1))
    rf_1.duration = 0.5e-3
    sequence_1.add_step(rf_1)
    rf_2 = Step("rf")
    rf_2.add_analog_function(4, SineWave(210e6, 1))
    rf_2.duration = 0.5e-3
    sequence_2.add_step(rf_2)

    # different data for the two sequences on the tested AWG.
    probe_1 = Step("probe")
    probe_1.add_analog_function(1, SineWave(100e6, 1))
    probe_1.duration = 100e-6
    sequence_1.add_step(probe_1, delay_time_after_previous=1e-3)
    probe_2 = Step("probe")
    probe_2.add_analog_function(1, SineWave(110e6, 1))
    probe_2.duration = 100e-6
    sequence_2.add_step(probe_2, delay_time_after_previous=1e-3)

    # same for the two sequences.
    measure = Step("measurement")
    measure.add_analog_function(0, SineWave(80e6, 0.1))
    measure.duration = 1e-3
    sequence_1.add_step(measure, repeats=10)
    sequence_2.add_step(measure, repeats=10)

    analog_channels = [0, 1, 2, 3]
    digital_channels = [0, 1, 2]
    analog_channels_to_store_digital_data = [1, 2, 3]
    m4i6622 = M4i6622Segmenter(
        analog_channels, digital_channels, analog_channels_to_store_digital_data
    )

    sequences = [sequence_1, sequence_2]
    dedup_steps: list[Step] = []
    sequence_to_steps_map: dict[int, list[int]] = {}
    for sequence_index in range(len(sequences)):
        sequence = sequences[sequence_index]
        steps = sequence.get_steps()
        sequence_to_steps_map[sequence_index] = []
        for step_index in range(len(steps)):
            step = steps[step_index]
            try:
                step_saved_index = dedup_steps.index(step)
            except ValueError:
                dedup_steps.append(step)
                step_saved_index = len(dedup_steps) - 1
            sequence_to_steps_map[sequence_index].append(step_saved_index)
    m4i6622.set_steps(dedup_steps, sequence_to_steps_map)

    device_steps = m4i6622._device_steps
    out = [device_step.duration for device_step in device_steps]
    val = [1e-05, 0.001, 0.0005, 0.001, 0.0001, 0.001, 1e-05, 5e-4, 0.0001]
    np.testing.assert_allclose(out, val)

    out = m4i6622._sequence_to_segments_map
    val = {0: [0, 1, 2, 3, 4, 5, 6], 1: [0, 1, 2, 3, 7, 5, 6]}
    assert out == val

    out = m4i6622._check_awg_memory_limit([0, 1])
    val = [0, 1]
    assert out == val

    out = m4i6622.valid_max_segment_block_size
    val = 31250016
    assert out == val

    segments, step_to_segment_map = m4i6622._get_divided_segments([0, 1])
    assert len(segments) == 8
    val = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 2, 8: 7}
    assert step_to_segment_map == val

    for sequence_index in [0, 1]:
        out_num_of_samples = 0
        for step_order, step_index in enumerate(sequence_to_steps_map[sequence_index]):
            segment = segments[step_to_segment_map[step_index]]
            for (
                segment_block_index,
                segment_block_repeat,
            ) in segment.segment_block_indices_and_repeats:
                if segment_block_index == -1:
                    num_of_samples = m4i6622.valid_max_segment_block_size
                else:
                    num_of_samples = len(segment.segment_blocks[segment_block_index].awg_data) // 4
                out_num_of_samples += (
                    num_of_samples
                    * segment_block_repeat
                    * sequences[sequence_index].get_repeats()[step_order]
                )
        val_num_of_samples = 0
        sequence = sequences[sequence_index]
        steps = sequence.get_steps()
        repeats = sequence.get_repeats()
        for kk in range(len(steps)):
            val_num_of_samples += (
                get_segment_sample_size_from_time(steps[kk].duration) * repeats[kk]
            )
        assert out_num_of_samples == val_num_of_samples

        # TODO: test scheduling experiments again.
