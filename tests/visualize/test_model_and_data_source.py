import numpy as np

from qfabric import (
    DigitalPulse,
    LinearRamp,
    Sequence,
    SineSweep,
    SineWave,
    Step,
)
from qfabric.visualizer.data_source import color_cycle, get_sequence_plot_data_source
from qfabric.visualizer.models import (
    FunctionBlockModel,
    SequenceModel,
    _value_to_str_with_prefix,
)


def test_value_to_str_with_prefix_scaling_and_sign():
    # scalar magnitude ranges
    assert _value_to_str_with_prefix(0.0) == "0"
    assert _value_to_str_with_prefix(1e-9) == "1.0n"
    assert _value_to_str_with_prefix(1e-6) == "1.0µ"
    assert _value_to_str_with_prefix(1e-3) == "1.0m"
    assert _value_to_str_with_prefix(1.0) == "1.0"
    assert _value_to_str_with_prefix(1e3) == "1.0k"
    assert _value_to_str_with_prefix(1e6) == "1.0M"
    assert _value_to_str_with_prefix(1e9) == "1.0G"

    # sign
    assert _value_to_str_with_prefix(-1e-3).startswith("-")

    # bool and non-scalar
    assert _value_to_str_with_prefix(True) == "True"
    arr = np.array([1, 2, 3])
    assert _value_to_str_with_prefix(arr) == str(arr)


def _build_demo_sequence():
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

    analog_channels = {0: "AOM", 1: "EOM", 2: "laser mod"}
    digital_channels = {0: "trigger_1", 1: "trigger_2"}

    return sequence, analog_channels, digital_channels


def test_sequence_model_basic_structure():
    sequence, analog_map, digital_map = _build_demo_sequence()

    model = SequenceModel(sequence, analog_map=analog_map, digital_map=digital_map)

    # channel names should respect analog_map/digital_map ordering
    assert "AOM" in model.channel_names
    assert "EOM" in model.channel_names
    assert "laser mod" in model.channel_names
    assert "trigger_1" in model.channel_names
    assert "trigger_2" in model.channel_names

    # steps + repeats
    assert len(model.steps) == 3
    # step names and durations
    assert model.steps[0].name == "sine"
    assert model.steps[1].name == "sweep"
    assert model.steps[2].name == "ramp"
    assert model.steps[0].duration == 5e-6

    # functions aggregation should include all analog + digital
    functions = model.functions
    assert len(functions) >= 4  # at least 3 analog + some digital


def test_sequence_model_step_labels_formatting():
    sequence, analog_map, digital_map = _build_demo_sequence()
    model = SequenceModel(sequence, analog_map=analog_map, digital_map=digital_map)

    labels = model.step_labels
    assert len(labels) == 3

    # label has name and duration; for repeated step, also total
    assert "sine" in labels[0]
    assert "sweep" in labels[1]
    assert "ramp" in labels[2]
    # "×10" style indication for repeats
    assert " × 10" in labels[2]


def test_function_block_model_parameters_and_description():
    sequence, analog_map, digital_map = _build_demo_sequence()
    model = SequenceModel(sequence, analog_map=analog_map, digital_map=digital_map)

    fb = model.functions[0]
    assert isinstance(fb, FunctionBlockModel)
    params = fb.parameters

    assert "pulse start time" in params
    assert isinstance(params["pulse start time"], (float, int))

    # description should contain the parameter names as lines
    desc = fb.parameters_description
    for name in params.keys():
        assert name in desc


def test_color_cycle_repeats_after_palette_length():
    palette_length = 10  # as defined in color_cycle
    seen = set(color_cycle(i) for i in range(palette_length))
    assert len(seen) == palette_length  # all distinct within one cycle

    # index beyond palette length should wrap
    assert color_cycle(0) == color_cycle(palette_length)
    assert color_cycle(1) == color_cycle(palette_length + 1)


def test_get_sequence_plot_data_source_consistency():
    sequence, analog_map, digital_map = _build_demo_sequence()
    model = SequenceModel(sequence, analog_map=analog_map, digital_map=digital_map)

    source = get_sequence_plot_data_source(model)

    required_keys = [
        "x_index",
        "is_analog",
        "channel_index",
        "center_time_ms",
        "y_index",
        "y_label",
        "name",
        "name_and_repeat",
        "tooltip",
        "fill_color",
        "duration_ms",
    ]
    for key in required_keys:
        assert key in source.data

    # all arrays same length and non-empty
    lengths = {len(source.data[key]) for key in required_keys}
    assert len(lengths) == 1
    (length,) = lengths
    assert length > 0

    # duration_ms must be positive
    assert np.all(np.array(source.data["duration_ms"]) > 0)
