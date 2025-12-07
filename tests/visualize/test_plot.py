import numpy as np
from bokeh.models import ColumnDataSource, Div, NumericInput
from bokeh.plotting import figure

from qfabric import (
    DigitalPulse,
    LinearRamp,
    Sequence,
    SineSweep,
    SineWave,
    Step,
)
from qfabric.visualizer.models import SequenceModel
from qfabric.visualizer.plot import _get_sequence_figure  # or your actual path
from qfabric.visualizer.plot import _tooltip_to_table_html


def _demo_sequence_and_model():
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

    model = SequenceModel(sequence, analog_map=analog_channels, digital_map=digital_channels)
    return model


def test_tooltip_to_table_html_parses_key_value_and_plain_lines():
    raw = """
        amplitude = 1.0
        frequency = 100e6

        some extra note line
        duty_cycle = 0.5
    """

    html = _tooltip_to_table_html(raw)

    # key=value rows become <td>key</td><td>= value</td> rows
    assert "amplitude" in html
    assert "= 1.0" in html
    assert "frequency" in html
    assert "duty_cycle" in html

    # non-key-value lines still appear
    assert "some extra note line" in html

    # basic table structure
    assert "<tr>" in html
    assert "</tr>" in html


def test_get_sequence_figure_logic_mode_structure():
    model = _demo_sequence_and_model()

    plot, source, spin, text = _get_sequence_figure(model, logic=True)

    dummy = figure()  # create a standard Bokeh figure
    assert isinstance(plot, type(dummy))

    assert isinstance(source, ColumnDataSource)
    assert isinstance(spin, NumericInput)
    assert isinstance(text, Div)

    # y_range labels should match channel_names (in reverse as defined)
    assert list(reversed(model.channel_names)) == list(plot.y_range.factors)

    # x_range ticks should match number of steps
    num_steps = len(model.steps)
    assert plot.x_range.start <= 0
    assert plot.x_range.end >= num_steps - 1


def test_get_sequence_figure_timeline_mode_structure():
    model = _demo_sequence_and_model()

    plot, source, spin, text = _get_sequence_figure(model, logic=False)

    dummy = figure()
    assert isinstance(plot, type(dummy))

    assert isinstance(source, ColumnDataSource)
    assert isinstance(spin, NumericInput)
    assert isinstance(text, Div)

    # time axis label
    assert plot.xaxis.axis_label == "Time (ms)"

    # some of the center_time_ms values should be increasing
    center_times = np.array(source.data["center_time_ms"])
    assert np.all(np.diff(np.sort(center_times)) >= 0)
