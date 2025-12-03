import os
import pickle
import subprocess
import sys
from functools import partial

import numpy as np
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, HoverTool, NumericInput, Range1d, TapTool
from bokeh.plotting import figure
from bokeh.server.server import Server

from qfabric.sequence.sequence import Sequence
from qfabric.visualizer.data_source import get_sequence_plot_data_source
from qfabric.visualizer.models import SequenceModel

sequence_plot_tools = "xpan, xwheel_zoom, reset, save"


class _MPLParameters:
    """
    This class is only for pass objects by reference.

    Pass this object as an argument and set :attr:`process`.
    """

    def __init__(self, model: SequenceModel, source: ColumnDataSource, spin: NumericInput):
        self.model = model
        self.source = source
        self.spin = spin
        self.process: subprocess.Popen = None


def _get_sequence_figure(sequence_model: SequenceModel, logic: bool):
    step_labels = sequence_model.step_labels
    ch_labels = list(reversed(sequence_model.channel_names))
    width = len(step_labels) * 180
    height = len(ch_labels) * 80
    spin = NumericInput(value=625, low=1e-3, high=10000, mode="float", title="Sample rate (MHz)")
    plot = figure(tools=sequence_plot_tools, y_range=ch_labels, width=width, height=height)

    source = get_sequence_plot_data_source(sequence_model)

    rect_kwargs = {"y": "y_label", "fill_color": "fill_color", "height": 0.8, "source": source}
    text_kwargs = {
        "y": "y_label",
        "text_color": "#313131",
        "text_align": "center",
        "text_baseline": "middle",
        "source": source,
    }

    if logic:
        rectangles = plot.rect(x="x_index", width=1, **rect_kwargs)
        plot.text(x="x_index", text="name", **text_kwargs)

        num_steps = len(step_labels)
        plot.x_range = Range1d(-0.6, num_steps - 0.4)
        plot.xaxis.ticker = np.arange(num_steps)
        plot.xaxis.major_label_overrides = dict(zip(np.arange(num_steps), step_labels))
    else:
        rectangles = plot.rect(x="center_time_ms", width="duration_ms", **rect_kwargs)
        plot.text(x="center_time_ms", text="name_and_repeat", **text_kwargs)
        plot.xaxis.axis_label = "Time (ms)"

    rectangles.nonselection_glyph.fill_alpha = 0.5
    tooltips = """
    <div>
        <span style="font-size: 12px;"><b>@name</b><br></span>
        <span style="font-size: 12px;"><i>Parameters:</i><br></span>
        <span style="font-size: 12px;">@tooltip{safe}</span>
    </div>
    """

    hover_tool = HoverTool(tooltips=tooltips)
    plot.add_tools(hover_tool)

    plot.xaxis.axis_label_text_font_size = "14px"
    plot.xaxis.major_label_text_font_size = "14px"
    plot.yaxis.axis_label_text_font_size = "14px"
    plot.yaxis.major_label_text_font_size = "14px"

    plot.toolbar.active_drag = None
    plot.toolbar.active_scroll = None

    taptool = TapTool()
    plot.add_tools(taptool)
    plot.toolbar.active_tap = taptool

    mpl_parameters = _MPLParameters(sequence_model, source, spin)
    source.selected.on_change("indices", partial(_open_new_plot_mpl, mpl_parameters))

    return (plot, source, spin)


def _open_new_plot_mpl(parameters: _MPLParameters, attr, old, new):
    """
    Opens plot showing function details in matplotlib.

    Matplotlib is used over bokeh for its capability to handle large datasets.
    Matplotlib is opened in a new process so it can run its own event loop
    to enable interactive features.
    """
    model = parameters.model
    source = parameters.source
    if parameters.process is not None:
        parameters.process.terminate()
        parameters.process = None

    inds = source.selected.indices
    if len(inds) == 0:
        return

    step_index = source.data["x_index"][inds[0]]
    channel_name = source.data["y_label"][inds[0]]
    step = model.steps[step_index]
    if channel_name.startswith("Analog"):
        channel_index = int(channel_name[9:])
        function = step.analog_functions[channel_index].func
        is_analog = True
    else:
        channel_index = int(channel_name[10:])
        function = step.digital_functions[channel_index].func
        is_analog = False

    data = {
        "function": function,
        "is_analog": is_analog,
        "channel_name": channel_name,
        "step_name": step.name,
        "duration": step.duration,
        "sample_rate": parameters.spin.value * 1e6,
    }
    plot_func_process = subprocess.Popen(
        [
            sys.executable,
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpl_viewer.py"),
        ],
        stdin=subprocess.PIPE,
    )
    parameters.process = plot_func_process
    plot_func_process.stdin.write(pickle.dumps(data))
    plot_func_process.stdin.close()


def _start_server(apps):
    server = Server(apps)
    server.start()
    print(f"Sequence displayed at http://localhost:{server.port}")
    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass


def logic_sequence(sequence: Sequence):
    """
    Displays the sequence in the logic mode.

    Each step is depicted as equal length. Click on each pulse
    to see details of the pulse.

    If run in Jupyter notebook or other python processes with an event loop,
    clicking on functions in the sequence will not show the details.

    Args:
        sequence (Sequence): sequence to be displayed.
    """

    def make_doc(doc):
        sequence_model = SequenceModel(sequence)
        plot, source, spin = _get_sequence_figure(sequence_model, logic=True)
        doc.add_root(column(spin, plot))

    apps = {"/": make_doc}
    _start_server(apps)


def timeline_sequence(sequence: Sequence):
    """
    Displays the sequence in the timeline mode.

    Each step is depicted with width proportional to duration.
    Click on each pulse to see details of the pulse.

    If run in Jupyter notebook or other python processes with an event loop,
    clicking on functions in the sequence will not show the details.

    Args:
        sequence (Sequence): sequence to be displayed.
    """

    def make_doc(doc):
        sequence_model = SequenceModel(sequence)
        plot, source, spin = _get_sequence_figure(sequence_model, logic=False)
        doc.add_root(column(spin, plot))

    apps = {"/": make_doc}
    _start_server(apps)
