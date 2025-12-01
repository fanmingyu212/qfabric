import os
import pickle
import subprocess
import sys

import numpy as np
from bokeh.layouts import column
from bokeh.models import HoverTool, NumericInput, Range1d, TapTool
from bokeh.plotting import figure
from bokeh.server.server import Server

from qfabric.sequence.sequence import Sequence
from qfabric.visualizer.data_source import get_sequence_plot_data_source
from qfabric.visualizer.models import SequenceModel


def logic_sequence(sequence: Sequence):
    """
    Displays the sequence in the logic mode.

    Each step is depicted as equal length. Clicking on each pulse
    shows the details of the pulse.

    If run in Jupyter notebook, clicking on functions in the sequence
    will not show the details.
    """

    def make_doc(doc):
        sequence_model = SequenceModel(sequence)

        plot_func_process = None
        step_labels = sequence_model.step_labels

        ch_labels = list(reversed(sequence_model.channel_names))
        width = len(step_labels) * 180
        height = len(ch_labels) * 80

        spin = NumericInput(
            value=625, low=1e-3, high=10000, mode="float", title="Sample rate (MHz)"
        )

        plot_logic = figure(
            tools="xpan, xwheel_zoom, reset, save", y_range=ch_labels, width=width, height=height
        )
        source_logic = get_sequence_plot_data_source(sequence_model)
        rectangles = plot_logic.rect(
            x="x_index",
            y="y_label",
            fill_color="fill_color",
            width=1,
            height=0.8,
            source=source_logic,
        )
        rectangles.nonselection_glyph.fill_alpha = 0.9
        plot_logic.text(
            x="x_index",
            y="y_label",
            text="name",
            text_color="#313131",
            text_align="center",
            text_baseline="middle",
            source=source_logic,
        )
        tooltips = """
        <div>
            <span style="font-size: 12px;"><b>Parameters:</b><br></span>
            <span style="font-size: 12px;">@tooltip{safe}</span>
        </div>
        """

        hover_tool = HoverTool(tooltips=tooltips)
        plot_logic.add_tools(hover_tool)

        plot_logic.x_range = Range1d(-0.6, len(step_labels) - 0.4)
        plot_logic.xaxis.ticker = np.arange(len(step_labels))
        plot_logic.xaxis.major_label_overrides = dict(zip(np.arange(len(step_labels)), step_labels))

        plot_logic.xaxis.axis_label_text_font_size = "14px"
        plot_logic.xaxis.major_label_text_font_size = "14px"
        plot_logic.yaxis.axis_label_text_font_size = "14px"
        plot_logic.yaxis.major_label_text_font_size = "14px"

        plot_logic.toolbar.active_drag = None
        plot_logic.toolbar.active_scroll = None

        taptool = TapTool()
        plot_logic.add_tools(taptool)
        plot_logic.toolbar.active_tap = taptool

        def open_new_plot_matplotlib(attr, old, new):
            # Uses matplotlib for handling lots of data points.
            # Opens matplotlib in a new process to allow interactive control.

            nonlocal plot_func_process
            if plot_func_process is not None:
                plot_func_process.terminate()

            inds = source_logic.selected.indices
            if len(inds) == 0:
                return

            step_index = source_logic.data["x_index"][inds[0]]
            channel_name = source_logic.data["y_label"][inds[0]]
            step = sequence_model.steps[step_index]
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
                "sample_rate": spin.value * 1e6,
            }
            plot_func_process = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpl_viewer.py"),
                ],
                stdin=subprocess.PIPE,
            )
            plot_func_process.stdin.write(pickle.dumps(data))
            plot_func_process.stdin.close()

        source_logic.selected.on_change("indices", open_new_plot_matplotlib)

        doc.add_root(column(spin, plot_logic))

    apps = {"/": make_doc}

    server = Server(apps)
    server.start()
    print(f"Sequence displayed at http://localhost:{server.port}")
    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass


def timeline_sequence(sequence: Sequence):
    """
    Displays the sequence in the timeline mode.

    Each step is depicted with width proportional to duration.
    Clicking on each pulse shows the details of the pulse.

    If run in Jupyter notebook, clicking on functions in the sequence
    will not show the details.
    """

    def make_doc(doc):
        sequence_model = SequenceModel(sequence)

        plot_func_process = None
        step_labels = sequence_model.step_labels

        ch_labels = list(reversed(sequence_model.channel_names))
        width = len(step_labels) * 180
        height = len(ch_labels) * 80

        spin = NumericInput(
            value=625, low=1e-3, high=10000, mode="float", title="Sample rate (MHz)"
        )

        plot_logic = figure(
            tools="xpan, xwheel_zoom, reset, save", y_range=ch_labels, width=width, height=height
        )
        source_logic = get_sequence_plot_data_source(sequence_model)
        rectangles = plot_logic.rect(
            x="center_time_ms",
            y="y_label",
            fill_color="fill_color",
            width="duration_ms",
            height=0.8,
            source=source_logic,
        )
        rectangles.nonselection_glyph.fill_alpha = 0.9
        plot_logic.text(
            x="center_time_ms",
            y="y_label",
            text="name_and_repeat",
            text_color="#313131",
            text_align="center",
            text_baseline="middle",
            source=source_logic,
        )
        tooltips = """
        <div>
            <span style="font-size: 12px;"><b>Parameters:</b><br></span>
            <span style="font-size: 12px;">@tooltip{safe}</span>
        </div>
        """

        hover_tool = HoverTool(tooltips=tooltips)
        plot_logic.add_tools(hover_tool)

        plot_logic.xaxis.axis_label = "Time (ms)"

        plot_logic.xaxis.axis_label_text_font_size = "14px"
        plot_logic.xaxis.major_label_text_font_size = "14px"
        plot_logic.yaxis.axis_label_text_font_size = "14px"
        plot_logic.yaxis.major_label_text_font_size = "14px"

        plot_logic.toolbar.active_drag = None
        plot_logic.toolbar.active_scroll = None

        taptool = TapTool()
        plot_logic.add_tools(taptool)
        plot_logic.toolbar.active_tap = taptool

        def open_new_plot_matplotlib(attr, old, new):
            # Uses matplotlib for handling lots of data points.
            # Opens matplotlib in a new process to allow interactive control.

            nonlocal plot_func_process
            if plot_func_process is not None:
                plot_func_process.terminate()

            inds = source_logic.selected.indices
            if len(inds) == 0:
                return

            step_index = source_logic.data["x_index"][inds[0]]
            channel_name = source_logic.data["y_label"][inds[0]]
            step = sequence_model.steps[step_index]
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
                "sample_rate": spin.value * 1e6,
            }
            plot_func_process = subprocess.Popen(
                [
                    sys.executable,
                    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mpl_viewer.py"),
                ],
                stdin=subprocess.PIPE,
            )
            plot_func_process.stdin.write(pickle.dumps(data))
            plot_func_process.stdin.close()

        source_logic.selected.on_change("indices", open_new_plot_matplotlib)

        doc.add_root(column(spin, plot_logic))

    apps = {"/": make_doc}

    server = Server(apps)
    server.start()
    print(f"Sequence displayed at http://localhost:{server.port}")
    server.io_loop.add_callback(server.show, "/")
    try:
        server.io_loop.start()
    except KeyboardInterrupt:
        pass
