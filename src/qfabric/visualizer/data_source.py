from bokeh.models import ColumnDataSource

from qfabric.visualizer.models import SequenceModel


def color_cycle(index: int) -> str:
    colors = [
        "#a1c9f4",
        "#ffb482",
        "#8de5a1",
        "#ff9f9b",
        "#d0bbff",
        "#debb9b",
        "#fab0e4",
        "#cfcfcf",
        "#fffea3",
        "#b9f2f0",
    ]  # Uses seaborn "pastel" palette
    return colors[index % len(colors)]


def get_sequence_plot_data_source(sequence_model: SequenceModel) -> ColumnDataSource:
    data = {}
    data["x_index"] = []
    data["center_time_ms"] = []
    data["y_index"] = []
    data["y_label"] = []
    data["name"] = []
    data["name_and_repeat"] = []
    data["tooltip"] = []
    data["fill_color"] = []
    data["duration_ms"] = []

    for function_block in sequence_model.functions:
        total_duration = function_block.duration * function_block.repeat
        data["x_index"].append(function_block.x_index)
        data["center_time_ms"].append((function_block.start_time + total_duration / 2) * 1e3)
        data["y_index"].append(function_block.y_index)
        data["y_label"].append(function_block.y_label)
        data["name"].append(function_block.name)
        if function_block.repeat == 1:
            data["name_and_repeat"].append(function_block.name)
        else:
            data["name_and_repeat"].append(
                function_block.name + "\u00d7" + str(function_block.repeat)
            )
        data["tooltip"].append(function_block.parameters_description)
        data["fill_color"].append(color_cycle(function_block.y_index))
        data["duration_ms"].append(total_duration * 1e3)

    return ColumnDataSource(data)
