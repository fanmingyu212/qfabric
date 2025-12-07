import pickle
import sys

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.widgets import SpanSelector

from qfabric.visualizer.power_spectrum import PowerSpectrum

data = pickle.loads(sys.stdin.buffer.read())
func = data["function"]
is_analog = data["is_analog"]
channel_name = data["channel_name"]
step_name = data["step_name"]
duration = data["duration"]
sample_rate = data["sample_rate"]

xdata = np.arange(0, duration, 1 / sample_rate)
ydata = func.output(xdata)

if is_analog:
    fig, axs = plt.subplots(2, 1, figsize=(6, 8), num=func.__class__.__name__)
    ax = axs[0]
else:
    fig, ax = plt.subplots(figsize=(6, 4), num=func.__class__.__name__)

ax.plot(xdata, ydata)
ax.set_title(f'{channel_name} (step "{step_name}")')
ax.set_xlabel("time (s)")
ax.set_ylabel(channel_name)

if is_analog:
    ax = axs[1]
    ax.text(
        0.05,
        0.95,
        "Drag to select a range in the above panel\nto view its power spectrum.",
        fontsize=12,
        transform=ax.transAxes,
        verticalalignment="top",
        horizontalalignment="left",
    )


def on_span_select(xmin: float, xmax: float):
    ax = axs[1]
    ax.cla()
    mask = (xdata >= xmin) & (xdata < xmax)
    if sum(mask) == 0:
        return
    ps = PowerSpectrum(ydata[mask], 1 / sample_rate)
    ax.plot(ps.f, ps.power_spectrum, color="orange")
    ax.set_yscale("log")
    ax.set_xlabel("frequency (Hz)")
    ax.set_ylabel("power spectrum (V$^2$ / Hz)")
    plt.draw()


if is_analog:
    span = SpanSelector(
        axs[0],
        on_span_select,
        "horizontal",
        useblit=True,
        props=dict(alpha=0.3, facecolor="tab:orange"),
        interactive=True,
    )

plt.tight_layout()
try:
    plt.show()
except KeyboardInterrupt:
    pass
