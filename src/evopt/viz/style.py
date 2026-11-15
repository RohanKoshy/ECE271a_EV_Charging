"""Shared look: one style sheet, one fixed colour per method, and a save helper."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

plt.style.use(Path(__file__).with_name("evopt.mplstyle"))

# A method keeps its colour in every figure. New methods take the next spare colour.
METHOD_COLORS = {"mpc": "#2a78d6", "day_ahead": "#eb6834", "hindsight": "#1baf7a", "fifs": "#eda100"}
SPARE_COLORS = ["#e87ba4", "#008300", "#6250d6", "#e34948"]
METHOD_LABELS = {"mpc": "MPC", "day_ahead": "Day-ahead plan", "hindsight": "Hindsight optimum", "fifs": "FIFS"}
INK, MUTED = "#1a1a19", "#73726c"
POWER_CMAP = LinearSegmentedColormap.from_list("power", ["#ffffff", "#cde2fb", "#5598e7", "#0d366b"])


def color(method: str) -> str:
    if method not in METHOD_COLORS:
        METHOD_COLORS[method] = SPARE_COLORS[(len(METHOD_COLORS) - 4) % len(SPARE_COLORS)]
    return METHOD_COLORS[method]


def label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def clock_axis(ax, site):
    """Label a slot-indexed x axis in clock hours."""
    ax.set_xlim(0, site.horizon_hours)
    ax.set_xticks(np.arange(0, site.horizon_hours + 1, 4))
    ax.set_xticklabels([f"{h:02d}:00" for h in range(0, site.horizon_hours + 1, 4)])


def hours(site) -> np.ndarray:
    """Left edge of each slot, in hours."""
    return np.arange(site.n_slots) * site.dt


def save(fig, folder, name):
    """Save as PDF (for the report) and PNG (for the README)."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(folder / f"{name}.{ext}")
    plt.close(fig)
