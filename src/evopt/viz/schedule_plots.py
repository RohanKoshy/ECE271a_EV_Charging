"""Figures drawn from one example day: `example` is the dict returned by run_experiment."""
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator

from evopt.viz.style import INK, MUTED, POWER_CMAP, clock_axis, color, hours, label


def fleet_load(example):
    """Fleet power over the day for each method, with the day's price on top."""
    site, schedules = example["site"], example["schedules"]
    t = hours(site)
    fig, axes = plt.subplots(len(schedules) + 1, 1, figsize=(8, 1.9 * (len(schedules) + 1)), sharex=True)
    axes[0].step(t, example["realized_prices"], where="post", color=INK, linewidth=1.5)
    axes[0].set(title="Realized energy price", ylabel="USD/kWh")
    top = max(s.sum(axis=0).max() for s in schedules.values()) * 1.15
    for ax, (method, schedule) in zip(axes[1:], schedules.items()):
        fleet = schedule.sum(axis=0)
        ax.fill_between(t, fleet, step="post", color=color(method), alpha=0.25, linewidth=0)
        ax.step(t, fleet, where="post", color=color(method))
        peak = fleet.argmax()
        ax.plot(t[peak], fleet[peak], "o", color=color(method), markeredgecolor="white", markersize=7)
        ax.annotate(f"peak {fleet[peak]:.0f} kW", (t[peak], fleet[peak]), xytext=(6, 4),
                    textcoords="offset points", color=INK, fontsize=9)
        if site.capacity_kw <= top:
            ax.axhline(site.capacity_kw, color=MUTED, linestyle="--", linewidth=1)
        ax.set(title=label(method), ylabel="Fleet kW", ylim=(0, top))
    clock_axis(axes[-1], site)
    fig.tight_layout()
    return fig


def schedule_heatmap(example, method=None):
    """Who charges when: one row per EV (sorted by arrival), colour is charging power."""
    site, sessions = example["site"], example["sessions"]
    available = example["schedules"]
    method = method or next((m for m in ("mpc", "day_ahead") if m in available), next(iter(available)))
    schedule = example["schedules"][method]
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(schedule, aspect="auto", cmap=POWER_CMAP, interpolation="nearest",
                      extent=(0, site.horizon_hours, len(sessions), 0))
    for i, ev in enumerate(sessions.itertuples()):       # plug-in window of each EV
        ax.plot([ev.arrival * site.dt, ev.departure * site.dt], [i + 1, i + 1], color="#c3c2b7", linewidth=0.4)
    ax.grid(False)
    ax.set(title=f"Charging power per EV: {label(method)}", ylabel="EV (sorted by arrival)")
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    clock_axis(ax, site)
    fig.colorbar(image, ax=ax, label="kW", pad=0.02)
    fig.tight_layout()
    return fig


def plan_vs_replan(example):
    """Where MPC departs from the day-ahead plan, next to forecast and realized prices."""
    site, schedules = example["site"], example["schedules"]
    t = hours(site)
    fig, (top, bottom) = plt.subplots(2, 1, figsize=(8, 5), sharex=True)
    top.step(t, example["forecast_prices"], where="post", color=MUTED, linestyle="--", label="Day-ahead forecast")
    top.step(t, example["realized_prices"], where="post", color=INK, label="Realized")
    top.set(title="Energy price", ylabel="USD/kWh")
    top.legend(loc="upper left")
    for method in ("day_ahead", "mpc"):
        if method in schedules:
            bottom.step(t, schedules[method].sum(axis=0), where="post", color=color(method), label=label(method))
    bottom.set(title="Fleet power", ylabel="Fleet kW")
    bottom.legend(loc="upper left")
    clock_axis(bottom, site)
    fig.tight_layout()
    return fig


def power_distribution(example):
    """Share of delivered energy at or below each charging power, per method."""
    sessions, schedules = example["sessions"], example["schedules"]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for method, schedule in schedules.items():
        power = np.sort(schedule[schedule > 1e-6])
        ax.plot(power, 100 * np.cumsum(power) / power.sum(), color=color(method), label=label(method))
    safe = sessions["p_safe"].unique()
    if len(safe) == 1:
        ax.axvline(safe[0], color=MUTED, linestyle="--", linewidth=1)
        ax.annotate("safe power", (safe[0], 3), xytext=(4, 0), textcoords="offset points", color=MUTED, fontsize=9)
    ax.set(title="How hard the batteries are charged", xlabel="Charging power per EV (kW)",
           ylabel="Energy delivered at or below this power (%)", ylim=(0, 100))
    ax.legend(loc="upper left")
    fig.tight_layout()
    return fig
