"""Figures drawn from the metrics table (one row per variant, scenario, price day, method)."""
import matplotlib.pyplot as plt
import numpy as np

from evopt.metrics import summarize
from evopt.viz.style import INK, color, label

COMPARED = {"total_cost_usd": "Total cost (USD)", "peak_kw": "Peak (kW)",
            "kwh_above_safe": "Energy above safe power (kWh)", "demand_met_pct": "Demand met (%)"}


def _name(row, many_variants, many_methods):
    parts = ([row["variant"]] if many_variants else []) + ([label(row["method"])] if many_methods else [])
    return "\n".join(parts) or label(row["method"])


def _flags(summary):
    return summary["variant"].nunique() > 1, summary["method"].nunique() > 1


def cost_breakdown(metrics):
    """Average daily cost split into energy and demand charge."""
    summary = summarize(metrics)
    names = [_name(row, *_flags(summary)) for _, row in summary.iterrows()]
    x = np.arange(len(summary))
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(x) + 2), 4))
    energy, demand = summary["energy_cost_usd"], summary["demand_charge_usd"]
    ax.bar(x, energy, width=0.6, color="#2a78d6", edgecolor="white", linewidth=2, label="Energy")
    ax.bar(x, demand, width=0.6, bottom=energy, color="#eb6834", edgecolor="white", linewidth=2, label="Demand charge")
    for xi, total in zip(x, energy + demand):
        ax.annotate(f"{total:,.0f}", (xi, total), xytext=(0, 3), textcoords="offset points",
                    ha="center", color=INK, fontsize=9)
    ax.set_xticks(x, names)
    ax.grid(axis="x", visible=False)
    ax.set(title="Average daily cost", ylabel="USD per day")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    fig.tight_layout()
    return fig


def risk_frontier(metrics):
    """Average cost against its day-to-day spread: lower-left is better."""
    summary = summarize(metrics)
    many_variants = summary["variant"].nunique() > 1
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for method, group in summary.groupby("method", sort=False):
        x, y = group["total_cost_std_usd"].to_numpy(), group["total_cost_usd"].to_numpy()
        same_everywhere = np.allclose(x, x[0]) and np.allclose(y, y[0])   # e.g. a baseline
        if same_everywhere:
            x, y = x[:1], y[:1]
        ax.plot(x, y, "o-", color=color(method), markersize=9, markeredgecolor="white",
                markeredgewidth=2, linewidth=1.2, label=label(method))
        if many_variants and not same_everywhere:
            for i, (xi, yi, variant) in enumerate(zip(x, y, group["variant"])):
                ax.annotate(variant, (xi, yi), xytext=(0, 10 if i % 2 == 0 else -16),
                            textcoords="offset points", ha="center", color=INK, fontsize=8)
    ax.margins(0.2)
    ax.set(title="Cost against cost risk", xlabel="Std of daily cost across price days (USD)",
           ylabel="Average daily cost (USD)")
    ax.legend(loc="best")
    fig.tight_layout()
    return fig


def cost_distribution(metrics):
    """Daily total cost across scenarios and price days, per method."""
    groups = list(metrics.groupby(["variant", "method"], sort=False))
    many_variants = metrics["variant"].nunique() > 1
    many_methods = metrics["method"].nunique() > 1
    fig, ax = plt.subplots(figsize=(max(6, 1.1 * len(groups) + 2), 4))
    for x, ((variant, method), group) in enumerate(groups):
        ax.boxplot(group["total_cost_usd"], positions=[x], widths=0.5, patch_artist=True,
                   boxprops={"facecolor": color(method), "alpha": 0.35, "edgecolor": color(method)},
                   medianprops={"color": color(method), "linewidth": 2},
                   whiskerprops={"color": color(method)}, capprops={"color": color(method)},
                   flierprops={"markeredgecolor": color(method), "markersize": 4})
    names = [_name({"variant": v, "method": m}, many_variants, many_methods) for (v, m), _ in groups]
    ax.set_xticks(range(len(groups)), names)
    ax.grid(axis="x", visible=False)
    ax.set(title="Daily cost across simulated days", ylabel="USD per day")
    fig.tight_layout()
    return fig


def variant_comparison(metrics):
    """Key metrics across variants (ablations and sweeps), one panel per metric."""
    summary = summarize(metrics)
    variants = list(summary["variant"].unique())
    fig, axes = plt.subplots(1, len(COMPARED), figsize=(3.2 * len(COMPARED), 3.6))
    for ax, (column, title) in zip(axes, COMPARED.items()):
        for method, group in summary.groupby("method", sort=False):
            x = [variants.index(v) for v in group["variant"]]
            ax.plot(x, group[column], "o-", color=color(method), markeredgecolor="white",
                    markersize=7, label=label(method))
        ax.set_xticks(range(len(variants)), variants, rotation=30, ha="right")
        ax.set_title(title)
        ax.ticklabel_format(axis="y", useOffset=False)
        if column == "demand_met_pct":
            ax.set_ylim(0, 105)
    axes[0].legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig
