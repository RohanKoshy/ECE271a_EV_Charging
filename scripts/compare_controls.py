"""Run matched control policies and save four fleet-load figures."""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml

from evopt.config import load_config
from evopt.metrics import summarize
from evopt.sim.control_comparison import run_comparison


LABELS = {"day_ahead": "Day-ahead: no replanning", "hourly_mpc": "MPC: hourly replanning",
          "event_3": "MPC: every 3 completions", "event_5": "MPC: every 5 completions"}
COLORS = {"day_ahead": "#d55e00", "hourly_mpc": "#0072b2", "event_3": "#009e73", "event_5": "#9467bd"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments/mixture_learning.yaml")
    parser.add_argument("--out", default="results/mpc_comparison")
    args = parser.parse_args()
    cfg = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.yaml").write_text(yaml.safe_dump(cfg))
    metrics, trace, examples = run_comparison(cfg, log=lambda message: print(message, flush=True))
    metrics.to_csv(out / "metrics.csv", index=False)
    trace.to_csv(out / "learning_trace.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(out / "summary.csv", index=False)
    for example in examples:
        prefix = f"scenario_{example['scenario']}_day_{example['day']}"
        example["sessions"].to_csv(out / f"{prefix}_sessions.csv", index=False)
        np.savez_compressed(out / f"{prefix}.npz", demand=example["demand"],
                            car_types=example["car_types"], prices=example["prices"], **example["schedules"])
    site = examples[0]["site"]
    hours = np.arange(site.n_slots) * site.dt
    ymax = max(s.sum(axis=0).max() for ex in examples for s in ex["schedules"].values()) * 1.08
    for name, label in LABELS.items():
        fig, axes = plt.subplots(len(examples), 1, figsize=(10, 2.8 * len(examples)),
                                 sharex=True, sharey=True, squeeze=False)
        for ax, example in zip(axes[:, 0], examples):
            fleet = example["schedules"][name].sum(axis=0)
            ax.step(hours, fleet, where="post", color=COLORS[name], linewidth=1.5)
            ax.fill_between(hours, fleet, step="post", color=COLORS[name], alpha=0.12)
            ax.axhline(site.capacity_kw, color="#444444", linestyle="--", linewidth=1, label="Site capacity")
            replans = trace[(trace.method == name) & (trace.scenario == example["scenario"])
                            & (trace.day == example["day"]) & trace.is_replan]
            ax.plot(replans.slot * site.dt, np.full(len(replans), ymax * .96), "|",
                    color=COLORS[name], markersize=7, label="Optimizer calls")
            row = metrics[(metrics.method == name) & (metrics.scenario == example["scenario"])
                          & (metrics.day == example["day"])].iloc[0]
            ax.set(title=f"Day {example['day'] + 1}: ${row.total_cost_usd:.2f}, "
                         f"{row.demand_met_pct:.1f}% demand met, {int(row.solves)} solves",
                   ylabel="Fleet power (kW)", ylim=(0, ymax), xlim=(0, site.horizon_hours))
            ax.grid(alpha=.2)
        axes[0, 0].legend(loc="upper right", fontsize=8)
        axes[-1, 0].set(xlabel="Hour of day", xticks=np.arange(0, 25, 2))
        fig.suptitle(label, fontsize=15)
        fig.tight_layout(rect=(0, 0, 1, .96))
        fig.savefig(out / f"{name}.png", dpi=160)
        fig.savefig(out / f"{name}.pdf")
        plt.close(fig)
    print(summary.round(2).to_string(index=False))
    print(f"Results: {out}")


if __name__ == "__main__":
    main()
