"""Run an experiment and write its results.

    python scripts/run_experiment.py configs/experiments/smoke.yaml

Writes metrics.csv, summary.csv, example_day.npz and figures to results/<name>/.
"""
import argparse
from pathlib import Path

import numpy as np

from evopt import viz
from evopt.config import load_config
from evopt.metrics import summarize
from evopt.sim.runner import run_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", help="path to an experiment YAML file")
    parser.add_argument("--out", default="results", help="output folder")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out = Path(args.out) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)

    metrics, example = run_experiment(cfg)
    summary = summarize(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    example["sessions"].to_csv(out / "example_sessions.csv", index=False)
    np.savez_compressed(out / "example_day.npz", realized_prices=example["realized_prices"],
                        realized_demand=example["realized_demand"], **example["schedules"])

    figures = {
        "fleet_load": viz.fleet_load(example),
        "schedule_heatmap": viz.schedule_heatmap(example),
        "power_distribution": viz.power_distribution(example),
        "cost_breakdown": viz.cost_breakdown(metrics),
        "cost_distribution": viz.cost_distribution(metrics),
        "risk_frontier": viz.risk_frontier(metrics),
    }
    if "mpc" in example["schedules"]:
        figures["plan_vs_replan"] = viz.plan_vs_replan(example)
    if metrics["variant"].nunique() > 1:
        figures["variant_comparison"] = viz.variant_comparison(metrics)
    for name, fig in figures.items():
        viz.save(fig, out / "figures", name)

    print(summary.round(2).to_string(index=False))
    print(f"\nResults written to {out}/")


if __name__ == "__main__":
    main()
