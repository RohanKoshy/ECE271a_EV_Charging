"""Run the ideal-plant learner with a completion-count replanning trigger."""
import argparse
from pathlib import Path

from evopt.config import load_config
from evopt.metrics import summarize
from evopt.sim.learning_runner import run_learning_experiment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", nargs="?", default="configs/experiments/mixture_learning.yaml")
    parser.add_argument("--out", default="results")
    parser.add_argument("--replan-every-completions", type=int, choices=[3, 5, 10])
    args = parser.parse_args()
    count = args.replan_every_completions
    if count is None:
        try:
            count = int(input("Re-optimize after how many newly completed sessions? [3/5/10]: "))
        except (ValueError, EOFError):
            parser.error("enter 3, 5, or 10, or pass --replan-every-completions")
        if count not in (3, 5, 10):
            parser.error("completion count must be 3, 5, or 10")
    cfg = load_config(args.config)
    cfg["model"]["mpc"]["replan_every_completions"] = count
    metrics, trace, _ = run_learning_experiment(cfg, log=lambda message: print(message, flush=True))
    out = Path(args.out) / cfg["experiment"]["name"]
    out.mkdir(parents=True, exist_ok=True)
    summary = summarize(metrics)
    metrics.to_csv(out / "metrics.csv", index=False)
    summary.to_csv(out / "summary.csv", index=False)
    trace.to_csv(out / "learning_trace.csv", index=False)
    print(summary.round(2).to_string(index=False))
    print(f"Results written to {out}/")


if __name__ == "__main__":
    main()
