"""Matched-workload comparison of day-ahead, periodic, and event-based charging."""
import time

import numpy as np
import pandas as pd

from evopt.data import load_sessions
from evopt.data.prices import PriceModel, load_price_days
from evopt.data.schema import Site
from evopt.learning.mixture import GaussianMixtureDemand, GaussianMixtureEstimator
from evopt.metrics import evaluate
from evopt.model import make_terms
from evopt.sim.compact_solver import solve_compact
from evopt.sim.learning_mpc import run_learning_mpc
from evopt.sim.plant import ChargingPlant


POLICIES = {"day_ahead": None, "hourly_mpc": None, "event_3": 3, "event_5": 5}


def run_comparison(cfg, log=print):
    site = Site(**cfg["site"])
    settings, exp = cfg["learning"], cfg["experiment"]
    truth = GaussianMixtureDemand(**settings["truth"])
    terms_cfg = cfg["model"]["terms"]
    terms = make_terms({k: v for k, v in terms_cfg.items() if k != "unmet_energy"})
    rows, traces, examples = [], [], []
    for scenario in range(exp.get("scenarios", 1)):
        rng = np.random.default_rng(exp.get("seed", 0) + scenario)
        train, test = load_price_days(cfg["prices"], rng)
        days = exp.get("days", 1)
        if len(test) < days or site.n_slots % train.shape[1]:
            raise ValueError("price data must cover days and divide the horizon")
        prices = PriceModel(train, site.n_slots // train.shape[1], cfg["prices"].get("shrinkage", 0.1))
        for day in range(days):
            sessions = load_sessions(cfg["data"], site, rng)
            demand, types = truth.sample_with_types(len(sessions), rng)
            realized = test[day]
            estimators = {name: GaussianMixtureEstimator(**settings["estimator"]) for name in POLICIES}
            schedules, counts, runtimes = {}, {}, {}
            for name, threshold in POLICIES.items():
                log(f"scenario {scenario}, day {day}: {name} starting")
                start = time.perf_counter()
                feedback, trace = run_learning_mpc(
                    sessions, site, prices, realized, terms, ChargingPlant(sessions, site, demand),
                    estimators[name], session_prefix=day,
                    replan_every_min=(site.horizon_hours * 60 if name == "day_ahead"
                                      else cfg["model"]["mpc"]["replan_every_min"]),
                    replan_every_completions=threshold,
                    gamma=terms_cfg.get("unmet_energy", {}).get("gamma", 1.0), solve_fn=solve_compact,
                    on_replan=lambda k: log(f"    optimized at hour {k * site.dt:.2f}"))
                runtimes[name] = time.perf_counter() - start
                counts[name] = int(trace.is_replan.sum())
                traces.append(trace.assign(scenario=scenario, day=day, method=name))
                schedules[name] = feedback.actual_kw
                log(f"  {name}: {counts[name]} solves, {runtimes[name]:.1f} s")
            for name, schedule in schedules.items():
                rows.append({"variant": "control_comparison", "scenario": scenario, "day": day,
                             "method": name, "solves": counts[name], "runtime_s": runtimes[name],
                             **evaluate(schedule, sessions, site, prices.to_slots(realized), demand)})
            examples.append({"scenario": scenario, "day": day, "site": site, "sessions": sessions,
                             "demand": demand, "car_types": types, "schedules": schedules,
                             "prices": prices.to_slots(realized)})
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), examples
