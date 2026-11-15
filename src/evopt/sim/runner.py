"""Runs an experiment config: scenarios x price days x methods -> metrics table."""
import time

import numpy as np
import pandas as pd

from evopt import baselines as _baselines  # noqa: F401  (registers the baselines)
from evopt.config import with_overrides
from evopt.data import load_sessions
from evopt.data.demand import known_demand, sample_demand
from evopt.data.prices import PriceModel, load_price_days
from evopt.data.schema import Site
from evopt.metrics import evaluate
from evopt.model import make_terms
from evopt.registry import BASELINES
from evopt.sim.mpc import run_mpc
from evopt.sim.plan import day_ahead, hindsight


def run_experiment(cfg: dict, log=print):
    """Run every variant in the config. Returns (metrics DataFrame, example dict).

    A config without `variants` runs once. Every variant uses the same random
    seed, so variants are compared on identical sessions and price days.
    """
    variants = cfg.get("variants") or {"default": {}}
    frames, example = [], None
    for name, overrides in variants.items():
        log(f"variant: {name}")
        metrics, variant_example = run_variant(with_overrides(cfg, overrides), log)
        metrics.insert(0, "variant", name)
        frames.append(metrics)
        example = example or variant_example
    return pd.concat(frames, ignore_index=True), example


def run_variant(cfg: dict, log=print):
    exp = cfg["experiment"]
    rng = np.random.default_rng(exp.get("seed", 0))
    site = Site(**cfg["site"])
    train, test = load_price_days(cfg["prices"], rng)
    price_model = PriceModel(train, site.n_slots // train.shape[1], cfg["prices"].get("shrinkage", 0.1))
    terms = make_terms(cfg["model"]["terms"])
    methods = cfg.get("methods", ["day_ahead"])
    mpc_cfg = cfg["model"].get("mpc", {})

    rows, example = [], None
    for scenario in range(exp.get("scenarios", 1)):
        sessions = load_sessions(cfg["data"], site, rng)
        start = time.time()
        plan = day_ahead(sessions, site, price_model, terms)
        same_every_day = {"day_ahead": plan} if "day_ahead" in methods else {}

        # Each simulated day draws a price day and every EV's true (Gaussian) demand.
        days = exp.get("days", exp.get("price_days", 1))
        for day, realized in enumerate(test[:days]):
            demand = sample_demand(sessions, rng)
            schedules = dict(same_every_day)
            # Baselines react to the car: they keep charging until it is actually full.
            for name in cfg.get("baselines", []):
                schedules[name] = BASELINES.create(name).schedule(known_demand(sessions, demand), site)
            if "mpc" in methods:
                schedules["mpc"] = run_mpc(sessions, site, price_model, realized, terms,
                                           initial_plan=plan, realized_demand=demand, **mpc_cfg)
            realized_slots = price_model.to_slots(realized)
            if "hindsight" in methods:
                schedules["hindsight"] = hindsight(sessions, site, realized_slots, terms, demand)
            for method, schedule in schedules.items():
                rows.append({"scenario": scenario, "day": day, "method": method,
                             **evaluate(schedule, sessions, site, realized_slots, demand)})
            if example is None:
                example = {"sessions": sessions, "site": site, "schedules": schedules,
                           "realized_prices": realized_slots, "realized_demand": demand,
                           "forecast_prices": price_model.forecast().mean}
        log(f"  scenario {scenario}: {len(sessions)} sessions, {time.time() - start:.1f} s")
    return pd.DataFrame(rows), example
