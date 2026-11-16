"""Standalone experiment runner for the Gaussian-learning alternative."""
import numpy as np
import pandas as pd

from evopt.data import load_sessions
from evopt.data.prices import PriceModel, load_price_days
from evopt.data.schema import Site
from evopt.learning.gaussian import GaussianDemand, GaussianEstimator
from evopt.learning.mixture import GaussianMixtureDemand, GaussianMixtureEstimator
from evopt.metrics import evaluate
from evopt.model import make_terms
from evopt.sim.learning_mpc import run_learning_mpc
from evopt.sim.plant import ChargingPlant


def run_learning_experiment(cfg, log=print):
    exp, settings = cfg["experiment"], cfg["learning"]
    mixture = "components" in settings["truth"]
    truth = (GaussianMixtureDemand(**settings["truth"]) if mixture
             else GaussianDemand(**settings["truth"]))
    estimator_class = GaussianMixtureEstimator if mixture else GaussianEstimator
    methods = cfg.get("methods", ["learning_mpc", "fixed_mpc", "oracle_mpc"])
    if not methods or set(methods) - {"learning_mpc", "fixed_mpc", "oracle_mpc"}:
        raise ValueError("methods must select learning_mpc, fixed_mpc, or oracle_mpc")
    terms_cfg = cfg["model"]["terms"]
    unmet = terms_cfg.get("unmet_energy", {})
    if unmet.get("mode", "expected") != "expected":
        raise ValueError("learning experiments require expected-shortfall mode")
    terms = make_terms({k: v for k, v in terms_cfg.items() if k != "unmet_energy"})
    site = Site(**cfg["site"])
    rows, traces, example = [], [], None
    for scenario in range(exp.get("scenarios", 1)):
        rng = np.random.default_rng(exp.get("seed", 0) + scenario)
        train, test = load_price_days(cfg["prices"], rng)
        days = exp.get("days", 1)
        if len(test) < days:
            raise ValueError("not enough held-out price days")
        if site.n_slots % train.shape[1]:
            raise ValueError("price periods must divide the site horizon")
        prices = PriceModel(train, site.n_slots // train.shape[1], cfg["prices"].get("shrinkage", 0.1))
        estimators = {name: estimator_class(**settings.get("estimator", {})) for name in methods}
        if "oracle_mpc" in estimators:
            estimators["oracle_mpc"].model = truth
        for day in range(days):
            sessions = load_sessions(cfg["data"], site, rng)
            if mixture:
                demand, car_types = truth.sample_with_types(len(sessions), rng)
            else:
                demand, car_types = truth.sample(len(sessions), rng), np.zeros(len(sessions), dtype=int)
            realized = test[day]
            schedules = {}
            for method in methods:
                plant = ChargingPlant(sessions, site, demand)
                feedback, trace = run_learning_mpc(
                    sessions, site, prices, realized, terms, plant, estimators[method],
                    session_prefix=day, gamma=unmet.get("gamma", 1.0),
                    learn=method == "learning_mpc", **cfg["model"].get("mpc", {}))
                schedules[method] = feedback.actual_kw
                rows.append({"variant": "mixture_learning" if mixture else "gaussian_learning", "scenario": scenario,
                             "day": day, "method": method,
                             **evaluate(feedback.actual_kw, sessions, site, prices.to_slots(realized), demand)})
                trace = trace.assign(scenario=scenario, day=day, method=method,
                                     true_mean=truth.mean, true_variance=truth.sd ** 2)
                traces.append(trace)
            if example is None:
                example = {"sessions": sessions, "site": site, "schedules": schedules,
                           "realized_prices": prices.to_slots(realized), "realized_demand": demand,
                           "true_car_types": car_types,
                           "forecast_prices": prices.forecast().mean}
            log(f"scenario {scenario}, day {day}: {len(sessions)} sessions")
    return pd.DataFrame(rows), pd.concat(traces, ignore_index=True), example
