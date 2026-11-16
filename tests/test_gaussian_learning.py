import numpy as np
import pandas as pd
import pytest
from scipy.integrate import quad
from scipy.stats import truncnorm

from evopt.data.prices import PriceModel
from evopt.data.schema import Site
from evopt.learning.gaussian import GaussianDemand, GaussianEstimator
from evopt.model import make_terms
from evopt.sim.learning_mpc import run_learning_mpc
from evopt.sim.learning_runner import run_learning_experiment
from evopt.sim.plant import ChargingPlant


def test_estimator_recovers_truncated_parameters_with_censoring():
    truth = GaussianDemand(8., 5.)
    values = truth.sample(5000, np.random.default_rng(3))
    estimator = GaussianEstimator(initial_mean=4., initial_sd=8.)
    for i, value in enumerate(values):
        estimator.observe(i, min(value, 10.), value <= 10.)
    estimate = estimator.fit()
    assert estimator.fit_success
    assert abs(estimate.mean - truth.mean) < 0.5
    assert abs(estimate.sd - truth.sd) < 0.5


def test_cumulative_observations_are_not_independent_samples():
    estimator = GaussianEstimator()
    estimator.observe("a", 2., False)
    estimator.observe("a", 5., False)
    estimator.observe("a", 7., True)
    assert estimator.records == {"a": (7., True)}
    with pytest.raises(ValueError):
        estimator.observe("a", 8., True)


def test_shortfall_matches_conditional_integral():
    model = GaussianDemand(2., 4.)
    for lower, delivered in [(0., 0.), (0., 5.), (8., 9.), (8., 3.)]:
        a = (lower - model.mean) / model.sd
        expected = quad(lambda r: max(r - delivered, 0.) * truncnorm.pdf(
            r, a, np.inf, loc=model.mean, scale=model.sd), lower, np.inf)[0]
        actual, _ = model.shortfall(delivered, lower)
        assert np.isclose(actual, expected, atol=1e-6)


def test_mpc_learns_only_from_executed_observations():
    site = Site(slot_minutes=60, horizon_hours=4, capacity_kw=10)
    sessions = pd.DataFrame({"session_id": [0, 1], "arrival": [0, 0], "departure": [4, 4],
                             "energy_requested": [999., 999.], "max_rate": [5., 5.], "p_safe": [5., 5.]})
    price_model = PriceModel(np.full((10, 4), 0.1), 1)
    estimator = GaussianEstimator(initial_mean=6., initial_sd=2.)
    terms = make_terms({"energy_cost": {}, "current_limit": {}, "equal_sharing": {}})
    plant = ChargingPlant(sessions, site, [2., 30.])
    feedback, trace = run_learning_mpc(sessions, site, price_model, np.full(4, 0.1),
                                      terms, plant, estimator, gamma=5.)
    assert trace.iloc[0]["observations"] == 0
    assert estimator.records[("day", 0)] == (pytest.approx(2.), True)
    assert estimator.records[("day", 1)][1] is False
    assert estimator.records[("day", 1)][0] == pytest.approx(feedback.delivered_kwh[1])
    assert (feedback.actual_kw.sum(axis=0) <= site.capacity_kw + 1e-6).all()
    assert feedback.delivered_kwh[1] <= 20. + 1e-6
    assert np.isfinite(trace["mean"]).all()


def test_runner_wires_alternative_without_changing_existing_config(cfg):
    cfg["experiment"].update(days=2, scenarios=1)
    cfg["data"]["synthetic"]["n_sessions"] = 3
    cfg["model"]["mpc"]["replan_every_min"] = 240
    cfg["methods"] = ["learning_mpc", "fixed_mpc", "oracle_mpc"]
    cfg["learning"] = {"truth": {"mean": 10., "sd": 3.},
                       "estimator": {"initial_mean": 6., "initial_sd": 4.}}
    metrics, trace, example = run_learning_experiment(cfg, log=lambda _: None)
    assert len(metrics) == 6
    assert set(example["schedules"]) == set(cfg["methods"])
    for method, mean in [("fixed_mpc", 6.), ("oracle_mpc", 10.)]:
        assert (trace.loc[trace.method == method, "mean"] == mean).all()
    adaptive = trace.loc[trace.method == "learning_mpc"]
    assert adaptive.loc[adaptive.day == 1, "observations"].min() > 0
    assert metrics.capacity_violation_kw.max() < 1e-4


def test_shortfall_epigraph_matches_scalar_tangents():
    from types import SimpleNamespace
    import cvxpy as cp
    from evopt.sim.learning_mpc import LearnedShortfall

    model = GaussianDemand(8., 3.)
    for delivered in (0., 8., 12.):
        e = cp.Variable(1, nonneg=True)
        ctx = SimpleNamespace(e=e, delivered=cp.Constant([delivered]))
        constraints, cost = LearnedShortfall(model, np.array([0.]), np.array([False])).build(ctx)
        problem = cp.Problem(cp.Minimize(cost), constraints)
        problem.solve(solver="CLARABEL")
        exact, _ = model.shortfall(delivered)
        assert problem.value <= exact + 1e-6
        assert problem.value >= exact - 0.01


@pytest.mark.parametrize("count, expected", [(3, [0, 3, 6]), (5, [0, 5]), (10, [0])])
def test_completion_count_trigger(count, expected, monkeypatch):
    site = Site(slot_minutes=60, horizon_hours=8, capacity_kw=6)
    sessions = pd.DataFrame({"session_id": range(6), "arrival": 0, "departure": 8,
                             "energy_requested": 10., "max_rate": 1., "p_safe": 1.})
    model = PriceModel(np.full((10, 8), 0.1), 1)

    def fake_solve(current, site, forecast, terms, past=None, solver=None):
        plan = np.ones((6, 8))
        if past is not None:
            plan[:, :past.shape[1]] = past
        return plan

    monkeypatch.setattr("evopt.sim.learning_mpc.solve", fake_solve)
    feedback, trace = run_learning_mpc(
        sessions, site, model, np.full(8, 0.1), [],
        ChargingPlant(sessions, site, np.arange(1., 7.)), GaussianEstimator(),
        replan_every_min=60, replan_every_completions=count, learn=False)
    assert trace.loc[trace.is_replan, "slot"].tolist() == expected
    assert trace.loc[trace.is_replan, "new_completions"].tolist() == [0] + [count] * (len(expected) - 1)
    np.testing.assert_allclose(feedback.delivered_kwh, np.arange(1., 7.))
