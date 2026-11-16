import numpy as np
import pytest

from evopt.learning.mixture import GaussianMixtureDemand, GaussianMixtureEstimator
from evopt.sim.learning_runner import run_learning_experiment


COMPONENTS = [{"mean": 8., "sd": 2.}, {"mean": 20., "sd": 3.}, {"mean": 35., "sd": 4.}]
WEIGHTS = [0.4, 0.4, 0.2]


def test_sampling_uses_hidden_type_and_nonnegative_demands():
    model = GaussianMixtureDemand(COMPONENTS, WEIGHTS)
    values, types = model.sample_with_types(20000, np.random.default_rng(4))
    assert (values >= 0).all()
    np.testing.assert_allclose(np.bincount(types) / len(types), WEIGHTS, atol=0.015)
    for j, component in enumerate(COMPONENTS):
        assert abs(values[types == j].mean() - component["mean"]) < 0.2


def test_conditional_shortfall_and_quantile():
    model = GaussianMixtureDemand(COMPONENTS, WEIGHTS)
    samples = model.sample(200000, np.random.default_rng(5))
    for lower in (0., 18., 30.):
        subset = samples[samples > lower]
        delivered = lower + 5.
        value, slope = model.shortfall(delivered, lower)
        assert abs(value - np.maximum(subset - delivered, 0.).mean()) < 0.1
        assert abs(-slope - np.mean(subset > delivered)) < 0.01
        q = float(model.quantile(0.9, lower))
        assert abs(np.mean(subset <= q) - 0.9) < 0.01
    weights = model.conditional_weights(30.)
    assert weights[2] > 0.99
    mean, variance = model.moments()
    assert abs(mean - samples.mean()) < 0.1
    assert abs(variance - samples.var()) < 1.


def test_mixture_estimator_recovers_components_from_censored_sessions():
    model = GaussianMixtureDemand(COMPONENTS, WEIGHTS)
    samples = model.sample(1800, np.random.default_rng(8))
    estimator = GaussianMixtureEstimator(
        initial_components=[{"mean": 6., "sd": 3.}, {"mean": 17., "sd": 4.}, {"mean": 30., "sd": 5.}],
        initial_weights=[0.3, 0.4, 0.3])
    for i, value in enumerate(samples):
        limit = 18. if i % 2 else 50.
        estimator.observe(i, min(value, limit), value <= limit)
    fitted = estimator.fit()
    assert estimator.fit_success
    np.testing.assert_allclose([c.mean for c in fitted.components], [8., 20., 35.], atol=1.5)
    np.testing.assert_allclose([c.sd for c in fitted.components], [2., 3., 4.], atol=1.)
    np.testing.assert_allclose(fitted.weights, WEIGHTS, atol=0.06)


def test_runner_wires_hidden_mixture(cfg):
    cfg["experiment"].update(days=1, scenarios=1)
    cfg["data"]["synthetic"]["n_sessions"] = 4
    cfg["methods"] = ["learning_mpc"]
    cfg["model"]["mpc"]["replan_every_completions"] = 3
    cfg["learning"] = {
        "truth": {"components": COMPONENTS, "weights": WEIGHTS},
        "estimator": {"initial_components": COMPONENTS, "initial_weights": [0.3, 0.4, 0.3]}}
    metrics, trace, example = run_learning_experiment(cfg, log=lambda _: None)
    assert len(metrics) == 1
    assert set(example["true_car_types"]) <= {0, 1, 2}
    assert "true_car_types" not in example["sessions"]
    assert trace.iloc[0]["observations"] == 0
    assert trace.iloc[0]["component_0_weight"] == pytest.approx(0.3)
    assert all(f"component_{j}_variance" in trace for j in range(3))
    assert metrics.capacity_violation_kw.max() < 1e-4
