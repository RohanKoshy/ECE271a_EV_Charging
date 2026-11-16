import numpy as np

from evopt.sim.control_comparison import run_comparison


def test_comparison_uses_matched_workloads(cfg, monkeypatch):
    from evopt.sim.plant import ChargingPlant

    demands = []

    def plant(sessions, site, demand):
        demands.append(np.array(demand))
        return ChargingPlant(sessions, site, demand)

    monkeypatch.setattr("evopt.sim.control_comparison.ChargingPlant", plant)
    cfg["experiment"].update(days=1)
    cfg["data"]["synthetic"]["n_sessions"] = 4
    cfg["model"]["mpc"]["replan_every_min"] = 240
    components = [{"mean": 5., "sd": 1.}, {"mean": 10., "sd": 2.}, {"mean": 15., "sd": 3.}]
    cfg["learning"] = {"truth": {"components": components, "weights": [.3, .4, .3]},
                       "estimator": {"initial_components": components, "initial_weights": [.3, .4, .3]}}
    metrics, trace, examples = run_comparison(cfg, log=lambda _: None)
    assert len(metrics) == 4
    assert metrics.loc[metrics.method == "day_ahead", "solves"].iloc[0] == 1
    assert metrics.loc[metrics.method == "hourly_mpc", "solves"].iloc[0] == 6
    example = examples[0]
    for name, schedule in example["schedules"].items():
        delivered = example["site"].dt * schedule.sum(axis=1)
        assert (delivered <= example["demand"] + 1e-6).all()
        assert schedule.sum(axis=0).max() <= example["site"].capacity_kw + 1e-4
        assert (schedule <= example["sessions"].p_safe.to_numpy()[:, None] + 1e-4).all()
    assert set(trace.method) == {"day_ahead", "hourly_mpc", "event_3", "event_5"}
    assert len(demands) == 4
    for demand in demands:
        np.testing.assert_array_equal(demand, example["demand"])


def test_compact_qp_preserves_objective_and_past(case):
    from evopt.model import solve
    from evopt.sim.compact_solver import solve_compact
    from evopt.model.terms.unmet_energy import expected_shortfall

    sessions, site, model, _, terms = case
    forecast = model.forecast()
    dense = solve(sessions, site, forecast, terms)
    compact = solve_compact(sessions, site, forecast, terms)

    def objective(p):
        energy = site.dt * p.sum(axis=1)
        shortfall = expected_shortfall(energy, sessions.energy_requested, sessions.energy_sd).sum()
        return (site.dt * forecast.mean @ p.sum(axis=0) + shortfall
                + site.demand_charge_per_day * p.sum(axis=0).max() + 1e-7 * np.sum(p ** 2))

    assert abs(objective(dense) - objective(compact)) < 0.01
    past = compact[:, :site.n_slots // 2].copy()
    replanned = solve_compact(sessions, site, forecast, terms, past=past)
    np.testing.assert_array_equal(replanned[:, :past.shape[1]], past)
    assert replanned.sum(axis=0).max() <= site.capacity_kw + 1e-4
