"""Every method must respect the physical limits, whatever it optimizes."""
import numpy as np
import pytest

from evopt.baselines.fifs import FIFS
from evopt.data.demand import demand_cap
from evopt.data.schema import availability
from evopt.sim.mpc import run_mpc
from evopt.sim.plan import day_ahead, hindsight


@pytest.fixture
def schedules(case):
    sessions, site, model, test, terms = case
    return {
        "day_ahead": day_ahead(sessions, site, model, terms),
        "mpc": run_mpc(sessions, site, model, test[0], terms, replan_every_min=240),
        "hindsight": hindsight(sessions, site, model.to_slots(test[0]), terms),
        "fifs": FIFS().schedule(sessions, site),
    }


def test_limits_hold_for_every_method(case, schedules):
    sessions, site = case[0], case[1]
    plugged = availability(sessions, site.n_slots)
    for name, p in schedules.items():
        assert p.shape == plugged.shape, name
        assert (p >= 0).all(), name
        assert (p[~plugged] == 0).all(), f"{name} charges outside the plug-in window"
        assert (p <= sessions["max_rate"].to_numpy()[:, None] + 1e-6).all(), name
        assert (site.dt * p.sum(axis=1) <= demand_cap(sessions) + 1e-6).all(), name
        assert p.sum(axis=0).max() <= site.capacity_kw + 1e-4, name
        if name != "fifs":   # the baseline ignores the overcurrent limit
            assert (p <= sessions["p_safe"].to_numpy()[:, None] + 1e-6).all(), name


def test_mpc_keeps_the_past(case):
    sessions, site, model, test, terms = case
    plan = day_ahead(sessions, site, model, terms)
    applied = run_mpc(sessions, site, model, test[0], terms, replan_every_min=240, initial_plan=plan)
    first_replan = 240 // site.slot_minutes
    assert np.allclose(applied[:, :first_replan], plan[:, :first_replan])
