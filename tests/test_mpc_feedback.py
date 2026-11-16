"""Integration: MPC fixes measured delivery and responds to plant underdelivery."""
import numpy as np

from evopt.data.demand import known_demand
from evopt.sim.feedback_mpc import run_feedback_mpc
from evopt.sim.plant import ChargingPlant


def test_mpc_passes_observed_past_to_optimizer(case, monkeypatch):
    sessions, site, model, prices, terms = case
    sessions = known_demand(sessions)
    initial = np.zeros((len(sessions), site.n_slots))
    # Use a connected EV so the first executed command has a known response.
    i = 0
    k = int(sessions.iloc[i].arrival)
    initial[i, k] = min(2., float(sessions.iloc[i].max_rate),
                        float(sessions.iloc[i].energy_requested) / site.dt)
    calls = []

    def fake_solve(current, site_arg, forecast, terms_arg, past=None, solver=None):
        calls.append(past.copy())
        plan = np.zeros_like(initial)
        plan[:, :past.shape[1]] = past
        return plan

    monkeypatch.setattr("evopt.sim.feedback_mpc.solve", fake_solve)
    # Replan immediately after the nonzero command, then follow the new plan.
    plant = ChargingPlant(sessions, site, sessions.energy_requested, response_fraction=0.5)
    actual, feedback = run_feedback_mpc(
        sessions, site, model, prices[0], terms,
        replan_every_min=(k + 1) * site.slot_minutes,
        initial_plan=initial, plant=plant, return_feedback=True)
    assert calls
    assert np.isclose(calls[0][i, k], initial[i, k] * 0.5)
    assert np.isclose(actual[i, k], initial[i, k] * 0.5)
    assert np.isclose(feedback.delivered_kwh[i], site.dt * initial[i, k] * 0.5)
