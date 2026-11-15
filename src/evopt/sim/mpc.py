"""MPC over revealed prices and revealed demand.

Sessions are known a day ahead, but each EV's energy demand is only a Gaussian
(mean, sd). At each re-plan time the controller (1) keeps what has already been
delivered (an EV stops when full, so that is cut off at its true demand),
(2) updates the price forecast with the prices seen so far, (3) learns the true
demand of every EV that has plugged in by now, and (4) re-solves the same
problem for the remaining slots.
"""
from evopt.data.demand import deliver, known_demand
from evopt.model import solve


def run_mpc(sessions, site, price_model, realized_per_period, terms,
            replan_every_min=60, initial_plan=None, realized_demand=None, solver="CLARABEL"):
    """Return the (N, T) schedule actually applied over the day.

    realized_demand: (N,) true demand in kWh. None means demand never gets revealed.
    """
    step = max(1, round(replan_every_min / site.slot_minutes))
    arrival = sessions["arrival"].to_numpy()
    schedule = initial_plan
    if schedule is None:
        schedule = solve(sessions, site, price_model.forecast(), terms, solver=solver)
    for k in range(step, site.n_slots, step):
        periods_seen = k // price_model.slots_per_period
        forecast = price_model.forecast(realized_per_period[:periods_seen])
        past, current = schedule[:, :k], sessions
        if realized_demand is not None:
            past = deliver(past, site.dt, realized_demand)
            current = known_demand(sessions, realized_demand, mask=arrival <= k)
        schedule = solve(current, site, forecast, terms, past=past, solver=solver)
    if realized_demand is not None:
        schedule = deliver(schedule, site.dt, realized_demand)
    return schedule
