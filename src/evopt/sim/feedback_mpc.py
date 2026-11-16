"""Periodic MPC with slot-level plant execution and observed-energy feedback.

Session windows remain known in advance. Passing realized_demand preserves the
legacy assumption that demand is revealed at the first replan after arrival.
A custom plant can instead keep physical demand private. Replans fix measured
past power, not commanded power. Event-triggered solves are not implemented here.
"""
from evopt.data.demand import demand_cap, known_demand
from evopt.model import solve
from evopt.sim.plant import ChargingPlant, FeedbackState


def run_feedback_mpc(sessions, site, price_model, realized_per_period, terms,
            replan_every_min=60, initial_plan=None, realized_demand=None, solver="CLARABEL",
            plant=None, return_feedback=False):
    """Return the (N, T) schedule actually applied over the day.

    realized_demand: (N,) true demand in kWh. None means demand never gets revealed.
    plant: optional ChargingPlant with independent physical demand and response.
        Omit realized_demand with a custom plant to avoid revealing its ground truth.
    return_feedback: return (actual schedule, FeedbackState) for diagnostics.
    """
    step = max(1, round(replan_every_min / site.slot_minutes))
    arrival = sessions["arrival"].to_numpy()
    schedule = initial_plan
    if schedule is None:
        schedule = solve(sessions, site, price_model.forecast(), terms, solver=solver)
    if plant is None:
        # Without a realization, retain the former no-cutoff behavior up to the
        # planner's energy cap, rather than cutting uncertain plans at the mean.
        demand = demand_cap(sessions) if realized_demand is None else realized_demand
        plant = ChargingPlant(sessions, site, demand)
    if plant.site != site or plant.connected.shape != schedule.shape or plant.next_slot != 0:
        raise ValueError("plant must be fresh and match the site's horizon and session count")
    feedback = FeedbackState(sessions, site)
    for k in range(site.n_slots):
        if k and k % step == 0:
            periods_seen = k // price_model.slots_per_period
            forecast = price_model.forecast(realized_per_period[:periods_seen])
            current = sessions.copy()
            if realized_demand is not None:
                current = known_demand(current, realized_demand, mask=arrival <= k)
            # A completion signal establishes the total delivered demand, even
            # when exact demand was not disclosed on arrival.
            current = known_demand(current, feedback.delivered_kwh, mask=feedback.complete)
            schedule = solve(current, site, forecast, terms,
                             past=feedback.actual_kw[:, :k].copy(), solver=solver)
        feedback.update(plant.step(schedule[:, k]))
    return (feedback.actual_kw, feedback) if return_feedback else feedback.actual_kw
