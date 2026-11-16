"""MPC using learned Gaussian demand, with periodic or completion-count triggers."""
import cvxpy as cp
import numpy as np
import pandas as pd

from evopt.model import solve
from evopt.sim.plant import FeedbackState


class LearnedShortfall:
    def __init__(self, model, lower, complete, gamma=1.0):
        self.model, self.lower, self.complete, self.gamma = model, lower, complete, gamma

    def build(self, ctx):
        constraints = []
        for i, lower in enumerate(self.lower):
            if self.complete[i]:
                constraints.append(ctx.e[i] >= lower - ctx.delivered[i])
                continue
            grid = np.linspace(lower, max(lower + self.model.sd, float(
                self.model.quantile(0.9999, lower))), 65)
            value, slope = self.model.shortfall(grid, lower)
            constraints.append(ctx.e[i] >= value + cp.multiply(slope, ctx.delivered[i] - grid))
        gamma = ctx.prices.mean.max() if self.gamma == "max_price" else float(self.gamma)
        return constraints, gamma * ctx.e.sum()


def run_learning_mpc(sessions, site, price_model, realized_per_period, terms, plant,
                     estimator, session_prefix="day", replan_every_min=60,
                     gamma=1.0, solver="CLARABEL", learn=True, replan_every_completions=None, solve_fn=None, on_replan=None):
    if plant.site != site or plant.next_slot != 0 or plant.connected.shape != (len(sessions), site.n_slots):
        raise ValueError("plant must be fresh and match sessions and site")
    if replan_every_completions is not None and (
            isinstance(replan_every_completions, bool)
            or not isinstance(replan_every_completions, (int, np.integer))
            or replan_every_completions <= 0):
        raise ValueError("replan_every_completions must be a positive integer")
    optimize = solve if solve_fn is None else solve_fn
    step = max(1, round(replan_every_min / site.slot_minutes))
    feedback = FeedbackState(sessions, site)
    trace = []
    plan = None
    arrival = sessions.arrival.to_numpy()
    pending_completions = 0

    def record(slot, is_replan):
        model = estimator.model
        physical_mean, physical_var = model.moments()
        trace.append({"slot": slot, "is_replan": is_replan,
                      "new_completions": pending_completions,
                      "mean": model.mean, "variance": model.sd ** 2,
                      "physical_mean": float(physical_mean), "physical_variance": float(physical_var),
                      "observations": len(estimator.records),
                      "completed": sum(v[1] for v in estimator.records.values()),
                      "fit_success": estimator.fit_success})
        for j, (component, weight) in enumerate(zip(
                getattr(model, "components", [model]), getattr(model, "weights", [1.]))):
            trace[-1].update({f"component_{j}_mean": component.mean,
                              f"component_{j}_variance": component.sd ** 2,
                              f"component_{j}_weight": float(weight)})

    for k in range(site.n_slots):
        due = (k % step == 0 if replan_every_completions is None
               else pending_completions >= replan_every_completions)
        if k == 0 or due:
            if learn:
                estimator.fit()
            model = estimator.model
            lower = feedback.delivered_kwh.copy()
            current = sessions.copy()
            cap = model.quantile(0.9999, lower)
            current["energy_requested"] = np.where(feedback.complete, lower, cap)
            current["energy_sd"] = 0.0
            forecast = price_model.forecast(realized_per_period[:k // price_model.slots_per_period])
            shortfall = LearnedShortfall(model, lower, feedback.complete.copy(), gamma)
            plan = optimize(current, site, forecast, [*terms, shortfall],
                         past=feedback.actual_kw[:, :k].copy() if k else None, solver=solver)
            record(k, True)
            if on_replan is not None:
                on_replan(k)
            pending_completions = 0
        observation = plant.step(plan[:, k])
        pending_completions += int(np.count_nonzero(observation.complete & ~feedback.complete))
        feedback.update(observation)
        if learn:
            for i in np.flatnonzero(arrival <= k):
                estimator.observe((session_prefix, int(i)), feedback.delivered_kwh[i], feedback.complete[i])
    if learn:
        estimator.fit()
    record(site.n_slots, False)
    return feedback, pd.DataFrame(trace)
