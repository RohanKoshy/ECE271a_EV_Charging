"""Builds and solves the charging problem.

Base (always on), following problem (1) of Calafiore et al. (ECC 2025), except
that each EV's demand R[i] ~ Normal(mu[i], sd[i]^2) is a Gaussian random variable:

    variables   p[i, t] >= 0   charging power of EV i in slot t (kW)
                e[i]    >= 0   expected energy EV i does not receive (kWh)
    subject to  p[i, t] <= max_rate[i] while plugged in, 0 otherwise
                energy delivered to i <= mu[i] + 3 sd[i]
                fleet power in each slot <= site capacity

The link between e[i] and the delivered energy (E[(R - delivered)+] <= e[i], or a
chance constraint) is added by the unmet_energy term. Prices are treated as known.

Everything in the objective comes from the terms in model/terms/.
"""
from dataclasses import dataclass

import cvxpy as cp
import numpy as np
import pandas as pd

from evopt.data.demand import demand_cap
from evopt.data.prices import PriceForecast
from evopt.data.schema import Site, availability
from evopt.model import terms as _terms  # noqa: F401  (registers the terms)
from evopt.registry import TERMS


@dataclass
class Context:
    """What a term can read when it adds its cost and constraints."""
    p: cp.Expression          # (N, T) charging power, kW
    e: cp.Variable            # (N,) expected unmet energy, kWh (bounded below by the unmet_energy term)
    fleet_kw: cp.Expression   # (T,) total charging power per slot
    sessions: pd.DataFrame
    site: Site
    prices: PriceForecast
    delivered: cp.Expression  # (N,) energy scheduled for each EV, kWh
    p_future: cp.Variable     # (N, T-k) the slots still to be decided


def make_terms(terms_cfg: dict) -> list:
    """{'demand_charge': {}, 'price_risk': {'kappa': 0.01}} -> list of term objects."""
    return [TERMS.create(name, **(params or {})) for name, params in terms_cfg.items()]


def build(sessions, site, prices, terms, past=None):
    """Return (cvxpy problem, power expression).

    `past` is an (N, k) array of power already delivered in the first k slots.
    MPC passes it so that only the remaining slots are decided.
    """
    N, T = len(sessions), site.n_slots
    k = 0 if past is None else past.shape[1]
    upper = sessions["max_rate"].to_numpy()[:, None] * availability(sessions, T)
    cap = demand_cap(sessions)

    p_future = cp.Variable((N, T - k), nonneg=True, name="power_kw")
    p = p_future if k == 0 else cp.hstack([past, p_future])
    e = cp.Variable(N, nonneg=True, name="unmet_kwh")
    delivered = site.dt * cp.sum(p, axis=1)
    fleet_kw = cp.sum(p, axis=0)

    constraints = [
        p_future <= upper[:, k:],
        delivered <= cap,
        fleet_kw[k:] <= site.capacity_kw,
    ]
    ctx = Context(p, e, fleet_kw, sessions, site, prices, delivered, p_future)
    cost = 0
    for term in terms:
        term_constraints, term_cost = term.build(ctx)
        constraints += term_constraints
        cost = cost + term_cost
    return cp.Problem(cp.Minimize(cost), constraints), p


def solve(sessions, site, prices, terms, past=None, solver="CLARABEL") -> np.ndarray:
    """Solve and return the (N, T) schedule in kW."""
    problem, p = build(sessions, site, prices, terms, past)
    problem.solve(solver=solver)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver ended with status '{problem.status}'")
    return _clean(p.value, sessions, site)


def _clean(schedule, sessions, site) -> np.ndarray:
    """Remove solver round-off so the schedule satisfies the hard limits exactly."""
    upper = sessions["max_rate"].to_numpy()[:, None] * availability(sessions, site.n_slots)
    schedule = np.clip(schedule, 0, upper)
    schedule[schedule < 1e-6] = 0
    delivered = site.dt * schedule.sum(axis=1)
    cap = demand_cap(sessions)
    over = delivered > cap
    schedule[over] *= (cap[over] / delivered[over])[:, None]
    return schedule
