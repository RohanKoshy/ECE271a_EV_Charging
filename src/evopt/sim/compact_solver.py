"""Equivalent charging QP with variables only for available future EV slots."""
import cvxpy as cp
import numpy as np
from scipy.sparse import csc_matrix

from evopt.data.demand import demand_cap
from evopt.data.schema import availability
from evopt.model.problem import Context, _clean
from evopt.model.terms.current_limit import CurrentLimit
from evopt.model.terms.equal_sharing import EqualSharing


def solve_compact(sessions, site, prices, terms, past=None, solver="CLARABEL"):
    n, horizon = len(sessions), site.n_slots
    k = 0 if past is None else past.shape[1]
    past = np.zeros((n, 0)) if past is None else past
    cap = demand_cap(sessions)
    allowed = availability(sessions, horizon)[:, k:]
    allowed &= (cap - site.dt * past.sum(axis=1) > 1e-8)[:, None]
    rows, columns = np.nonzero(allowed)
    width = horizon - k
    if not len(rows):
        return np.hstack([past, np.zeros((n, width))])
    x = cp.Variable(len(rows), nonneg=True)
    mapping = csc_matrix((np.ones(len(rows)), (rows * width + columns, np.arange(len(rows)))),
                         shape=(n * width, len(rows)))
    future = cp.reshape(mapping @ x, (n, width), order="C")
    p = future if k == 0 else cp.hstack([past, future])
    delivered = cp.Variable(n)
    future_fleet = cp.Variable(width)
    fleet = future_fleet if k == 0 else cp.hstack([past.sum(axis=0), future_fleet])
    e = cp.Variable(n, nonneg=True)
    upper = sessions.max_rate.to_numpy()[rows]
    constraints = [x <= upper, delivered <= cap, future_fleet <= site.capacity_kw,
                   delivered == site.dt * cp.sum(p, axis=1),
                   future_fleet == cp.sum(future, axis=0)]
    ctx = Context(p, e, fleet, sessions, site, prices, delivered, future)
    cost = 0
    for term in terms:
        if isinstance(term, CurrentLimit):
            constraints.append(x <= sessions.p_safe.to_numpy()[rows])
        elif isinstance(term, EqualSharing):
            cost += term.eps * (cp.sum_squares(x) + np.sum(past ** 2))
        else:
            extra, value = term.build(ctx)
            constraints.extend(extra)
            cost += value
    problem = cp.Problem(cp.Minimize(cost), constraints)
    options = {"direct_solve_method": "qdldl", "max_threads": 1} if solver == "CLARABEL" else {}
    problem.solve(solver=solver, **options)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
        raise RuntimeError(f"Solver ended with status '{problem.status}'")
    result = _clean(p.value, sessions, site)
    result[:, :k] = past
    return result
