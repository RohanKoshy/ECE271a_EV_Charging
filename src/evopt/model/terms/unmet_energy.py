"""Penalty on requested energy that is not delivered, with Gaussian demand.

EV i's demand is R_i ~ Normal(mu_i, sd_i^2) and the plan delivers d_i kWh.
gamma is the price of one undelivered kWh. A number, or "max_price" for the
highest expected price of the day (the choice in Calafiore et al.). Note that
"max_price" lets the optimizer skip charging when another term makes it
expensive, so the default config uses a larger fixed value.

mode "expected" (default) charges gamma * E[(R_i - d_i)+], the expected shortfall:

    E[(R - d)+] = sd * (phi(z) + z * Phi(z)),   z = (mu - d) / sd,

which is convex in d. cvxpy has no Gaussian CDF, so the function is replaced by
the maximum of its tangent lines at z = -3, -2.75, ..., 4 (always a lower bound,
and within 0.003 * sd of the exact value). Each tangent is one linear
constraint on e_i.

mode "quantile" is a chance constraint P(R_i <= d_i) >= 1 - eps. It is linear:
plan for d_i >= mu_i + z_(1-eps) * sd_i, and e_i is the slack on that target.
eps must be at least 0.00135 so the target stays under the 3-sigma cap.

With sd = 0 both modes reduce to the deterministic constraint e_i >= mu_i - d_i.
"""
import numpy as np
from scipy.stats import norm

import cvxpy as cp

from evopt.data.demand import demand_sd
from evopt.model.terms.base import Term
from evopt.registry import TERMS

Z_GRID = np.arange(-3.0, 4.01, 0.25)


def expected_shortfall(d, mu, sd):
    """E[(R - d)+] for R ~ Normal(mu, sd^2); the exact function the tangents approximate."""
    d, mu, sd = np.broadcast_arrays(np.asarray(d, float), np.asarray(mu, float), np.asarray(sd, float))
    out = np.maximum(mu - d, 0.0)
    pos = sd > 0
    z = (mu[pos] - d[pos]) / sd[pos]
    out = np.array(out)
    out[pos] = sd[pos] * (norm.pdf(z) + z * norm.cdf(z))
    return out


@TERMS.register("unmet_energy")
class UnmetEnergy(Term):
    def __init__(self, gamma=1.0, mode="expected", eps=0.05):
        if mode not in ("expected", "quantile"):
            raise ValueError(f"unmet_energy mode must be 'expected' or 'quantile', got '{mode}'")
        self.gamma = gamma
        self.mode = mode
        self.eps = eps

    def build(self, ctx):
        gamma = ctx.prices.mean.max() if self.gamma == "max_price" else float(self.gamma)
        mu = ctx.sessions["energy_requested"].to_numpy(dtype=float)
        sd = demand_sd(ctx.sessions)
        d, e = ctx.delivered, ctx.e

        if self.mode == "quantile":
            target = mu + norm.ppf(1 - self.eps) * sd
            return [e >= target - d], gamma * cp.sum(e)

        constraints = []
        certain = np.flatnonzero(sd <= 1e-9)
        if len(certain):
            constraints.append(e[certain] >= mu[certain] - d[certain])
        random = np.flatnonzero(sd > 1e-9)
        if len(random):
            n, K = len(random), len(Z_GRID)
            m, s = mu[random, None], sd[random, None]
            at = m - s * Z_GRID                                   # tangent points d_k, (n, K)
            value = s * (norm.pdf(Z_GRID) + Z_GRID * norm.cdf(Z_GRID))
            slope = np.broadcast_to(-norm.cdf(Z_GRID), (n, K))
            ones = np.ones((1, K))
            d_rep = cp.reshape(d[random], (n, 1), order="C") @ ones     # (n, K): d_i in every column
            e_rep = cp.reshape(e[random], (n, 1), order="C") @ ones
            constraints.append(e_rep >= value + cp.multiply(slope, d_rep - at))
        return constraints, gamma * cp.sum(e)
