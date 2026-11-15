"""Price risk: penalize the variance of the energy bill (this makes the LP a QP).

With random prices the bill dt * price @ fleet has variance
dt^2 * fleet' Sigma fleet. Adding kappa times that variance is the
risk-sensitive "LP with random cost" of Boyd & Vandenberghe, sec. 4.4.1.
Sigma = F F', so the variance is the squared norm of dt * F' fleet.

kappa (1/USD) is the risk aversion; 0 gives the plain expected-cost problem.
"""
import cvxpy as cp

from evopt.model.terms.base import Term
from evopt.registry import TERMS


@TERMS.register("price_risk")
class PriceRisk(Term):
    def __init__(self, kappa=0.01):
        self.kappa = kappa

    def build(self, ctx):
        factor = ctx.prices.factor
        if self.kappa == 0 or factor.shape[1] == 0:
            return [], 0
        bill_deviation = factor.T @ (ctx.site.dt * ctx.fleet_kw)
        return [], self.kappa * cp.sum_squares(bill_deviation)
