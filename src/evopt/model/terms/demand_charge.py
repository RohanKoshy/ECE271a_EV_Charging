"""Demand charge: pay for the single highest fleet power of the day.

Epigraph form of a max (Boyd & Vandenberghe, sec. 4.1.3): minimize alpha * s
subject to s >= fleet power in every slot. During MPC the slots already
delivered are constants, so a peak that has already happened stays paid for.
"""
import cvxpy as cp

from evopt.model.terms.base import Term
from evopt.registry import TERMS


@TERMS.register("demand_charge")
class DemandCharge(Term):
    def __init__(self, rate_per_kw=None, prior_peak_kw=0.0):
        self.rate = rate_per_kw            # USD per kW of daily peak; default: site tariff / billing days
        self.prior_peak_kw = prior_peak_kw  # peak already set earlier in the billing period

    def build(self, ctx):
        rate = ctx.site.demand_charge_per_day if self.rate is None else self.rate
        peak = cp.Variable(nonneg=True, name="peak_kw")
        return [peak >= ctx.fleet_kw, peak >= self.prior_peak_kw], rate * peak
