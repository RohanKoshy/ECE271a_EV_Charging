"""Overcurrent limit: no EV is charged above its safe power p_safe.

A hard constraint, not a penalty. p_safe (kW) is the largest charging power the
battery tolerates; it is a column of the session table (one fixed value from
`data.p_safe_kw`, or per-session values from the data source). Slots already
delivered (MPC `past`) are not constrained again.
"""
from evopt.model.terms.base import Term
from evopt.registry import TERMS


@TERMS.register("current_limit")
class CurrentLimit(Term):
    def build(self, ctx):
        limit = ctx.sessions["p_safe"].to_numpy(dtype=float)[:, None]
        return [ctx.p_future <= limit], 0
