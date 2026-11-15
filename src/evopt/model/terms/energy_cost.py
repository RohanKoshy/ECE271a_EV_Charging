"""Expected cost of the energy bought: sum over slots of price * energy."""
from evopt.model.terms.base import Term
from evopt.registry import TERMS


@TERMS.register("energy_cost")
class EnergyCost(Term):
    def build(self, ctx):
        return [], ctx.site.dt * (ctx.prices.mean @ ctx.fleet_kw)
