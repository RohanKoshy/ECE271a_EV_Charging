"""One-shot plans: the day-ahead plan and the hindsight optimum."""
from evopt.data.demand import known_demand
from evopt.data.prices import known
from evopt.model import solve
from evopt.model.terms.price_risk import PriceRisk


def day_ahead(sessions, site, price_model, terms, solver="CLARABEL"):
    """Plan the whole day before any price or demand is revealed."""
    return solve(sessions, site, price_model.forecast(), terms, solver=solver)


def hindsight(sessions, site, realized_prices, terms, realized_demand=None, solver="CLARABEL"):
    """Best possible schedule had prices and every EV's demand been known. A lower bound on cost."""
    no_risk = [t for t in terms if not isinstance(t, PriceRisk)]
    sessions = known_demand(sessions, realized_demand)
    return solve(sessions, site, known(realized_prices), no_risk, solver=solver)
