"""The terms can be added and removed, and each one does what it claims."""
import numpy as np

from evopt.data.demand import deliver, demand_cap, known_demand, sample_demand
from evopt.data.prices import PriceModel, known, synthetic_price_days
from evopt.metrics import evaluate
from evopt.model.terms.unmet_energy import expected_shortfall
from evopt.model import make_terms, solve
from evopt.sim.plan import day_ahead

BASE = {"energy_cost": {}, "unmet_energy": {"gamma": 1.0}}


def _run(case, extra):
    sessions, site, model, test, _ = case
    schedule = solve(sessions, site, model.forecast(), make_terms({**BASE, **extra}))
    return evaluate(schedule, sessions, site, model.to_slots(test[0]))


def test_demand_charge_lowers_the_peak(case):
    assert _run(case, {"demand_charge": {}})["peak_kw"] < _run(case, {})["peak_kw"] - 1


def test_current_limit_is_a_hard_constraint(case):
    sessions, site, model, _, _ = case
    free = solve(sessions, site, model.forecast(), make_terms(BASE))
    capped = solve(sessions, site, model.forecast(), make_terms({**BASE, "current_limit": {}}))
    limit = sessions["p_safe"].to_numpy()[:, None]
    assert (free > limit + 1e-6).any()            # the limit matters in this case
    assert (capped <= limit + 1e-6).all()


def test_price_risk_lowers_bill_variance(case):
    # prices are known by default now, so give this term a volatile price history
    sessions, site, _, _, _ = case
    days = synthetic_price_days(np.random.default_rng(0), volatility=0.25)
    forecast = PriceModel(days, site.n_slots // days.shape[1]).forecast()

    def bill_variance(kappa):
        p = solve(sessions, site, forecast, make_terms({**BASE, "price_risk": {"kappa": kappa}}))
        return ((forecast.factor.T @ (site.dt * p.sum(axis=0))) ** 2).sum()

    assert bill_variance(1.0) < bill_variance(0.0)


def test_known_prices_have_no_risk_term(case):
    sessions, site, model, test, terms = case
    prices = model.to_slots(test[0])
    with_risk = solve(sessions, site, known(prices), terms)
    without = solve(sessions, site, known(prices), [t for t in terms if type(t).__name__ != "PriceRisk"])
    assert abs(with_risk.sum() - without.sum()) < 1e-3


def test_everything_is_delivered_when_capacity_allows(case):
    sessions, site, model, _, terms = case
    # with uncertain demand the plan hedges: at least the mean request, at most mu + 3 sd,
    # except where the overcurrent limit and plug-in window make that much unreachable
    plan = day_ahead(sessions, site, model, terms)
    delivered = site.dt * plan.sum(axis=1)
    mean = sessions["energy_requested"].to_numpy()
    assert (delivered >= np.minimum(mean, _reachable(sessions, site)) - 1e-2).all()
    assert (delivered <= demand_cap(sessions) + 1e-2).all()


# ---- Gaussian demand ----

def _delivered(plan, site):
    return site.dt * plan.sum(axis=1)


def _reachable(sessions, site):
    """Most energy each EV can physically receive: min(max_rate, p_safe) over its plug-in window."""
    rate = np.minimum(sessions["max_rate"], sessions["p_safe"])
    return (rate * (sessions["departure"] - sessions["arrival"]) * site.dt).to_numpy()


def test_sessions_carry_a_demand_sd(case):
    sessions = case[0]
    assert np.allclose(sessions["energy_sd"], 0.2 * sessions["energy_requested"])


def test_zero_sd_recovers_the_deterministic_problem(case):
    sessions, site, model, _, _ = case
    terms = make_terms(BASE)
    sure = known_demand(sessions)
    plan = solve(sure, site, model.forecast(), terms)
    assert np.allclose(_delivered(plan, site), sessions["energy_requested"], atol=1e-2)


def test_expected_shortfall_matches_monte_carlo():
    rng = np.random.default_rng(0)
    mu, sd, d = 20.0, 5.0, np.array([12.0, 20.0, 27.0])
    draws = rng.normal(mu, sd, 400_000)
    mc = np.array([np.maximum(draws - x, 0).mean() for x in d])
    assert np.allclose(expected_shortfall(d, mu, sd), mc, atol=0.03)


def test_more_demand_uncertainty_means_planning_more_energy(case):
    sessions, site, model, _, _ = case
    low = solve(sessions.assign(energy_sd=0.05 * sessions["energy_requested"]), site, model.forecast(), make_terms(BASE))
    high = solve(sessions.assign(energy_sd=0.40 * sessions["energy_requested"]), site, model.forecast(), make_terms(BASE))
    assert high.sum() > low.sum()


def test_quantile_mode_plans_the_gaussian_quantile(case):
    from scipy.stats import norm
    sessions, site, model, _, _ = case
    terms = make_terms({"energy_cost": {}, "unmet_energy": {"gamma": 5.0, "mode": "quantile", "eps": 0.05}})
    plan = solve(sessions, site, model.forecast(), terms)
    target = sessions["energy_requested"] + norm.ppf(0.95) * sessions["energy_sd"]
    cap = sessions["energy_requested"] + 3 * sessions["energy_sd"]
    assert (_delivered(plan, site) <= cap + 1e-6).all()
    reachable = (sessions["max_rate"] * (sessions["departure"] - sessions["arrival"]) * site.dt).to_numpy()
    assert np.allclose(_delivered(plan, site), np.minimum(np.minimum(target, cap), reachable), atol=1e-2)


def test_ev_stops_charging_when_full():
    schedule = np.array([[10.0, 10.0, 10.0, 10.0]])
    out = deliver(schedule, 1.0, np.array([25.0]))
    assert np.allclose(out, [[10.0, 10.0, 5.0, 0.0]])


def test_evaluation_uses_the_realized_demand(case):
    sessions, site, model, test, terms = case
    plan = solve(sessions, site, model.forecast(), terms)
    prices = model.to_slots(test[0])
    mean = sessions["energy_requested"].to_numpy()
    short = evaluate(plan, sessions, site, prices, np.minimum(0.5 * mean, _reachable(sessions, site)))
    long = evaluate(plan, sessions, site, prices, 2.0 * mean)
    assert short["unmet_kwh"] < 1e-6 < long["unmet_kwh"]
    assert short["energy_cost_usd"] < long["energy_cost_usd"]


def test_sample_demand_is_gaussian_with_the_stated_moments(case):
    sessions = case[0].iloc[[0]].assign(energy_requested=40.0, energy_sd=4.0)
    draws = np.array([sample_demand(sessions, np.random.default_rng(s))[0] for s in range(4000)])
    assert abs(draws.mean() - 40) < 0.4 and abs(draws.std() - 4) < 0.3
