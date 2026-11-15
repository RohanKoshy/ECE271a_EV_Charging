"""The one evaluator. Every method is scored here, from its schedule alone."""
import numpy as np

from evopt.data.demand import deliver


def evaluate(schedule, sessions, site, realized_prices, realized_demand=None) -> dict:
    """Score an (N, T) kW schedule against the prices and demands that actually occurred.

    realized_demand (N,) kWh: each EV stops charging once it has this much.
    None uses the requested (mean) energy.
    """
    request = (sessions["energy_requested"].to_numpy(dtype=float) if realized_demand is None
               else np.asarray(realized_demand, dtype=float))
    schedule = deliver(schedule, site.dt, request)
    fleet_kw = schedule.sum(axis=0)
    delivered = site.dt * schedule.sum(axis=1)
    unmet = np.clip(request - delivered, 0, None)
    above_safe = np.clip(schedule - sessions["p_safe"].to_numpy()[:, None], 0, None)

    energy_cost = site.dt * float(realized_prices @ fleet_kw)
    demand_charge = site.demand_charge_per_day * float(fleet_kw.max())
    return {
        "energy_cost_usd": energy_cost,
        "demand_charge_usd": demand_charge,
        "total_cost_usd": energy_cost + demand_charge,
        "peak_kw": float(fleet_kw.max()),
        "demand_met_pct": 100 * delivered.sum() / max(request.sum(), 1e-9),
        "unmet_kwh": float(unmet.sum()),
        "fully_charged_pct": 100 * float(np.mean(unmet < 1e-3)),
        "kwh_above_safe": site.dt * float(above_safe.sum()),   # overcurrent measure (0 for any plan that obeys current_limit)
        "capacity_violation_kw": float(max(fleet_kw.max() - site.capacity_kw, 0)),
    }


def summarize(metrics):
    """Average each metric per variant and method; add the spread of total cost."""
    keys = ["variant", "method"]
    summary = metrics.drop(columns=["scenario", "day"]).groupby(keys, sort=False).mean()
    # Spread across simulated days (demand draws), within a scenario: this is the risk.
    spread = metrics.groupby(keys + ["scenario"], sort=False)["total_cost_usd"].std()
    summary["total_cost_std_usd"] = spread.groupby(keys, sort=False).mean()
    return summary.reset_index()
