"""Uncertain energy demand: each EV's request is a Gaussian random variable.

EV i asks for R_i ~ Normal(mu_i, sd_i^2) kWh, where mu_i is the `energy_requested`
column and sd_i is the `energy_sd` column (0 when absent, which recovers the
deterministic problem). The planner sees only (mu_i, sd_i). The true R_i is
drawn at evaluation time and, in MPC, revealed when the EV plugs in.

A car stops charging once it is full, so a schedule that exceeds R_i is cut off
at R_i (`deliver`). Planning for more than needed costs money; planning for less
leaves energy unmet.
"""
import numpy as np
import pandas as pd

# The planner never schedules more than mu + CAP_SIGMAS * sd for one EV.
CAP_SIGMAS = 3.0


def demand_sd(sessions: pd.DataFrame) -> np.ndarray:
    if "energy_sd" not in sessions:
        return np.zeros(len(sessions))
    return sessions["energy_sd"].to_numpy(dtype=float)


def demand_cap(sessions: pd.DataFrame) -> np.ndarray:
    """Most energy (kWh) the planner will schedule for each EV."""
    return sessions["energy_requested"].to_numpy(dtype=float) + CAP_SIGMAS * demand_sd(sessions)


def add_demand_uncertainty(sessions: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Attach `energy_sd = cv * energy_requested` unless the source already gave one."""
    s = sessions.copy()
    if "energy_sd" not in s:
        s["energy_sd"] = cfg.get("cv", 0.0) * s["energy_requested"]
    return s


def sample_demand(sessions: pd.DataFrame, rng) -> np.ndarray:
    """One realization of every EV's demand (kWh), truncated at 0."""
    mean = sessions["energy_requested"].to_numpy(dtype=float)
    return np.maximum(rng.normal(mean, demand_sd(sessions)), 0.0)


def known_demand(sessions: pd.DataFrame, demand=None, mask=None) -> pd.DataFrame:
    """Sessions whose demand is certain: the realized `demand`, or the mean if None.

    `mask` limits this to some EVs (those already plugged in, in MPC).
    """
    s = sessions.copy()
    mask = np.ones(len(s), bool) if mask is None else np.asarray(mask)
    if demand is not None:
        s.loc[mask, "energy_requested"] = np.asarray(demand, float)[mask]
    s.loc[mask, "energy_sd"] = 0.0
    return s


def deliver(schedule: np.ndarray, dt: float, demand: np.ndarray) -> np.ndarray:
    """Cut each EV's power off once it has received `demand` kWh."""
    before = dt * (np.cumsum(schedule, axis=1) - schedule)      # energy received before each slot
    room_kw = np.clip(np.asarray(demand)[:, None] - before, 0, None) / dt
    return np.minimum(schedule, room_kw)
