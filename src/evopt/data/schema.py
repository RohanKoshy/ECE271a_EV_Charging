"""The session table and site description that every stage shares.

A session table is a DataFrame with one row per EV visit:

    session_id        any identifier
    arrival           first slot the EV is plugged in (int)
    departure         first slot it is gone (int, exclusive)
    energy_requested  kWh
    max_rate          kW the EV/charger pair can take
    p_safe            kW the charger may not exceed (overcurrent limit; filled from data.p_safe_kw)

Optional: driver_id.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

REQUIRED = ["session_id", "arrival", "departure", "energy_requested", "max_rate"]


@dataclass(frozen=True)
class Site:
    slot_minutes: int = 5
    horizon_hours: int = 24
    capacity_kw: float = 1130.0
    demand_charge_usd_per_kw_month: float = 15.51
    billing_days: int = 30

    @property
    def dt(self) -> float:
        """Slot length in hours."""
        return self.slot_minutes / 60

    @property
    def n_slots(self) -> int:
        return round(self.horizon_hours / self.dt)

    @property
    def demand_charge_per_day(self) -> float:
        """Monthly demand charge prorated to one day (USD per kW of peak)."""
        return self.demand_charge_usd_per_kw_month / self.billing_days


def validate(sessions: pd.DataFrame, site: Site) -> pd.DataFrame:
    """Check required columns and clip plug-in windows to the horizon."""
    missing = [c for c in REQUIRED if c not in sessions.columns]
    if missing:
        raise ValueError(f"Session table is missing columns: {missing}")
    s = sessions.copy()
    s["arrival"] = s["arrival"].clip(0, site.n_slots - 1).astype(int)
    s["departure"] = np.clip(s["departure"], s["arrival"] + 1, site.n_slots).astype(int)
    if (s["energy_requested"] < 0).any() or (s["max_rate"] <= 0).any():
        raise ValueError("energy_requested must be >= 0 and max_rate > 0")
    return s.sort_values("arrival", kind="stable").reset_index(drop=True)


def availability(sessions: pd.DataFrame, n_slots: int) -> np.ndarray:
    """Boolean (N, T) matrix: True where EV i is plugged in during slot t."""
    t = np.arange(n_slots)
    arrival = sessions["arrival"].to_numpy()[:, None]
    departure = sessions["departure"].to_numpy()[:, None]
    return (arrival <= t) & (t < departure)


def hours_to_slot(hours, site: Site, round_up: bool = False) -> np.ndarray:
    slots = np.asarray(hours, dtype=float) / site.dt
    return (np.ceil(slots) if round_up else np.floor(slots)).astype(int)
