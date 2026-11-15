"""Synthetic sessions following Calafiore et al. (ECC 2025), Section V.

Arrival ~ Beta(5.52, 12) scaled to the day (mode near 7 am), parking duration
~ Normal(6 h, 1 h), request = w * max_rate * duration with w ~ Uniform[0, 1].
The three draws are independent of each other; that is the paper's assumption.
"""
import numpy as np
import pandas as pd

from evopt.registry import SOURCES


@SOURCES.register("synthetic")
class Synthetic:
    def __init__(self, n_sessions=100, ev_max_kw=22.0, arrival_beta=(5.52, 12.0),
                 dwell_hours=(6.0, 1.0), request_scale=1.0):
        self.n = n_sessions
        self.ev_max_kw = ev_max_kw
        self.arrival_beta = arrival_beta
        self.dwell_hours = dwell_hours
        self.request_scale = request_scale   # 1.0 = the paper; lower it for smaller requests

    def load(self, site, rng) -> pd.DataFrame:
        T = site.n_slots
        arrival = np.minimum(np.floor(rng.beta(*self.arrival_beta, self.n) * T), T - 1).astype(int)
        dwell = np.maximum(1, np.round(rng.normal(*self.dwell_hours, self.n) / site.dt)).astype(int)
        departure = np.minimum(arrival + dwell, T)     # our choice: clip stays at midnight
        hours = (departure - arrival) * site.dt
        energy = rng.uniform(0, 1, self.n) * self.request_scale * self.ev_max_kw * hours
        return pd.DataFrame({
            "session_id": np.arange(self.n),
            "arrival": arrival,
            "departure": departure,
            "energy_requested": energy,
            "max_rate": self.ev_max_kw,
        })
