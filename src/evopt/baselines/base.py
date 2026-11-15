"""Interfaces for baseline schedulers.

To add EDF or LLF later, subclass PriorityGreedy and override `priority`:

    @BASELINES.register("edf")
    class EDF(PriorityGreedy):
        def priority(self, ev, t, remaining_kwh):
            return ev.departure
"""
from abc import ABC, abstractmethod

import numpy as np

from evopt.data.schema import availability


class Baseline(ABC):
    @abstractmethod
    def schedule(self, sessions, site) -> np.ndarray:
        """Return the same (N, T) kW array the optimizer returns."""


class PriorityGreedy(Baseline):
    """Each slot, serve unfinished EVs at full rate in priority order until capacity runs out."""

    def __init__(self, allow_partial=False):
        # False: an EV charges at its full rate or waits (whole chargers only).
        self.allow_partial = allow_partial

    def priority(self, ev, t, remaining_kwh):
        """Lower value is served first. `ev` is a row of the session table."""
        raise NotImplementedError

    def schedule(self, sessions, site) -> np.ndarray:
        plugged = availability(sessions, site.n_slots)
        evs = list(sessions.itertuples())
        remaining = sessions["energy_requested"].to_numpy(dtype=float).copy()
        power = np.zeros(plugged.shape)
        for t in range(site.n_slots):
            waiting = [i for i in np.flatnonzero(plugged[:, t]) if remaining[i] > 1e-9]
            waiting.sort(key=lambda i: self.priority(evs[i], t, remaining[i]))
            headroom = site.capacity_kw
            for i in waiting:
                want = min(evs[i].max_rate, remaining[i] / site.dt)   # last slot takes only what is left
                if want > headroom and not self.allow_partial:
                    break                                              # later EVs wait their turn
                power[i, t] = min(want, headroom)
                headroom -= power[i, t]
                remaining[i] -= power[i, t] * site.dt
                if headroom <= 1e-9:
                    break
        return power
