"""Slot-level plant and controller feedback, in session-table row order.

The plant owns physical demand. FeedbackState knows only requests and observations.
Power is slot-average kW; energy is kWh. Measurements are exact in this simulator.
"""
from dataclasses import dataclass

import numpy as np

from evopt.data.schema import availability


def _vector(value, n, name):
    out = np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()
    if not np.isfinite(out).all() or (out < 0).any():
        raise ValueError(f"{name} must contain finite nonnegative values")
    return out


@dataclass(frozen=True)
class Observation:
    slot: int
    commanded_kw: np.ndarray
    actual_kw: np.ndarray
    delivered_kwh: np.ndarray
    connected: np.ndarray
    complete: np.ndarray


class ChargingPlant:
    """Ideal tracking by default; response_fraction models power underdelivery.

    Actual power = min(command * response_fraction, charger limit, energy left / dt)
    while connected. Commands must already respect aggregate site capacity.
    The optional safe-power cap remains a controller policy, not a physical limit.
    """

    def __init__(self, sessions, site, demand, response_fraction=1.0):
        self.site = site
        self.remaining_kwh = _vector(demand, len(sessions), "demand")
        self.response_fraction = _vector(response_fraction, len(sessions), "response_fraction")
        if (self.response_fraction > 1).any():
            raise ValueError("response_fraction must be <= 1")
        self.max_kw = _vector(sessions["max_rate"], len(sessions), "max_rate")
        self.connected = availability(sessions, site.n_slots)
        self.next_slot = 0

    def step(self, command_kw):
        if self.next_slot >= self.site.n_slots:
            raise ValueError("plant horizon is exhausted")
        command = _vector(command_kw, len(self.max_kw), "command_kw")
        if command.sum() > self.site.capacity_kw + 1e-4:
            raise ValueError("command exceeds site capacity")
        k = self.next_slot
        actual = np.minimum(command * self.response_fraction, self.max_kw)
        actual = np.minimum(actual, self.remaining_kwh / self.site.dt)
        actual = np.where(self.connected[:, k], actual, 0.0)
        energy = actual * self.site.dt
        self.remaining_kwh = np.maximum(0.0, self.remaining_kwh - energy)
        self.next_slot += 1
        return Observation(k, command, actual, energy, self.connected[:, k].copy(),
                           self.connected[:, k] & (self.remaining_kwh <= 1e-9))


class FeedbackState:
    """Update request accounting from observations, without reading plant demand.

    Tracking error compares commands with actual slot-average power. Completion
    is an explicit plant signal, never inferred merely from low power draw.
    """

    def __init__(self, sessions, site):
        self.site = site
        self.requested_kwh = _vector(sessions["energy_requested"], len(sessions), "request")
        self.actual_kw = np.zeros((len(sessions), site.n_slots))
        self.commanded_kw = np.zeros_like(self.actual_kw)
        self.complete = np.zeros(len(sessions), dtype=bool)
        self.next_slot = 0

    def update(self, observation):
        k = observation.slot
        if k != self.next_slot or k >= self.site.n_slots:
            raise ValueError("observations must arrive once each in slot order")
        n = len(self.requested_kwh)
        command = _vector(observation.commanded_kw, n, "commanded_kw")
        actual = _vector(observation.actual_kw, n, "actual_kw")
        energy = _vector(observation.delivered_kwh, n, "delivered_kwh")
        complete = np.asarray(observation.complete, dtype=bool)
        if complete.shape != (n,) or not np.allclose(energy, actual * self.site.dt):
            raise ValueError("inconsistent observation shape or delivered energy")
        self.actual_kw[:, k] = actual
        self.commanded_kw[:, k] = command
        self.complete |= complete
        self.next_slot += 1

    @property
    def delivered_kwh(self):
        return self.site.dt * self.actual_kw[:, :self.next_slot].sum(axis=1)

    @property
    def remaining_requested_kwh(self):
        return np.where(self.complete, 0.0, np.maximum(0.0, self.requested_kwh - self.delivered_kwh))

    @property
    def tracking_error_kw(self):
        return self.commanded_kw[:, :self.next_slot] - self.actual_kw[:, :self.next_slot]

    @property
    def peak_kw(self):
        return float(self.actual_kw[:, :self.next_slot].sum(axis=0).max(initial=0.0))
