"""Uncertain electricity prices: a mean and covariance estimated from past days.

Prices are stored per period (24 hourly values per day) and repeated onto slots.
The optimizer only ever sees a PriceForecast: an expected price per slot and a
factor F with covariance = F @ F.T.
"""
from dataclasses import dataclass

import numpy as np

# SCE TOU EV-4 summer energy prices (Lee et al., Table III), USD/kWh by hour of day.
TOU_HOURLY = np.array([0.056] * 8 + [0.092] * 4 + [0.267] * 6 + [0.092] * 5 + [0.056])


@dataclass(frozen=True)
class PriceForecast:
    mean: np.ndarray      # (T,) expected USD/kWh per slot
    factor: np.ndarray    # (T, r); price covariance = factor @ factor.T


def known(prices_per_slot: np.ndarray) -> PriceForecast:
    """A forecast with no uncertainty (used for the hindsight optimum)."""
    return PriceForecast(np.asarray(prices_per_slot, float), np.zeros((len(prices_per_slot), 0)))


class PriceModel:
    """Gaussian price model fitted on training days of shape (days, periods)."""

    def __init__(self, train_days: np.ndarray, slots_per_period: int, shrinkage: float = 0.1):
        sample = np.cov(train_days, rowvar=False)
        self.mean = train_days.mean(axis=0)
        self.cov = (1 - shrinkage) * sample + shrinkage * np.diag(np.diag(sample))
        self.slots_per_period = slots_per_period

    def forecast(self, observed=()) -> PriceForecast:
        """Forecast the day given the realized prices of its first len(observed) periods.

        Standard Gaussian conditioning: observed periods become certain, and the
        mean and covariance of the remaining periods are updated.
        """
        observed = np.asarray(observed, dtype=float)
        h = len(observed)
        mean, cov = self.mean.copy(), self.cov.copy()
        if h:
            gain = self.cov[h:, :h] @ np.linalg.pinv(self.cov[:h, :h])
            mean[:h] = observed
            mean[h:] = self.mean[h:] + gain @ (observed - self.mean[:h])
            cov = np.zeros_like(self.cov)
            cov[h:, h:] = self.cov[h:, h:] - gain @ self.cov[:h, h:]
        return PriceForecast(self.to_slots(mean), self.to_slots(_factor(cov)))

    def to_slots(self, per_period: np.ndarray) -> np.ndarray:
        return np.repeat(per_period, self.slots_per_period, axis=0)


def _factor(cov: np.ndarray) -> np.ndarray:
    """Return F with F @ F.T = cov, dropping directions with no variance."""
    values, vectors = np.linalg.eigh((cov + cov.T) / 2)
    keep = values > 1e-9 * max(values.max(), 1e-12)
    return vectors[:, keep] * np.sqrt(values[keep])


def load_price_days(cfg: dict, rng):
    """Return (train_days, test_days), each of shape (days, periods)."""
    params = cfg.get("params", {})
    if cfg["source"] == "synthetic":
        days = synthetic_price_days(rng, **params)
    elif cfg["source"] == "csv":
        days = np.loadtxt(params["path"], delimiter=",", skiprows=params.get("skiprows", 0))
    else:
        raise ValueError(f"Unknown price source '{cfg['source']}'")
    n_train = int(len(days) * cfg.get("train_fraction", 0.8))
    return days[:n_train], days[n_train:]


def synthetic_price_days(rng, n_days=200, volatility=0.25, persistence=0.9) -> np.ndarray:
    """PLACEHOLDER price history until a real dataset is chosen.

    The TOU shape times a lognormal deviation that follows an AR(1) process across
    hours, so a surprise in one hour carries into the next. `volatility` and
    `persistence` are assumptions.
    """
    shocks = rng.standard_normal((n_days, len(TOU_HOURLY)))
    x = np.empty_like(shocks)
    x[:, 0] = shocks[:, 0]
    for h in range(1, x.shape[1]):
        x[:, h] = persistence * x[:, h - 1] + np.sqrt(1 - persistence**2) * shocks[:, h]
    return TOU_HOURLY * np.exp(volatility * x - volatility**2 / 2)
