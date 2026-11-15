"""Data layer: prices, safe power, config overrides, FIFS order."""
import numpy as np
import pandas as pd

from evopt.baselines.fifs import FIFS
from evopt.config import with_overrides
from evopt.data import load_sessions
from evopt.data.schema import Site


def test_forecast_conditioning(case):
    _, site, model, test, _ = case
    assert model.forecast().mean.shape == (site.n_slots,)
    seen = model.forecast(test[0][:10])
    assert np.allclose(seen.mean[: 10 * model.slots_per_period], model.to_slots(test[0])[: 10 * model.slots_per_period])
    assert np.allclose(seen.factor[: 10 * model.slots_per_period], 0)      # observed prices are certain
    assert model.forecast(test[0]).factor.shape[1] == 0                    # nothing left to be unsure of


def test_seeing_prices_improves_the_forecast(case):
    """Forecast error for the afternoon shrinks once the morning's prices are known."""
    _, _, model, test, _ = case
    afternoon = slice(12 * model.slots_per_period, None)

    def error(hours_seen):
        return np.mean([np.abs(model.forecast(day[:hours_seen]).mean - model.to_slots(day))[afternoon].mean()
                        for day in test])

    assert error(12) < error(0)


def test_sessions_get_the_safe_power_from_config(cfg):
    cfg = with_overrides(cfg, {"data.p_safe_kw": 4.0})
    sessions = load_sessions(cfg["data"], Site(**cfg["site"]), np.random.default_rng(0))
    assert (sessions["p_safe"] == 4.0).all()


def test_fifs_serves_the_earlier_arrival_first():
    site = Site(slot_minutes=60, capacity_kw=10)
    sessions = pd.DataFrame({"session_id": [0, 1], "arrival": [0, 1], "departure": [6, 6],
                             "energy_requested": [30.0, 30.0], "max_rate": [10.0, 10.0], "p_safe": 5.0})
    power = FIFS().schedule(sessions, site)
    assert power[0, :3].tolist() == [10, 10, 10] and power[1, :3].sum() == 0   # second EV waits
    assert power[1, 3:6].tolist() == [10, 10, 10]
