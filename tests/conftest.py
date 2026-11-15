import numpy as np
import pytest

from evopt.config import load_config, with_overrides
from evopt.data import load_sessions
from evopt.data.prices import PriceModel, load_price_days
from evopt.data.schema import Site
from evopt.model import make_terms

SMALL = {"data.synthetic.n_sessions": 12, "site.slot_minutes": 30, "site.capacity_kw": 100}


@pytest.fixture
def cfg():
    return with_overrides(load_config("configs/base.yaml"), SMALL)


@pytest.fixture
def case(cfg):
    """A tiny problem: sessions, site, fitted price model, held-out days, terms."""
    rng = np.random.default_rng(1)
    site = Site(**cfg["site"])
    train, test = load_price_days(cfg["prices"], rng)
    model = PriceModel(train, site.n_slots // train.shape[1])
    sessions = load_sessions(cfg["data"], site, rng)
    return sessions, site, model, test, make_terms(cfg["model"]["terms"])
