"""Data layer: every source is converted to the one session table in schema.py."""
from evopt.data.demand import add_demand_uncertainty
from evopt.data.schema import Site, validate
from evopt.registry import SOURCES, import_submodules

import_submodules(__name__, __path__)


def load_sessions(cfg: dict, site: Site, rng):
    """Load sessions from the configured source and attach the safe charging power."""
    # Arguments sit under the source's own name (data.synthetic, data.acn, ...), so a
    # base config's settings for one source are never passed to another.
    source = SOURCES.create(cfg["source"], **cfg.get(cfg["source"], {}))
    sessions = validate(source.load(site, rng), site)
    if "p_safe" not in sessions:   # one safe power for every EV; absent = no limit beyond max_rate
        sessions["p_safe"] = float(cfg.get("p_safe_kw", sessions["max_rate"].max()))
    return add_demand_uncertainty(sessions, cfg.get("demand", {}))
