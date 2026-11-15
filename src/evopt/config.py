"""YAML configs: an experiment file extends a base file and overrides parts of it."""
import copy
from pathlib import Path

import yaml


def load_config(path) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text()) or {}
    parent = cfg.pop("extends", None)
    if parent:
        cfg = deep_merge(load_config(path.parent / parent), cfg)
    return cfg


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def with_overrides(cfg: dict, overrides: dict) -> dict:
    """Return a copy of cfg with dotted-path overrides, e.g. {'data.demand.cv': 0.3}."""
    out = copy.deepcopy(cfg)
    for dotted, value in (overrides or {}).items():
        *parents, leaf = dotted.split(".")
        node = out
        for key in parents:
            node = node.setdefault(key, {})
        node[leaf] = copy.deepcopy(value)
    return out
