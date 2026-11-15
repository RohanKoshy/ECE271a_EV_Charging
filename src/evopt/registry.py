"""Plug-in registries: objective terms, baselines and data sources register by name."""
import importlib
import pkgutil


class Registry:
    def __init__(self, kind: str):
        self.kind = kind
        self._items = {}

    def register(self, name: str):
        def decorator(cls):
            self._items[name] = cls
            return cls
        return decorator

    def create(self, name: str, **params):
        if name not in self._items:
            raise KeyError(f"Unknown {self.kind} '{name}'. Available: {sorted(self._items)}")
        return self._items[name](**params)

    def names(self):
        return sorted(self._items)


TERMS = Registry("term")
BASELINES = Registry("baseline")
SOURCES = Registry("data source")


def import_submodules(package_name: str, package_path) -> None:
    """Import every module in a package so its @register decorators run.

    This is what lets a new plug-in be a single new file with no other edits.
    """
    for module in pkgutil.iter_modules(package_path):
        importlib.import_module(f"{package_name}.{module.name}")
