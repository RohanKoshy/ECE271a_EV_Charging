"""Objective terms. Each file registers one term; add a file to add a term."""
from evopt.registry import import_submodules

import_submodules(__name__, __path__)
