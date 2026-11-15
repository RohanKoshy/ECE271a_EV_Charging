"""The optimization model: a fixed base plus pluggable objective terms."""
from evopt.model.problem import Context, build, make_terms, solve

__all__ = ["Context", "build", "make_terms", "solve"]
