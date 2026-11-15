"""Tiny quadratic regularizer that makes the schedule unique.

Many per-EV splits give the same fleet load and cost. Penalizing the sum of
squared powers (Lee et al., "fairly distributing capacity") picks the one that
spreads charging evenly. eps should be small enough not to change costs.
"""
import cvxpy as cp

from evopt.model.terms.base import Term
from evopt.registry import TERMS


@TERMS.register("equal_sharing")
class EqualSharing(Term):
    def __init__(self, eps=1e-7):
        self.eps = eps

    def build(self, ctx):
        return [], self.eps * cp.sum_squares(ctx.p)
