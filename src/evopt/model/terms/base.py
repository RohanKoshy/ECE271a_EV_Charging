"""Interface every objective term implements."""
from abc import ABC, abstractmethod


class Term(ABC):
    @abstractmethod
    def build(self, ctx):
        """Return (list of extra constraints, cost expression in USD)."""
