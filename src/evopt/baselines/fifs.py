"""First-In-First-Served: the baseline in both Calafiore papers.

EVs charge at full rate in order of arrival until done or departed; later
arrivals wait when the site capacity is used up. It ignores prices.
"""
from evopt.baselines.base import PriorityGreedy
from evopt.registry import BASELINES


@BASELINES.register("fifs")
class FIFS(PriorityGreedy):
    def priority(self, ev, t, remaining_kwh):
        return ev.arrival
