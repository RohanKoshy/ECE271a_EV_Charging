"""Figures. Every plotting function takes saved results and returns a matplotlib Figure."""
from evopt.viz.schedule_plots import fleet_load, plan_vs_replan, power_distribution, schedule_heatmap
from evopt.viz.style import save
from evopt.viz.summary_plots import cost_breakdown, cost_distribution, risk_frontier, variant_comparison

__all__ = ["fleet_load", "plan_vs_replan", "power_distribution", "schedule_heatmap", "save",
           "cost_breakdown", "cost_distribution", "risk_frontier", "variant_comparison"]
