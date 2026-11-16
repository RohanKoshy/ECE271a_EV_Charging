# Plant and observation feedback

`ChargingPlant` executes one slot at a time. It owns physical remaining demand,
clips power to charger limits and connection windows, and stops charging when
full. The default response follows the command exactly. `response_fraction`
(a scalar or one value per session, between zero and one) models underdelivery;
it is a simple test model, not a calibrated battery or efficiency model.

Each `Observation` reports slot-average commanded and actual kW, delivered kWh,
connection status, and an explicit completion signal. There is no sensor noise.
Rows always use the input session table's order. Completion is assumed observable;
it is not inferred from low charging power.

`FeedbackState.update(observation)` records commands and measurements separately.
Delivered energy is accumulated from actual power, and remaining requested energy
is the original request minus measured delivery, clipped at zero (also zero after
a completion signal). This is request accounting, not an estimate of hidden true
battery demand. Tracking errors and the observed peak are available for analysis.

The opt-in `run_feedback_mpc` controller in `sim/feedback_mpc.py` executes every slot but still solves on its existing periodic timer. Future
optimizations fix the observed past. Completed sessions have their energy target
set to observed delivered energy. Other demand and price assumptions are unchanged.

```python
from evopt.sim.plant import ChargingPlant
from evopt.sim.feedback_mpc import run_feedback_mpc

plant = ChargingPlant(sessions, site, demand=true_demand, response_fraction=0.8)
actual, feedback = run_feedback_mpc(
    sessions, site, price_model, realized_prices_per_period, terms,
    plant=plant, return_feedback=True,
)
```

Omitting `realized_demand` keeps a custom plant's demand private. Passing it retains
the original perfect demand-revelation assumption at arrival replans. The existing `sim/mpc.py`, experiment runner, and default experiments are unchanged;
they do not use this new controller. Call `run_feedback_mpc` explicitly to opt in. The
return value remains the actual power matrix unless `return_feedback=True`.

This first step does not implement event triggers, online arrival forecasts,
adaptive charging-limit estimation, historical ACN telemetry replay, or battery
tapering. Historical current measurements should calibrate a response model;
they are not counterfactual responses to newly optimized commands.
