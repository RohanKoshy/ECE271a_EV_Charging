"""Physical response and observation-based accounting, independent of the solver."""
import unittest

import numpy as np
import pandas as pd

from evopt.data.demand import deliver
from evopt.data.schema import Site
from evopt.sim.plant import ChargingPlant, FeedbackState


class TestPlant(unittest.TestCase):
    def setUp(self):
        self.site = Site(slot_minutes=60, horizon_hours=4, capacity_kw=10)
        self.sessions = pd.DataFrame({"arrival": [0, 1], "departure": [4, 3],
                                      "max_rate": [6., 6.], "energy_requested": [10., 8.]})

    def test_ideal_matches_existing_cutoff(self):
        commands = np.array([[6., 4., 4., 4.], [0., 6., 4., 0.]])
        plant = ChargingPlant(self.sessions, self.site, [9., 7.])
        actual = np.column_stack([plant.step(commands[:, k]).actual_kw for k in range(4)])
        np.testing.assert_allclose(actual, deliver(commands, self.site.dt, [9., 7.]))

    def test_feedback_uses_actual_not_command(self):
        plant = ChargingPlant(self.sessions, self.site, [20., 20.], response_fraction=0.5)
        state = FeedbackState(self.sessions, self.site)
        state.update(plant.step([6., 0.]))
        np.testing.assert_allclose(state.delivered_kwh, [3., 0.])
        np.testing.assert_allclose(state.remaining_requested_kwh, [7., 8.])
        np.testing.assert_allclose(state.tracking_error_kw[:, 0], [3., 0.])
        self.assertEqual(state.peak_kw, 3.)

    def test_disconnect_completion_and_rate_limits(self):
        plant = ChargingPlant(self.sessions, self.site, [2., 20.])
        first = plant.step([8., 2.])
        np.testing.assert_allclose(first.actual_kw, [2., 0.])
        self.assertTrue(first.complete[0])
        np.testing.assert_allclose(plant.step([0., 10.]).actual_kw, [0., 6.])
        plant.step([0., 0.])
        np.testing.assert_allclose(plant.step([0., 6.]).actual_kw, [0., 0.])
        with self.assertRaises(ValueError):
            plant.step([0., 0.])

    def test_duplicate_observation_rejected(self):
        plant = ChargingPlant(self.sessions, self.site, [10., 8.])
        state = FeedbackState(self.sessions, self.site)
        observation = plant.step([5., 0.])
        state.update(observation)
        with self.assertRaises(ValueError):
            state.update(observation)

    def test_invalid_commands_rejected_without_advancing(self):
        plant = ChargingPlant(self.sessions, self.site, [10., 8.])
        for command in ([8., 8.], [-1., 0.], [np.nan, 0.]):
            with self.assertRaises(ValueError):
                plant.step(command)
        self.assertEqual(plant.next_slot, 0)
