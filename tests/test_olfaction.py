import unittest
from math import pi

import numpy as np

from fly_drone.olfaction import FoodOdorPlume
from fly_drone.odor_navigation import OdorWindController, WindFoodOdorPlume


class FoodOdorPlumeTests(unittest.TestCase):
    def setUp(self):
        self.plume = FoodOdorPlume()
        self.source = np.array([2.0, 0.0])

    def test_downwind_receives_odor_but_upwind_does_not(self):
        self.assertGreater(self.plume.concentration(np.array([0.0, 0.0]),
                                                    self.source, 0.0), 0.1)
        self.assertEqual(self.plume.concentration(np.array([3.0, 0.0]),
                                                  self.source, 0.0), 0.0)

    def test_bilateral_samples_reflect_crosswind_offset(self):
        centered = self.plume.sample(np.array([0.0, 0.0]), 0.0,
                                     self.source, 0.0)
        offset = self.plume.sample(np.array([0.0, 0.3]), 0.0,
                                   self.source, 0.0)
        self.assertEqual(set(offset), {"L", "R"})
        self.assertNotEqual(offset["L"], offset["R"])
        self.assertGreater(sum(centered.values()), sum(offset.values()))

    def test_wind_vane_reports_body_relative_upwind_direction(self):
        sample = WindFoodOdorPlume().sample(np.array([0.0, 0.0]), pi / 2,
                                            self.source, 0.0)
        self.assertAlmostEqual(sample["upwind_bearing"], -pi / 2)

    def test_swapping_antennae_reverses_odor_turn_without_tag_bearing(self):
        class FakeReadout:
            last_counts = {}

            def decode(self, left, right):
                return left - right

        sample = {"L": 0.3, "R": 0.1, "upwind_bearing": 0.0}
        readout = FakeReadout()
        forward = OdorWindController(mode="neural_odor", readout=readout)
        swapped = OdorWindController(mode="neural_swapped", readout=readout)
        self.assertGreater(forward(1.0, sample)[0], 0.0)
        self.assertLess(swapped(1.0, sample)[0], 0.0)


if __name__ == "__main__":
    unittest.main()
