import unittest

import numpy as np

from fly_drone.wing_adapter import WingIntent, WingRotorAdapter


MIXER = np.array([[1, 1, 1, 1],
                  [0.13, -0.13, -0.13, 0.13],
                  [-0.13, -0.13, 0.13, 0.13],
                  [0.012, -0.012, 0.012, -0.012]])


class WingAdapterTests(unittest.TestCase):
    def test_neutral_wings_hover_at_each_quad_mass(self):
        for mass in (0.5, 0.75, 1.0):
            with self.subTest(mass=mass):
                result = WingRotorAdapter(mass_kg=mass, mixer=MIXER).mix(
                    WingIntent(1.0, 0.0, 0.0, 0.0, 0.0))
                self.assertTrue(np.allclose(result.thrusts_n, mass * 9.81 / 4))
                self.assertTrue(np.allclose(result.achieved_wrench,
                                            [mass * 9.81, 0, 0, 0]))

    def test_wing_asymmetries_map_to_distinct_body_axes(self):
        adapter = WingRotorAdapter(mass_kg=0.75, mixer=MIXER)
        roll = adapter.mix(WingIntent(1.0, 0, 0, 1, -1))
        yaw = adapter.mix(WingIntent(1.0, 1, -1, 0, 0))
        self.assertGreater(roll.achieved_wrench[1], 0)
        self.assertAlmostEqual(roll.achieved_wrench[3], 0)
        self.assertGreater(yaw.achieved_wrench[3], 0)
        self.assertAlmostEqual(yaw.achieved_wrench[1], 0)
        for result in (roll, yaw):
            self.assertTrue(np.allclose(MIXER @ result.thrusts_n,
                                        result.achieved_wrench))


if __name__ == "__main__":
    unittest.main()
