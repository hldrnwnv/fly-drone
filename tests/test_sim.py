import math
import unittest

from fly_drone.sim import Drone, fly_path, wrap_angle


class DroneSimulationTests(unittest.TestCase):
    def test_bearing_and_turn_direction(self):
        drone = Drone()
        target = (0.0, 1.0)
        self.assertAlmostEqual(drone.bearing_to(target), math.pi / 2)
        drone.step(1.0, 0.2)
        self.assertGreater(drone.y, 0.0)
        self.assertLess(abs(drone.bearing_to(target)), math.pi / 2)

    def test_yaw_command_is_clipped(self):
        a, b = Drone(), Drone()
        a.step(4.0, 0.2)
        b.step(1.0, 0.2)
        self.assertEqual((a.x, a.y, a.yaw), (b.x, b.y, b.yaw))

    def test_path_stops_at_target(self):
        result = fly_path(lambda _angle: 0.0, (2.0, 0.0), steps=20)
        self.assertTrue(result["reached"])
        self.assertEqual(result["steps"], 8)

    def test_wrap_angle(self):
        self.assertAlmostEqual(wrap_angle(3 * math.pi), -math.pi)


if __name__ == "__main__":
    unittest.main()
