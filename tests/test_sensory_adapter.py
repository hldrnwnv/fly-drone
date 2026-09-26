import unittest

import numpy as np

from fly_drone.perception import VisualFeatures
from fly_drone.sensory_adapter import FlySensoryEncoder


class FakeBrain:
    def cells(self, names, *, side):
        length = 24 if names == ["SApp08"] else 100
        return np.arange(length) + (1000 if side == "R" else 0)


class SensoryAdapterTests(unittest.TestCase):
    def setUp(self):
        self.encoder = FlySensoryEncoder(FakeBrain())

    def test_gyro_keeps_three_axes_and_signs_separate(self):
        encoded = self.encoder.gyro(np.array([0.5, -1.0, 2.5]))
        self.assertEqual(encoded.channels["roll_R_drive"], 0.5)
        self.assertEqual(encoded.channels["pitch_L_drive"], 1.0)
        self.assertEqual(encoded.channels["yaw_R_drive"], 2.0)
        self.assertEqual(len(encoded.inject), 3)
        self.assertTrue(np.array_equal(encoded.inject[0][0],
                                       self.encoder.gyro_cells["roll", "R"]))

    def test_motion_and_relative_depth_are_explicit_channels(self):
        flow = np.zeros((16, 24, 2), dtype=np.float32)
        flow[..., 0] = 2.0
        depth = np.tile(np.linspace(0, 1, 24), (16, 1)).astype(np.float32)
        encoded = self.encoder.vision(VisualFeatures(flow, depth, 12.0))
        self.assertEqual(encoded.channels["flow_available"], 1.0)
        self.assertAlmostEqual(encoded.channels["flow_L_a_drive"], 0.2)
        self.assertEqual(encoded.channels["flow_L_b_drive"], 0)
        self.assertGreater(encoded.channels["loom_R_drive"], 0)
        self.assertGreater(len(encoded.inject), 0)

    def test_first_frame_has_no_fabricated_flow(self):
        encoded = self.encoder.vision(VisualFeatures(None, np.ones((4, 4)), 2.0))
        self.assertEqual(encoded.channels["flow_available"], 0.0)
        self.assertEqual(encoded.inject, [])

    def test_relative_motion_is_separate_from_flow_only_input(self):
        depth = np.zeros((16, 24), dtype=np.float32)
        depth[:8] = 1.0
        flow = np.zeros((16, 24, 2), dtype=np.float32)
        flow[:8, :, 0] = 4.0
        flow[8:, :, 0] = 1.0
        combined = self.encoder.vision(VisualFeatures(flow, depth, 12.0))
        flow_only = self.encoder.vision(VisualFeatures(flow, np.ones_like(depth), 12.0))
        self.assertAlmostEqual(combined.channels["parallax_L_drive"], 0.3)
        self.assertEqual(flow_only.channels["parallax_L_drive"], 0.0)
        self.assertEqual(flow_only.channels["loom_L_drive"], 0.0)
        self.assertEqual(combined.channels["flow_L_a_drive"],
                         flow_only.channels["flow_L_a_drive"])


if __name__ == "__main__":
    unittest.main()
