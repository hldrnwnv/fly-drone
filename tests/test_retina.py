import unittest

import numpy as np

from fly_drone.retina import RetinaProjector


class RetinaProjectorTests(unittest.TestCase):
    def test_photoreceptor_positions_follow_image_left_to_right(self):
        frame = np.zeros((8, 8, 3), dtype=np.uint8)
        frame[:, :4] = 255
        projector = RetinaProjector(np.array([-1.0, -0.5, 0.5, 1.0]))
        signal = projector.encode(frame)
        self.assertGreater(signal[0], 0.99)
        self.assertGreater(signal[1], 0.99)
        self.assertLess(signal[2], 0.01)
        self.assertLess(signal[3], 0.01)

    def test_moving_bright_region_changes_drive_with_order(self):
        left = np.zeros((8, 8, 3), dtype=np.uint8)
        right = left.copy()
        left[:, :2] = 255
        right[:, -2:] = 255
        projector = RetinaProjector(np.array([-1.0, 1.0]))
        self.assertEqual(projector.encode(left).tolist(), [1.0, 0.0])
        self.assertEqual(projector.encode(right).tolist(), [0.0, 1.0])


if __name__ == "__main__":
    unittest.main()
