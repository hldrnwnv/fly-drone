import unittest

import numpy as np

from fly_drone.retina_2d import RetinaProjector2D, normalized_eye_coordinates


class Retina2DTests(unittest.TestCase):
    def test_source_hex_axes_form_two_image_axes_per_eye(self):
        columns = np.array([[18, 19], [19, 19], [18, 20],
                            [18, 19], [19, 19], [18, 20]], dtype=float)
        xy = normalized_eye_coordinates(columns, np.array(list("LLLRRR")))
        self.assertGreater(xy[1, 0], xy[0, 0])
        self.assertLess(xy[2, 0], xy[0, 0])
        self.assertLess(xy[1, 1], xy[0, 1])
        np.testing.assert_array_equal(xy[:3], xy[3:])

    def test_vertical_content_and_eye_mask_change_only_located_receptors(self):
        xy = np.array([[0.5, 0.0], [0.5, 1.0], [np.nan, np.nan],
                       [0.5, 0.0], [0.5, 1.0]], dtype=np.float32)
        projector = RetinaProjector2D(xy, np.array(list("LLLRR")), background=0.1)
        image = np.zeros((4, 4, 3), dtype=np.uint8)
        image[0] = 255
        left = projector.encode(image, eye="L")
        np.testing.assert_allclose(left, [1, 0, 0.1, 0.1, 0.1], atol=1e-6)
        right = projector.encode(image, eye="R")
        np.testing.assert_allclose(right, [0.1, 0.1, 0.1, 1, 0], atol=1e-6)


if __name__ == "__main__":
    unittest.main()
