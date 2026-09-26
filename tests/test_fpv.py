import hashlib
import importlib.util
import unittest


@unittest.skipUnless(importlib.util.find_spec("mujoco"), "install fly-drone[fpv] for MuJoCo tests")
class FPVPhysicsTests(unittest.TestCase):
    def test_geometric_steering_passes_gate_with_stable_altitude(self):
        from fly_drone.fpv import TARGETS, fly_fpv

        result = fly_fpv(lambda angle: max(-1.0, min(1.0, angle / 0.7)), TARGETS["left"])
        self.assertEqual(result["status"], "passed_gate")
        self.assertLess(max(abs(frame["position"][2] - 1.2) for frame in result["trace"]), 0.15)
        self.assertTrue(any(max(abs(angle) for angle in frame["rpy"][:2]) > 0.05
                            for frame in result["trace"]))

    def test_straight_flight_misses_offset_gate(self):
        from fly_drone.fpv import TARGETS, fly_fpv

        result = fly_fpv(lambda _angle: 0.0, TARGETS["left"])
        self.assertEqual(result["status"], "missed_gate")

    def test_same_fpv_frame_reports_opposite_gate_bearings(self):
        from fly_drone.fpv import FPVQuad, TARGETS
        from fly_drone.vision import GateVision

        quad = FPVQuad()
        with GateVision("left") as left, GateVision("right") as right:
            left_image = left.observe(quad, TARGETS["left"], 0)
            right_image = right.observe(quad, TARGETS["right"], 0)
        self.assertEqual(left_image["frame_sha256"], right_image["frame_sha256"])
        self.assertTrue(left_image["visible"] and right_image["visible"])
        self.assertGreater(left_image["bearing"], 0.5)
        self.assertLess(right_image["bearing"], -0.5)

    def test_camera_guided_flight_with_dropped_frames(self):
        from fly_drone.cli import geometric_controller
        from fly_drone.fpv import TARGETS, fly_fpv
        from fly_drone.vision import GateVision

        with GateVision("left", scenario="dropout_30", seed=2026) as sensor:
            result = fly_fpv(geometric_controller, TARGETS["left"], sensor=sensor)
        self.assertEqual(result["status"], "passed_gate")
        self.assertTrue(any(obs["reason"] == "dropout" for obs in result["control_observations"]))
        self.assertEqual(result["decision_budget_ms"], 400)

    def test_rgb_controller_receives_frames_without_detector_angle(self):
        from fly_drone.fpv import TARGETS, fly_fpv
        from fly_drone.vision import GateVision

        received = []

        def controller(frame):
            received.append(frame)
            return 0.0

        with GateVision("left", include_frame=True, detect=False) as sensor:
            result = fly_fpv(controller, TARGETS["left"], sensor=sensor,
                             duration=0.8, frame_input=True, visible_gate="left")
        self.assertEqual(len(received), 2)
        self.assertTrue(all(frame.shape == (180, 320, 3) for frame in received))
        self.assertTrue(all("frame" not in obs for obs in result["control_observations"]))
        self.assertTrue(all(obs["reason"] == "raw_frame" for obs in result["control_observations"]))
        self.assertEqual(result["control_observations"][0]["frame_sha256"],
                         hashlib.sha256(received[0].tobytes()).hexdigest())

    def test_moving_target_is_seen_and_followed_by_geometry(self):
        from fly_drone.chase import fly_chase
        from fly_drone.cli import geometric_controller

        result = fly_chase(geometric_controller, duration=18.0)
        self.assertEqual(result["status"], "completed")
        self.assertTrue(result["tracked"])
        self.assertGreater(result["visible_decisions"] / result["total_decisions"], 0.9)
        self.assertGreater(result["trace"][-1]["cow_position"][0], 12.0)

    def test_tracker_diagnostics_do_not_reach_odor_controller(self):
        from fly_drone.chase import fly_chase

        class DiagnosticPlume:
            def sample(self, *_args):
                return {"L": 0.1, "R": 0.2, "upwind_bearing": 0.0,
                        "source_xy": [99.0, 99.0]}

        received = []

        def controller(_bearing, sample):
            received.append(sample.copy())
            return 0.0, {}

        result = fly_chase(controller, duration=0.4,
                           odor_perception=DiagnosticPlume(), speed_source="fixed")
        self.assertEqual(received, [{"L": 0.1, "R": 0.2, "upwind_bearing": 0.0}])
        self.assertEqual(result["control_observations"][0]["odor_sensor"]["source_xy"],
                         [99.0, 99.0])


if __name__ == "__main__":
    unittest.main()
