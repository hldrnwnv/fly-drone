"""Follow a moving visual target with the same FPV quad and steering readout."""

from __future__ import annotations

import hashlib
from math import atan, atan2, cos, radians, sin, tan
from time import perf_counter
from typing import Callable

import mujoco
import numpy as np

from .fpv import FPVQuad, Stabilizer
from .sim import wrap_angle


def cow_pose(t: float) -> tuple[tuple[float, float], float]:
    """A deterministic forward path with continuous lateral turns."""
    x = 2.2 + 0.62 * t
    y = 1.0 * sin(0.5 * t)
    yaw = atan2(0.5 * cos(0.5 * t), 0.62)
    return (x, y), yaw


class CowVision:
    width = 320
    height = 180
    vertical_fov_degrees = 95
    tag_width_m = 0.56

    def __init__(self):
        self.renderer = None
        self.last_bearing = 0.0
        self.last_range = 2.0
        self.last_image = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if self.renderer is not None:
            self.renderer.close()

    def observe(self, quad: FPVQuad) -> dict:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(quad.model, width=self.width, height=self.height)
        self.renderer.update_scene(quad.data, camera="fpv")
        image = self.renderer.render()
        self.last_image = image.copy()
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        pixels = image.astype(np.int16)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        mask = (red > 150) & (blue > 125) & (red > 1.45 * green) & (blue > 1.35 * green)
        rows, columns = np.nonzero(mask)
        if len(columns) < 16:
            return {"bearing": self.last_bearing, "range_m": self.last_range,
                    "visible": False, "reason": "tag_not_visible", "frame_sha256": digest}
        x0, x1 = int(columns.min()), int(columns.max())
        y0, y1 = int(rows.min()), int(rows.max())
        focal_px = (self.height / 2) / tan(radians(self.vertical_fov_degrees) / 2)
        self.last_bearing = -atan(((x0 + x1) / 2 - self.width / 2) / focal_px)
        self.last_range = float(np.clip(self.tag_width_m * focal_px / max(x1 - x0 + 1, 1),
                                        0.5, 12.0))
        return {"bearing": self.last_bearing, "range_m": self.last_range,
                "visible": True, "reason": "detected", "bbox": [x0, y0, x1, y1],
                "tag_pixels": len(columns), "frame_sha256": digest}


def fly_chase(controller, *, duration: float = 18.0, decision_interval: float = 0.4,
              joint_control: bool = False, speed_source: str | None = None,
              fixed_speed_m_s: float = 0.65,
              frame_sink: Callable[[float, np.ndarray], None] | None = None,
              visual_perception=None, fast_frame_interval: float | None = None,
              fast_frame_sink: Callable[[float, np.ndarray], None] | None = None,
              odor_perception=None,
              initial_xy: tuple[float, float] = (0.0, 0.0),
              target_pose: Callable[[float], tuple[tuple[float, float], float]] = cow_pose,
              visual_dropout: Callable[[float], bool] | None = None,
              odor_visual_context: bool = False) -> dict:
    """Camera observations to yaw and speed, then shared four-motor stabilizer."""
    if joint_control and visual_perception is not None:
        raise ValueError("joint_control and visual_perception cannot be combined")
    if odor_perception is not None and (joint_control or visual_perception is not None):
        raise ValueError("odor perception requires the single-output chase controller")
    if not 0.0 <= fixed_speed_m_s <= 1.5:
        raise ValueError("fixed_speed_m_s must be within the stabilizer speed limits")
    quad = FPVQuad(show_cow=True, show_gates=False)
    quad.data.qpos[:2] = initial_xy
    mujoco.mj_forward(quad.model, quad.data)
    autopilot = Stabilizer(altitude=1.2, forward_speed=0.65)
    control_steps = round(decision_interval / quad.dt)
    if control_steps < 1:
        raise ValueError("decision_interval must be at least one physics step")
    fast_steps = None
    if fast_frame_interval is not None:
        fast_steps = round(fast_frame_interval / quad.dt)
        if (fast_steps < 1 or control_steps % fast_steps or
                abs(fast_steps * quad.dt - fast_frame_interval) > 1e-6):
            raise ValueError("fast_frame_interval must divide the decision interval")
        if visual_perception is None:
            raise ValueError("fast frames require visual_perception")
    fast_frames: list[np.ndarray] = []
    fast_observation = None
    trace = []
    observations = []
    status = "completed"
    yaw_rate = 0.0
    command = 0.0
    speed_command = autopilot.forward_speed
    observation = None
    with CowVision() as camera:
        for step in range(round(duration / quad.dt)):
            t = float(quad.data.time)
            cow_xy, cow_yaw = target_pose(t)
            quad.set_cow_pose(cow_xy, cow_yaw)
            state = quad.state()
            if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
                status = "crashed"
                break
            if fast_steps is not None and step % fast_steps == 0:
                fast_observation = camera.observe(quad)
                fast_frames.append(camera.last_image.copy())
                if fast_frame_sink is not None:
                    fast_frame_sink(t, camera.last_image.copy())
            if step % control_steps == 0:
                decision_start = perf_counter()
                observation = (fast_observation if fast_steps is not None
                               else camera.observe(quad))
                if frame_sink is not None:
                    frame_sink(t, camera.last_image.copy())
                if visual_dropout is not None and visual_dropout(t):
                    observation = {**observation, "bearing": 0.0, "range_m": 1.7,
                                   "visible": False, "reason": "scripted_visual_dropout"}
                odor_sample = (odor_perception.sample(state.position[:2], state.yaw,
                                                     np.asarray(cow_xy), t)
                               if odor_perception is not None else None)
                if odor_sample is not None:
                    if odor_visual_context:
                        odor_sample["tag_visible"] = observation["visible"]
                    observation["odor_sensor"] = odor_sample
                true_bearing = wrap_angle(atan2(cow_xy[1] - state.position[1],
                                                cow_xy[0] - state.position[0]) - state.yaw)
                if joint_control:
                    wanted_yaw, wanted_speed = controller(observation["bearing"],
                                                          observation["range_m"])
                    command = float(np.clip(wanted_yaw, -1.0, 1.0))
                    speed_command = float(np.clip(wanted_speed, 0.0, 1.5))
                else:
                    odor_speed = None
                    if visual_perception is not None:
                        visual_features = visual_perception.observe(camera.last_image,
                                                                    time_s=t)
                        if fast_steps is not None:
                            wanted_yaw, neural_inputs = controller(
                                observation["bearing"], visual_features, fast_frames)
                            fast_frames = []
                        else:
                            wanted_yaw, neural_inputs = controller(
                                observation["bearing"], visual_features)
                        observation["visual_input"] = neural_inputs
                        observation["visual_inference_ms"] = visual_features.inference_ms
                        observation["visual_flow_valid"] = visual_features.flow_xy_px is not None
                    else:
                        if odor_sample is not None:
                            controller_sample = {key: odor_sample[key]
                                                 for key in ("L", "R", "upwind_bearing",
                                                             "tag_visible") if key in odor_sample}
                            odor_result = controller(observation["bearing"], controller_sample)
                            if len(odor_result) == 3:
                                wanted_yaw, odor_speed, odor_input = odor_result
                            else:
                                wanted_yaw, odor_input = odor_result
                            observation["odor_input"] = odor_input
                        else:
                            wanted_yaw = controller(observation["bearing"])
                    command = float(np.clip(wanted_yaw, -1.0, 1.0))
                    if speed_source == "controller":
                        if odor_speed is None:
                            raise ValueError("controller speed source requires odor speed")
                        speed_command = float(np.clip(odor_speed, 0.0, 1.5))
                    elif speed_source == "fixed":
                        speed_command = fixed_speed_m_s
                    else:
                        speed_command = float(np.clip(
                            0.65 + 0.75 * (observation["range_m"] - 1.7), 0.0, 1.5))
                        if not observation["visible"]:
                            speed_command = min(speed_command, 0.8)
                yaw_rate = 1.1 * command
                autopilot.forward_speed = speed_command
                observations.append({"t": t, "true_bearing": true_bearing,
                                     "true_center_distance_m": float(np.linalg.norm(
                                         np.asarray(cow_xy) - state.position[:2])),
                                     **observation, "steering_command": command,
                                     "speed_command_m_s": speed_command,
                                     "decision_ms": 1000 * (perf_counter() - decision_start)})
            motors = quad.apply(*autopilot.command(state, yaw_rate))
            if step % 8 == 0:
                current = quad.state()
                relative = np.asarray(cow_xy) - current.position[:2]
                cow_heading = np.array([cos(cow_yaw), sin(cow_yaw)])
                cow_lateral = np.array([-cow_heading[1], cow_heading[0]])
                trace.append({"t": round(float(quad.data.time), 4),
                              "position": current.position.tolist(),
                              "quaternion": quad.data.qpos[3:7].tolist(),
                              "velocity": current.velocity.tolist(),
                              "rpy": [current.roll, current.pitch, current.yaw],
                              "motors_N": motors.tolist(), "neural_command": command,
                              "speed_command_m_s": speed_command,
                              "measured_bearing": observation["bearing"] if observation else None,
                              "target_visible": observation["visible"] if observation else None,
                              "cow_position": [cow_xy[0], cow_xy[1], 0.0], "cow_yaw": cow_yaw,
                              "cow_distance_m": float(np.linalg.norm(relative)),
                              "behind_distance_m": float(relative @ cow_heading),
                              "lateral_error_m": abs(float(relative @ cow_lateral))})
    evaluated = [frame for frame in trace if frame["t"] >= 4.0]
    in_follow_band = [frame for frame in evaluated
                      if 1.4 <= frame["behind_distance_m"] <= 3.5
                      and frame["lateral_error_m"] < 1.0]
    follow_fraction = len(in_follow_band) / len(evaluated) if evaluated else 0.0
    resolved_speed_source = speed_source or ("connectome_readout" if joint_control else
                                             "image_range_rule")
    return {"kind": "chase", "status": status,
            "speed_source": resolved_speed_source,
            "tracked": status == "completed" and follow_fraction >= 0.7,
            "follow_fraction_after_4s": follow_fraction,
            "mean_lateral_error_after_4s_m": float(np.mean(
                [frame["lateral_error_m"] for frame in evaluated])) if evaluated else None,
            "mean_cow_distance_after_4s_m": float(np.mean(
                [frame["cow_distance_m"] for frame in evaluated])) if evaluated else None,
            "visible_decisions": sum(obs["visible"] for obs in observations),
            "total_decisions": len(observations),
            "decision_mean_ms": float(np.mean([obs["decision_ms"] for obs in observations]))
            if observations else None,
            "decision_deadline_misses": sum(obs["decision_ms"] > 1000 * decision_interval
                                            for obs in observations),
            "duration": float(quad.data.time), "trace": trace,
            "control_observations": observations,
            "control_roles": ("connectome readout: yaw and forward speed; stabilizer: four motor thrusts"
                              if resolved_speed_source == "connectome_readout" else
                              "neural/geometric yaw; image-based rule: forward speed; stabilizer: four motor thrusts")}


def fly_wing_direct(controller, *, duration: float = 8.0,
                    decision_interval: float = 0.4,
                    altitude_assist: bool = False, perception=None) -> dict:
    """Drive all four rotor thrusts from named wing motor cell spikes."""
    quad = FPVQuad(show_cow=True, show_gates=False)
    control_steps = round(decision_interval / quad.dt)
    if control_steps < 1:
        raise ValueError("decision_interval must be at least one physics step")
    trace: list[dict] = []
    observations: list[dict] = []
    status = "completed"
    motors = np.zeros(4)
    observation = None
    if perception is not None:
        perception.reset()
    with CowVision() as camera:
        for step in range(round(duration / quad.dt)):
            t = float(quad.data.time)
            cow_xy, cow_yaw = cow_pose(t)
            quad.set_cow_pose(cow_xy, cow_yaw)
            state = quad.state()
            if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
                status = "crashed"
                break
            if step % control_steps == 0:
                started = perf_counter()
                observation = camera.observe(quad)
                visual_features = (perception.observe(camera.last_image, time_s=t)
                                   if perception is not None else None)
                sample = controller.command(observation["bearing"], state.body_rates,
                                            camera.last_image, mass_kg=quad.mass,
                                            mixer=quad.mixer,
                                            visual_features=visual_features)
                motors = np.asarray(sample.motor_forces_n)
                observations.append({"t": t, **observation,
                                     "true_bearing_rad": wrap_angle(atan2(
                                         cow_xy[1] - state.position[1],
                                         cow_xy[0] - state.position[0]) - state.yaw),
                                     "motor_spikes": sample.spike_counts,
                                     "motors_N": sample.motor_forces_n,
                                     "wing_intent": sample.wing_intent,
                                     "desired_wrench": sample.desired_wrench,
                                     "achieved_wrench": sample.achieved_wrench,
                                     "haltere_drive": sample.haltere_drive,
                                     "visual_drive": sample.visual_drive,
                                     "sensory_drive": sample.sensory_drive,
                                     "body_rates_rad_s": state.body_rates.tolist(),
                                     "decision_ms": 1000 * (perf_counter() - started)})
            # Diagnostic: add only collective vertical feedback. The neural
            # left/right and fore/aft differentials remain uncorrected.
            correction = (0.75 * (4.0 * (1.2 - state.position[2]) -
                                  2.8 * state.velocity[2])) if altitude_assist else 0.0
            applied_motors = np.clip(motors + correction / 4, 0.0, 8.0)
            quad.data.ctrl[:] = applied_motors
            mujoco.mj_step(quad.model, quad.data)
            if step % 8 == 0:
                current = quad.state()
                relative = np.asarray(cow_xy) - current.position[:2]
                heading = np.array([cos(cow_yaw), sin(cow_yaw)])
                lateral = np.array([-heading[1], heading[0]])
                trace.append({"t": round(float(quad.data.time), 4),
                              "position": current.position.tolist(),
                              "quaternion": quad.data.qpos[3:7].tolist(),
                              "velocity": current.velocity.tolist(),
                              "rpy": [current.roll, current.pitch, current.yaw],
                              "motors_N": applied_motors.tolist(),
                              "raw_neural_motors_N": motors.tolist(),
                              "altitude_correction_total_N": float(correction),
                              "measured_bearing": observation["bearing"] if observation else None,
                              "target_visible": observation["visible"] if observation else None,
                              "cow_position": [cow_xy[0], cow_xy[1], 0.0],
                              "cow_yaw": cow_yaw,
                              "cow_distance_m": float(np.linalg.norm(relative)),
                              "behind_distance_m": float(relative @ heading),
                              "lateral_error_m": abs(float(relative @ lateral))})
    evaluated = [frame for frame in trace if frame["t"] >= 4.0]
    follow_fraction = (sum(1.4 <= frame["behind_distance_m"] <= 3.5
                           and frame["lateral_error_m"] < 1.0 for frame in evaluated) /
                       len(evaluated)) if evaluated else 0.0
    return {"kind": "chase", "status": status, "duration": float(quad.data.time),
            "tracked": status == "completed" and follow_fraction >= 0.7,
            "follow_fraction_after_4s": follow_fraction, "trace": trace,
            "control_observations": observations,
            "control_roles": ("MaleCNS wing signals to four rotor thrusts plus diagnostic "
                              "altitude-only feedback" if altitude_assist else
                              f"MaleCNS wing signals via {controller.mapping} mapping; no stabilizer")}
