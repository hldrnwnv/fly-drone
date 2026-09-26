"""MuJoCo 6-DOF FPV quad with four thrust actuators and a rate stabilizer."""

from __future__ import annotations

from dataclasses import dataclass
from math import atan2, asin, cos, sin
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from .sim import wrap_angle


MODEL_PATH = Path(__file__).resolve().parents[2] / "assets" / "fpv_quad.xml"
TARGETS = {"left": (4.0, 3.0, 1.2), "right": (4.0, -3.0, 1.2)}


@dataclass(frozen=True)
class QuadState:
    position: np.ndarray
    velocity: np.ndarray
    roll: float
    pitch: float
    yaw: float
    body_rates: np.ndarray


class FPVQuad:
    def __init__(self, *, show_cow: bool = False, show_gates: bool = True):
        self.model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
        self.data = mujoco.MjData(self.model)
        self.cow_mocap_id = int(self.model.body_mocapid[
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "cow")])
        for geom_id in range(self.model.ngeom):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
            if name.startswith("cow_"):
                self.model.geom_rgba[geom_id, 3] = float(show_cow)
            elif name.startswith(("left_", "right_")) and not show_gates:
                self.model.geom_rgba[geom_id, 3] = 0.0
        mujoco.mj_forward(self.model, self.data)
        self.dt = float(self.model.opt.timestep)
        self.mass = 0.75
        # Rows: total thrust, roll, pitch, yaw torque. Columns: FL, FR, RR, RL.
        arm, reaction = 0.13, 0.012
        self.mixer = np.array([[1, 1, 1, 1],
                               [arm, -arm, -arm, arm],
                               [-arm, -arm, arm, arm],
                               [reaction, -reaction, reaction, -reaction]], dtype=float)

    def set_cow_pose(self, xy: tuple[float, float], yaw: float) -> None:
        self.data.mocap_pos[self.cow_mocap_id] = [xy[0], xy[1], 0.0]
        self.data.mocap_quat[self.cow_mocap_id] = [cos(yaw / 2), 0, 0, sin(yaw / 2)]
        mujoco.mj_forward(self.model, self.data)

    def state(self) -> QuadState:
        q = self.data.qpos
        matrix = np.empty(9, dtype=float)
        mujoco.mju_quat2Mat(matrix, q[3:7])
        rotation = matrix.reshape(3, 3)
        pitch = asin(float(np.clip(-rotation[2, 0], -1.0, 1.0)))
        return QuadState(position=q[:3].copy(), velocity=self.data.qvel[:3].copy(),
                         roll=atan2(rotation[2, 1], rotation[2, 2]), pitch=pitch,
                         yaw=atan2(rotation[1, 0], rotation[0, 0]),
                         body_rates=self.data.qvel[3:6].copy())

    def apply(self, total_thrust: float, roll_torque: float, pitch_torque: float,
              yaw_torque: float) -> np.ndarray:
        requested = np.linalg.solve(self.mixer,
                                    [total_thrust, roll_torque, pitch_torque, yaw_torque])
        motors = np.clip(requested, 0.0, 8.0)
        self.data.ctrl[:] = motors
        mujoco.mj_step(self.model, self.data)
        return motors


class Stabilizer:
    """Hold altitude and forward speed while following an external yaw-rate signal."""

    def __init__(self, *, altitude: float = 1.2, forward_speed: float = 1.1):
        self.altitude = altitude
        self.forward_speed = forward_speed

    def command(self, state: QuadState, yaw_rate: float) -> tuple[float, float, float, float]:
        heading = np.array([cos(state.yaw), sin(state.yaw)])
        wanted_velocity = self.forward_speed * heading
        wanted_accel = np.clip(1.8 * (wanted_velocity - state.velocity[:2]), -2.4, 2.4)
        # Express world velocity error in body coordinates, then tilt the thrust vector.
        body_ax = cos(state.yaw) * wanted_accel[0] + sin(state.yaw) * wanted_accel[1]
        body_ay = -sin(state.yaw) * wanted_accel[0] + cos(state.yaw) * wanted_accel[1]
        wanted_roll = float(np.clip(-body_ay / 9.81, -0.28, 0.28))
        wanted_pitch = float(np.clip(body_ax / 9.81, -0.28, 0.28))
        vertical_accel = 4.0 * (self.altitude - state.position[2]) - 2.8 * state.velocity[2]
        tilt_cos = max(cos(state.roll) * cos(state.pitch), 0.55)
        total_thrust = float(np.clip(0.75 * (9.81 + vertical_accel) / tilt_cos, 0.0, 28.0))
        roll_torque = float(np.clip(0.04 * (wanted_roll - state.roll) -
                                    0.025 * state.body_rates[0], -0.16, 0.16))
        pitch_torque = float(np.clip(0.04 * (wanted_pitch - state.pitch) -
                                     0.025 * state.body_rates[1], -0.16, 0.16))
        yaw_torque = float(np.clip(0.03 * (yaw_rate - state.body_rates[2]), -0.09, 0.09))
        return total_thrust, roll_torque, pitch_torque, yaw_torque


def fly_fpv(controller, target: tuple[float, float, float], *, duration: float = 14.0,
            brain_interval: float = 0.4, sensor=None,
            initial_xy: tuple[float, float] = (0.0, 0.0), initial_yaw: float = 0.0) -> dict:
    quad = FPVQuad()
    quad.data.qpos[:2] = initial_xy
    quad.data.qpos[3:7] = [cos(initial_yaw / 2), 0, 0, sin(initial_yaw / 2)]
    mujoco.mj_forward(quad.model, quad.data)
    autopilot = Stabilizer(altitude=target[2])
    control_steps = round(brain_interval / quad.dt)
    if control_steps < 1:
        raise ValueError("brain_interval must be at least one physics step")
    trace = []
    yaw_rate = 0.0
    command = 0.0
    status = "timeout"
    controller_wall_s = 0.0
    decision_wall_s = 0.0
    deadline_misses = 0
    controller_calls = 0
    control_observations = []
    observation = None
    gate_heading = np.asarray(target[:2], dtype=float)
    gate_heading /= np.linalg.norm(gate_heading)
    gate_lateral = np.array([-gate_heading[1], gate_heading[0]])
    for step in range(round(duration / quad.dt)):
        state = quad.state()
        relative_xy = state.position[:2] - np.asarray(target[:2])
        if float(relative_xy @ gate_heading) >= 0:
            inside_gate = abs(float(relative_xy @ gate_lateral)) < 0.62 and 0.3 < state.position[2] < 2.2
            status = "passed_gate" if inside_gate else "missed_gate"
            break
        if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
            status = "crashed"
            break
        if step % control_steps == 0:
            decision_start = perf_counter()
            true_bearing = wrap_angle(atan2(target[1] - state.position[1],
                                            target[0] - state.position[0]) - state.yaw)
            observation = (sensor.observe(quad, target, controller_calls) if sensor is not None else
                           {"bearing": true_bearing, "visible": True, "reason": "state_sensor"})
            bearing = observation["bearing"]
            start = perf_counter()
            command = float(np.clip(controller(bearing), -1.0, 1.0))
            controller_wall_s += perf_counter() - start
            decision_ms = 1000 * (perf_counter() - decision_start)
            decision_wall_s += decision_ms / 1000
            deadline_misses += decision_ms > 1000 * brain_interval
            control_observations.append({"t": float(quad.data.time), "true_bearing": true_bearing,
                                         **observation, "command": command, "decision_ms": decision_ms})
            controller_calls += 1
            yaw_rate = 1.1 * command
        motors = quad.apply(*autopilot.command(state, yaw_rate))
        if step % 8 == 0:
            s = quad.state()
            trace.append({"t": round(quad.data.time, 4), "position": s.position.tolist(),
                          "quaternion": quad.data.qpos[3:7].tolist(), "velocity": s.velocity.tolist(),
                          "rpy": [s.roll, s.pitch, s.yaw], "motors_N": motors.tolist(),
                          "neural_command": command,
                          "measured_bearing": observation["bearing"] if observation else None,
                          "gate_visible": observation["visible"] if observation else None,
                          "distance": float(np.linalg.norm(s.position - np.asarray(target)))})
    final = quad.state()
    return {"target": list(target), "initial_xy": list(initial_xy), "initial_yaw": initial_yaw,
            "status": status, "reached": status == "passed_gate",
            "final_distance": float(np.linalg.norm(final.position - np.asarray(target))),
            "duration": float(quad.data.time), "trace": trace,
            "controller_calls": controller_calls,
            "controller_wall_s": controller_wall_s,
            "controller_mean_ms": 1000 * controller_wall_s / controller_calls if controller_calls else None,
            "decision_mean_ms": 1000 * decision_wall_s / controller_calls if controller_calls else None,
            "decision_deadline_misses": deadline_misses,
            "decision_budget_ms": 1000 * brain_interval,
            "gate_lateral_error_m": abs(float((final.position[:2] - np.asarray(target[:2])) @ gate_lateral)),
            "control_observations": control_observations,
            "model": "MuJoCo four-thrust 0.75 kg FPV quad; assisted altitude and forward-speed control"}
