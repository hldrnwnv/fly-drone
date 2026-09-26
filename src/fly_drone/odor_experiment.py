"""Paired held-out plume-history flights for research issue #3."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from math import atan2, cos, sin
from pathlib import Path

import numpy as np

from .chase import fly_chase
from .odor_navigation import (MaleCNSOdorReadout, OdorHistoryController,
                              OdorWindController, WindFoodOdorPlume)


@dataclass(frozen=True)
class OdorCase:
    name: str
    kind: str
    wind_angle: float
    phase_s: float
    seed: int
    source_xy: tuple[float, float] | None = None
    initial_y: float = 0.0
    path_phase: float = 0.0
    path_frequency: float = 0.45

    def pose(self, time_s: float) -> tuple[tuple[float, float], float]:
        if self.kind == "static":
            return self.source_xy, 0.0
        x = 2.4 + 0.57 * time_s
        y = 0.55 * sin(self.path_frequency * time_s + self.path_phase)
        y += 0.16 * sin(0.9 * time_s + 0.4 * self.path_phase)
        vy = 0.55 * self.path_frequency * cos(
            self.path_frequency * time_s + self.path_phase)
        vy += 0.144 * cos(0.9 * time_s + 0.4 * self.path_phase)
        return (x, y), atan2(vy, 0.57)


class KalmanTargetSource:
    """Virtual plume at a noisy external tracker's predicted target position.

    The tracker receives scripted target coordinates through a noisy measurement
    channel. Its state is never given to the odor controller. This is a separate
    source-generation condition, not an olfactory inference by MaleCNS.
    """

    def __init__(self, plume: WindFoodOdorPlume, *, seed: int,
                 measurement_sigma_m: float = 0.25, horizon_s: float = 0.4):
        self.plume = plume
        self.rng = np.random.default_rng(seed)
        self.sigma = measurement_sigma_m
        self.horizon_s = horizon_s
        self.state: np.ndarray | None = None
        self.covariance = np.diag([1.0, 1.0, 0.5, 0.5])
        self.last_time: float | None = None
        self.observation_index = 0

    def sample(self, position_xy: np.ndarray, yaw: float,
               target_xy: np.ndarray, time_s: float) -> dict[str, float]:
        target = np.asarray(target_xy, dtype=float)
        dt = 0.0 if self.last_time is None else time_s - self.last_time
        transition = np.eye(4)
        transition[0, 2] = transition[1, 3] = dt
        measurement = None
        if self.state is None:
            measurement = target + self.rng.normal(0, self.sigma, 2)
            self.state = np.r_[measurement, [0.0, 0.0]]
            measured = True
        else:
            self.state = transition @ self.state
            self.covariance = transition @ self.covariance @ transition.T
            self.covariance += np.diag([0.01, 0.01, 0.04, 0.04])
            # Every other decision is prediction-only; the pattern is fixed.
            measured = self.observation_index % 2 == 0
            if measured:
                measurement = target + self.rng.normal(0, self.sigma, 2)
                observation = np.column_stack((np.eye(2), np.zeros((2, 2))))
                residual = measurement - observation @ self.state
                innovation = observation @ self.covariance @ observation.T
                innovation += np.eye(2) * self.sigma ** 2
                gain = self.covariance @ observation.T @ np.linalg.inv(innovation)
                self.state += gain @ residual
                self.covariance = (np.eye(4) - gain @ observation) @ self.covariance
        source = self.state[:2] + self.horizon_s * self.state[2:]
        self.last_time = time_s
        self.observation_index += 1
        result = self.plume.sample(position_xy, yaw, source, time_s)
        result.update({"source_xy": source.tolist(),
                       "source_current_offset_m": float(np.linalg.norm(source - target)),
                       "tracker_measured": measured,
                       "tracker_measurement_xy": measurement.tolist()
                       if measured else None,
                       "tracker_state": self.state.tolist()})
        return result


TRAIN_CASES = (
    OdorCase("train_static_a", "static", 0.08, 1.2, 3101, (3.7, 0.65)),
    OdorCase("train_static_b", "static", -0.12, 2.7, 3102, (4.1, -0.7)),
    OdorCase("train_moving_a", "moving", 0.10, 1.6, 3103,
             initial_y=-0.3, path_phase=0.4, path_frequency=0.38),
)
EVAL_CASES = (
    OdorCase("static_a", "static", -0.20, 4.3, 4101, (3.6, 0.55)),
    OdorCase("static_b", "static", 0.22, 5.1, 4102, (4.2, -0.65)),
    OdorCase("static_c", "static", -0.31, 6.6, 4103, (3.9, -0.9)),
    OdorCase("static_d", "static", 0.28, 7.4, 4104, (4.4, 0.8)),
    OdorCase("moving_a", "moving", -0.18, 4.8, 4201,
             initial_y=0.35, path_phase=0.8, path_frequency=0.32),
    OdorCase("moving_b", "moving", 0.24, 5.7, 4202,
             initial_y=-0.4, path_phase=-0.5, path_frequency=0.42),
    OdorCase("moving_c", "moving", -0.27, 6.9, 4203,
             initial_y=0.0, path_phase=1.5, path_frequency=0.48),
    OdorCase("moving_d", "moving", 0.16, 8.2, 4204,
             initial_y=0.25, path_phase=-1.0, path_frequency=0.36),
)


def run_case(case: OdorCase, *, mode: str, cast_angle: float,
             readout: MaleCNSOdorReadout | None, tracker: bool = False) -> dict:
    wind = (-cos(case.wind_angle), -sin(case.wind_angle))
    plume = WindFoodOdorPlume(wind_xy=wind, phase_s=case.phase_s)
    source = (KalmanTargetSource(plume, seed=case.seed + 1000)
              if tracker else plume)
    if mode.startswith("neural") or mode == "instant_neural":
        readout.reset(case.seed)
    if mode.startswith("instant_"):
        controller = OdorWindController(
            mode="raw_odor" if mode == "instant_raw" else "neural_odor",
            readout=readout)
    else:
        controller = OdorHistoryController(mode=mode, readout=readout,
                                           cast_angle=cast_angle, seed=case.seed)
    run = fly_chase(controller, duration=12.0 if case.kind == "static" else 18.0,
                    odor_perception=source, speed_source="fixed",
                    fixed_speed_m_s=0.65, initial_xy=(0.0, case.initial_y),
                    target_pose=case.pose)
    distances = [frame["cow_distance_m"] for frame in run["trace"]]
    first_hit = next((frame["t"] for frame in run["trace"]
                      if frame["cow_distance_m"] < 0.75), None)
    result = {"status": run["status"], "closest_distance_m": min(distances),
              "tracked": run["tracked"],
              "follow_fraction_after_4s": run["follow_fraction_after_4s"],
              "reached_0p75_m": first_hit is not None and run["status"] == "completed",
              "first_hit_s": first_hit,
              "mean_decision_ms": run["decision_mean_ms"],
              "deadline_misses": run["decision_deadline_misses"]}
    if tracker:
        observations = run["control_observations"]
        result["tracker_mean_current_offset_m"] = float(np.mean([
            obs["odor_sensor"]["source_current_offset_m"] for obs in observations]))
        result["tracker_mean_future_error_m"] = float(np.mean([
            np.linalg.norm(np.asarray(obs["odor_sensor"]["source_xy"]) -
                           np.asarray(case.pose(obs["t"] + 0.4)[0]))
            for obs in observations]))
    return {"metrics": result, "run": run}


def run_experiment(output: Path) -> dict:
    from .cli import verify_data

    output.mkdir(parents=True, exist_ok=True)
    hashes = verify_data()
    readout = MaleCNSOdorReadout()
    calibration = readout.fit()
    tuning = {}
    training_runs = {}
    for cast_angle in (0.45, 0.8):
        records = {}
        for case in TRAIN_CASES:
            record = run_case(case, mode="raw_odor", cast_angle=cast_angle,
                              readout=None)
            records[case.name] = record["metrics"]
            training_runs[f"{case.name}_cast_{cast_angle}"] = record["run"]
        score = (sum(record["reached_0p75_m"] for name, record in records.items()
                     if name.startswith("train_static"))
                 + records["train_moving_a"]["follow_fraction_after_4s"])
        tuning[str(cast_angle)] = {"score": score, "cases": records}
    chosen = max((0.45, 0.8), key=lambda angle: (tuning[str(angle)]["score"], -angle))
    modes = ("wind_only", "raw_odor", "neural_odor", "neural_swapped",
             "instant_raw", "instant_neural")
    runs = {}
    evaluation = {}
    for case in EVAL_CASES:
        evaluation[case.name] = {"case": asdict(case), "modes": {}}
        for mode in modes:
            record = run_case(case, mode=mode, cast_angle=chosen,
                              readout=readout if "neural" in mode else None)
            evaluation[case.name]["modes"][mode] = record["metrics"]
            runs[f"{case.name}_{mode}"] = record["run"]
        print(f"History plume {case.name}: "
              f"{ {mode: evaluation[case.name]['modes'][mode]['tracked'] for mode in modes} }",
              flush=True)
    tracker_cases = (EVAL_CASES[4], EVAL_CASES[5])
    tracker_results = {}
    for case in tracker_cases:
        tracker_results[case.name] = {}
        for mode in modes:
            record = run_case(case, mode=mode, cast_angle=chosen,
                              readout=readout if "neural" in mode else None,
                              tracker=True)
            tracker_results[case.name][mode] = record["metrics"]
            runs[f"tracker_{case.name}_{mode}"] = record["run"]
    summary = {
        "static_reached": {mode: sum(evaluation[case.name]["modes"][mode]["reached_0p75_m"]
                                     for case in EVAL_CASES if case.kind == "static")
                           for mode in modes},
        "moving_tracked": {mode: sum(evaluation[case.name]["modes"][mode]["tracked"]
                                     for case in EVAL_CASES if case.kind == "moving")
                           for mode in modes},
        "moving_follow_fraction": {mode: [evaluation[case.name]["modes"][mode]
                                          ["follow_fraction_after_4s"]
                                          for case in EVAL_CASES if case.kind == "moving"]
                                   for mode in modes},
        "tracker_moving_tracked": {mode: sum(tracker_results[case.name][mode]["tracked"]
                                             for case in tracker_cases) for mode in modes}}
    payload = {"protocol": {"issue": 3, "data_sha256": hashes,
                            "controller": "upwind surge on rising bilateral concentration; "
                                          "crosswind cast after decline or odor loss",
                            "speed_m_s": 0.65, "decision_interval_s": 0.4,
                            "static_success": "completed and within 0.75 m at least once",
                            "moving_success": "completed and >=70% of t>=4 s in "
                                              "1.4-3.5 m behind target, lateral error <1 m",
                            "training_cases": [asdict(case) for case in TRAIN_CASES],
                            "evaluation_cases": [asdict(case) for case in EVAL_CASES],
                            "tracker": "constant-velocity Kalman filter, 0.25 m Gaussian "
                                       "measurement noise each second decision, "
                                       "0.4 s forward prediction"},
               "calibration": calibration, "tuning": tuning,
               "chosen_cast_angle": chosen, "summary": summary,
               "evaluation": evaluation, "tracker_results": tracker_results,
               "training_runs": training_runs, "runs": runs}
    (output / "results.json").write_text(json.dumps(payload, ensure_ascii=False,
                                                     indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    return payload
