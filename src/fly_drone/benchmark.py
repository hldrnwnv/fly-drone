"""Matched camera and control-budget experiment for FPV gate flight."""

from __future__ import annotations

import gc
import hashlib
import json
from pathlib import Path

import numpy as np

from .brain import FlySteering
from .cli import geometric_controller, verify_data
from .fpv import TARGETS, fly_fpv
from .vision import GateVision


SCENARIOS = ("nominal", "dropout_30")
CONTROLLERS = ("geometric", "connectome")


def _episode_seed(seed: int, scenario: str, side: str, trial: int) -> int:
    return seed + 10_000 * SCENARIOS.index(scenario) + 100 * list(TARGETS).index(side) + trial


def _start_pose(seed: int, side: str, trial: int) -> tuple[tuple[float, float], float]:
    rng = np.random.default_rng(seed + 100 * list(TARGETS).index(side) + trial)
    xy = tuple(float(value) for value in rng.uniform(-0.15, 0.15, size=2))
    return xy, float(rng.uniform(-0.08, 0.08))


def _run(controller, scenario: str, side: str, sensor_seed: int,
         initial_xy: tuple[float, float], initial_yaw: float) -> dict:
    with GateVision(side, scenario=scenario, seed=sensor_seed) as sensor:
        return fly_fpv(controller, TARGETS[side], sensor=sensor,
                       initial_xy=initial_xy, initial_yaw=initial_yaw)


def _open_loop(controller, observations: list[dict]) -> dict:
    """Replay exactly the same measured-angle sequence from a geometric flight."""
    samples = []
    for observation in observations:
        predicted = float(controller(observation["bearing"]))
        true = observation["true_bearing"]
        samples.append({"frame_sha256": observation["frame_sha256"],
                        "bearing_rad": observation["bearing"], "true_bearing_rad": true,
                        "visible": observation["visible"], "prediction": predicted,
                        "direction_correct": bool(np.sign(predicted) == np.sign(true))
                        if abs(true) >= 0.08 else None})
    active = [x for x in samples if x["direction_correct"] is not None]
    return {"samples": samples, "direction_correct": sum(x["direction_correct"] for x in active),
            "direction_total": len(active),
            "direction_accuracy": sum(x["direction_correct"] for x in active) / len(active)
            if active else None}


def _aggregate(runs: dict[str, dict], controller_names: tuple[str, ...]) -> dict:
    selected = [(key, run) for key, run in runs.items()
                if key.split("/", 1)[0] in controller_names]
    chosen = [run for _, run in selected]
    decisions = [obs for run in chosen for obs in run["control_observations"]]
    visible = [obs for obs in decisions if obs["visible"]]
    latencies = np.array([obs["decision_ms"] for obs in decisions], dtype=float)
    return {"flights": len(chosen), "passed": sum(run["reached"] for run in chosen),
            "pass_rate": sum(run["reached"] for run in chosen) / len(chosen),
            "by_scenario": {
                scenario: {
                    "flights": sum(f"/{scenario}/" in key for key, _ in selected),
                    "passed": sum(run["reached"] for key, run in selected
                                  if f"/{scenario}/" in key),
                } for scenario in SCENARIOS},
            "mean_gate_lateral_error_m": float(np.mean([run["gate_lateral_error_m"] for run in chosen])),
            "camera_observations": len(decisions), "gate_detected": len(visible),
            "visible_bearing_mae_rad": float(np.mean([abs(obs["bearing"] - obs["true_bearing"])
                                                        for obs in visible])) if visible else None,
            "decision_mean_ms": float(np.mean(latencies)),
            "decision_p95_ms": float(np.percentile(latencies, 95)),
            "deadline_misses": sum(run["decision_deadline_misses"] for run in chosen)}


def benchmark(*, output: Path, seed: int = 7, wiring_seed: int = 2026,
              train_samples: int = 96, validation_samples: int = 32,
              eval_seeds: int = 5, ticks_per_action: int = 20,
              wiring_replicates: int = 3) -> dict:
    if min(train_samples, validation_samples, eval_seeds, ticks_per_action,
           wiring_replicates) < 1:
        raise ValueError("all sample, trial and tick counts must be positive")
    output.mkdir(parents=True, exist_ok=True)
    runs: dict[str, dict] = {}
    open_loop: dict[str, dict] = {}
    model_fits: dict[str, dict] = {}
    samples: dict[str, dict] = {}
    hashes = verify_data()
    trial_keys = [(scenario, side, trial) for scenario in SCENARIOS
                  for side in TARGETS for trial in range(eval_seeds)]

    # The geometric controller defines one immutable camera sequence per episode.
    # Every open-loop controller sees its exact angles and frame hashes.
    for scenario, side, trial in trial_keys:
        key = f"geometric/{scenario}/{side}/{trial}"
        print(f"Camera flight {key}", flush=True)
        initial_xy, initial_yaw = _start_pose(seed, side, trial)
        runs[key] = _run(geometric_controller, scenario, side,
                         _episode_seed(seed, scenario, side, trial), initial_xy, initial_yaw)
        open_loop[key] = {"geometric": _open_loop(geometric_controller,
                                                  runs[key]["control_observations"])}

    shuffled_names = tuple(f"degree_shuffled_{index}" for index in range(wiring_replicates))
    network_specs = [("connectome", "real", wiring_seed)] + [
        (name, "degree_shuffled", wiring_seed + index)
        for index, name in enumerate(shuffled_names)]
    for name, network, current_wiring_seed in network_specs:
        print(f"Loading {name} network", flush=True)
        fly = FlySteering(seed=seed, ticks_per_action=ticks_per_action, network=network,
                          wiring_seed=current_wiring_seed)
        adjacency_sha256 = hashlib.sha256(fly.brain.indices.tobytes()).hexdigest()
        fit = fly.fit(train_samples)
        validation = fly.validate(validation_samples)
        model_fits[name] = {"network": network, "wiring_seed": fly.wiring_seed,
                            "adjacency_sha256": adjacency_sha256,
                            "neuron_count": fly.brain.n, "edge_count": len(fly.brain.indices),
                            "LC10a_left": len(fly.left), "LC10a_right": len(fly.right),
                            "DNa02_readout_count": len(fly.trace.idx),
                            "train": fit, "validation": validation}
        samples[name] = {"train": fly.training_records,
                         "validation": fly.validation_records}
        fly.readout.save(output / f"{name}_readout.npz")
        for scenario, side, trial in trial_keys:
            prefix = f"{scenario}/{side}/{trial}"
            print(f"Flight {name}/{prefix}", flush=True)
            initial_xy, initial_yaw = _start_pose(seed, side, trial)
            fly.reset(seed + 1000 + _episode_seed(seed, scenario, side, trial))
            runs[f"{name}/{prefix}"] = _run(fly.action, scenario, side,
                                            _episode_seed(seed, scenario, side, trial),
                                            initial_xy, initial_yaw)
            # Reset separately: the open-loop comparison must start from one
            # common network state and never inherit the closed-loop flight.
            fly.reset(seed + 2000 + _episode_seed(seed, scenario, side, trial))
            open_loop[f"geometric/{prefix}"][name] = _open_loop(
                fly.action, runs[f"geometric/{prefix}"]["control_observations"])
        del fly
        gc.collect()

    groups = {name: (name,) for name in CONTROLLERS + shuffled_names}
    groups["degree_shuffled_pooled"] = shuffled_names
    summary = {name: _aggregate(runs, members) for name, members in groups.items()}
    for name, members in groups.items():
        evaluations = [item[member] for item in open_loop.values() for member in members]
        correct = sum(item["direction_correct"] for item in evaluations)
        total = sum(item["direction_total"] for item in evaluations)
        summary[name]["open_loop_direction_correct"] = correct
        summary[name]["open_loop_direction_total"] = total
        summary[name]["open_loop_direction_accuracy"] = correct / total if total else None
    result = {
        "protocol": {"simulator": "MuJoCo 3.14.0", "model": "assets/fpv_quad.xml",
                     "camera": "fpv 320x180 RGB, vertical FOV 95 degrees",
                     "sensor": "fixed green/orange colour segmentation to target bearing; last bearing held if unseen",
                     "scenarios": list(SCENARIOS), "dropout_probability": 0.3,
                     "physical_timestep_s": 0.005, "decision_interval_s": 0.4,
                     "decision_budget_ms": 400,
                     "note": "Both neural graphs run the same number of simulation ticks. Geometry uses fewer operations; all share a 400 ms decision deadline. Offline simulation does not enforce missed deadlines.",
                     "same_camera_frames": "open_loop only; closed-loop trajectories yield different images",
                     "starting_pose": "matched within side/trial; x and y jitter +/-0.15 m, yaw jitter +/-0.08 rad",
                     "target_layout": TARGETS,
                     "seed": seed, "wiring_seed": wiring_seed,
                     "wiring_replicates": wiring_replicates,
                     "train_samples": train_samples, "validation_samples": validation_samples,
                     "eval_seeds": eval_seeds, "ticks_per_action": ticks_per_action,
                     "source": "MaleCNS v1.0 via flybrain 0.1.0", "data_sha256": hashes},
        "model_fits": model_fits, "summary": summary,
        "runs": runs, "open_loop": open_loop, "samples": samples,
    }
    path = output / "results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {path.resolve()}", flush=True)
    return result
