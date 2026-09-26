"""Issue #2: paired RGB-frame comparison with a frozen MaleCNS graph.

One coloured gate is visible at a time, so the target is specified by the
scene itself. The raw-frame controllers receive only RGB or a dropout marker.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from math import atan2, cos, sin
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from fly_drone.cli import geometric_controller, verify_data
from fly_drone.fpv import FPVQuad, TARGETS, fly_fpv
from fly_drone.raw_vision import RawFrameReadout
from fly_drone.sim import wrap_angle
from fly_drone.vision import GateVision

SCENARIOS = ("nominal", "dropout_30")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def make_corpus(count: int, seed: int) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    """Render independent static camera poses, with labels from simulator state."""
    rng = np.random.default_rng(seed)
    images, labels, records = [], [], []
    for i in range(count):
        side = ("left", "right")[i % 2]
        target = TARGETS[side]
        xy = rng.uniform(-0.5, 0.5, size=2)
        true_bearing = float(rng.uniform(-1.0, 1.0))
        heading = atan2(target[1] - xy[1], target[0] - xy[0])
        yaw = wrap_angle(heading - true_bearing)
        quad = FPVQuad(visible_gate=side)
        quad.data.qpos[:2] = xy
        quad.data.qpos[3:7] = [cos(yaw / 2), 0, 0, sin(yaw / 2)]
        mujoco.mj_forward(quad.model, quad.data)
        with GateVision(side, include_frame=True) as sensor:
            observed = sensor.observe(quad, target, 0)
        frame = observed.pop("frame")
        images.append(frame)
        labels.append(float(geometric_controller(true_bearing)))
        records.append({"index": i, "side": side, "xy": xy.tolist(), "yaw": yaw,
                        "true_bearing_rad": true_bearing,
                        "label": labels[-1], "frame_sha256": observed["frame_sha256"],
                        "detector_visible": observed["visible"],
                        "detector_bearing_rad": observed["bearing"]})
    return np.stack(images), np.asarray(labels, dtype=np.float32), records


def start_pose(seed: int, side: str, trial: int) -> tuple[tuple[float, float], float]:
    rng = np.random.default_rng(seed + 100 * (side == "right") + trial)
    xy = tuple(float(x) for x in rng.uniform(-0.3, 0.3, size=2))
    return xy, float(rng.uniform(-0.15, 0.15))


def episode_seed(seed: int, scenario: str, side: str, trial: int) -> int:
    return seed + 10_000 * SCENARIOS.index(scenario) + 100 * (side == "right") + trial


def replay(controller, frames: list[np.ndarray | None], observations: list[dict]) -> dict:
    decisions = []
    for frame, observation in zip(frames, observations, strict=True):
        start = perf_counter()
        command = float(controller(frame))
        duration_ms = 1000 * (perf_counter() - start)
        true = observation["true_bearing"]
        decisions.append({"frame_sha256": observation["frame_sha256"],
                          "reason": observation["reason"], "true_bearing_rad": true,
                          "command": command, "decision_ms": duration_ms,
                          "direction_correct": bool(np.sign(command) == np.sign(true))
                          if abs(true) >= 0.08 else None})
    active = [x for x in decisions if x["direction_correct"] is not None]
    return {"decisions": decisions, "direction_correct": sum(x["direction_correct"] for x in active),
            "direction_total": len(active), "deadline_misses": sum(x["decision_ms"] > 400 for x in decisions)}


def summarize(runs: dict[str, dict], open_loop: dict[str, dict], name: str) -> dict:
    selected = [(key, value) for key, value in runs.items() if key.startswith(name + "/")]
    decisions = [obs for _, run in selected for obs in run["control_observations"]]
    replays = [values[name] for values in open_loop.values()]
    latency = np.asarray([obs["decision_ms"] for obs in decisions])
    return {"flights": len(selected), "gate_passes": sum(run["reached"] for _, run in selected),
            "by_scenario": {scenario: {"flights": sum(f"/{scenario}/" in key for key, _ in selected),
                                       "gate_passes": sum(run["reached"] for key, run in selected
                                                          if f"/{scenario}/" in key)}
                            for scenario in SCENARIOS},
            "camera_observations": len(decisions),
            "dropouts": sum(obs["reason"] == "dropout" for obs in decisions),
            "detector_misses": sum(obs["reason"] in ("no_gate", "gate_too_close") for obs in decisions),
            "open_loop_direction_correct": sum(x["direction_correct"] for x in replays),
            "open_loop_direction_total": sum(x["direction_total"] for x in replays),
            "closed_loop_decision_mean_ms": float(latency.mean()),
            "closed_loop_decision_p95_ms": float(np.percentile(latency, 95)),
            "closed_loop_deadline_misses": sum(run["decision_deadline_misses"] for _, run in selected)}


def run(output: Path, *, map_path: Path, train_count: int = 96,
        validation_count: int = 32, eval_seeds: int = 5, ticks: int = 20,
        seed: int = 71, wiring_seed: int = 2026) -> dict:
    if min(train_count, validation_count, eval_seeds, ticks) < 1:
        raise ValueError("counts and ticks must be positive")
    output.mkdir(parents=True, exist_ok=True)
    hashes = verify_data()
    train_frames, train_labels, train_records = make_corpus(train_count, seed)
    validation_frames, validation_labels, validation_records = make_corpus(
        validation_count, seed + 100_000)
    if {item["frame_sha256"] for item in train_records} & {
            item["frame_sha256"] for item in validation_records}:
        raise AssertionError("train/validation frame overlap")
    np.savez_compressed(output / "corpus_frames.npz", train=train_frames,
                        validation=validation_frames)
    runs, open_loop, frame_archive, fits = {}, {}, {}, {}
    keys = [(scenario, side, trial) for scenario in SCENARIOS for side in TARGETS
            for trial in range(eval_seeds)]
    for scenario, side, trial in keys:
        key = f"{scenario}/{side}/{trial}"
        print(f"reference {key}", flush=True)
        pose, yaw = start_pose(seed + 200_000, side, trial)
        frames: list[np.ndarray | None] = []
        with GateVision(side, scenario=scenario,
                        seed=episode_seed(seed, scenario, side, trial),
                        include_frame=True) as sensor:
            runs[f"visual/{key}"] = fly_fpv(
                geometric_controller, TARGETS[side], sensor=sensor,
                initial_xy=pose, initial_yaw=yaw, visible_gate=side,
                frame_sink=frames.append)
        frame_archive[key] = frames
        observations = runs[f"visual/{key}"]["control_observations"]
        open_loop[key] = {"visual": {"decisions": [
            {"frame_sha256": obs["frame_sha256"], "reason": obs["reason"],
             "true_bearing_rad": obs["true_bearing"], "command": obs["command"],
             "decision_ms": obs["decision_ms"],
             "direction_correct": bool(np.sign(obs["command"]) == np.sign(obs["true_bearing"]))
             if abs(obs["true_bearing"]) >= 0.08 else None} for obs in observations]}}
        active = [x for x in open_loop[key]["visual"]["decisions"]
                  if x["direction_correct"] is not None]
        open_loop[key]["visual"].update(direction_correct=sum(x["direction_correct"] for x in active),
                                         direction_total=len(active),
                                         deadline_misses=sum(x["decision_ms"] > 400 for x in observations))
    # A padded RGB array plus validity mask preserves the exact open-loop input.
    archive_arrays = {}
    for key, frames in frame_archive.items():
        archive_arrays[key.replace("/", "_")] = np.stack([
            frame if frame is not None else np.zeros_like(train_frames[0]) for frame in frames])
        archive_arrays[key.replace("/", "_") + "_valid"] = np.asarray(
            [frame is not None for frame in frames], dtype=bool)
    np.savez_compressed(output / "open_loop_frames.npz", **archive_arrays)
    names = ("connectome", "degree_shuffled_0", "degree_shuffled_1", "degree_shuffled_2")
    for index, name in enumerate(names):
        print(f"loading {name}", flush=True)
        model = RawFrameReadout(map_path, network="real" if index == 0 else "degree_shuffled",
                                wiring_seed=wiring_seed + index - 1, ticks_per_frame=ticks,
                                seed=seed)
        train_x = model.collect(train_frames, first_seed=seed + 300_000)
        validation_x = model.collect(validation_frames, first_seed=seed + 400_000)
        fit = model.fit(train_x, train_labels, validation_x, validation_labels)
        model.save(output / f"{name}_readout.npz")
        fits[name] = {"network": "real" if index == 0 else "degree_shuffled",
                      "wiring_seed": None if index == 0 else wiring_seed + index - 1,
                      "adjacency_sha256": model.adjacency_sha256,
                      "group_names": model.group_names, "train_features": train_x.tolist(),
                      "validation_features": validation_x.tolist(), **fit}
        for scenario, side, trial in keys:
            key = f"{scenario}/{side}/{trial}"
            print(f"{name} {key}", flush=True)
            model.reset(seed + 500_000 + episode_seed(seed, scenario, side, trial))
            pose, yaw = start_pose(seed + 200_000, side, trial)
            with GateVision(side, scenario=scenario,
                            seed=episode_seed(seed, scenario, side, trial),
                            include_frame=True, detect=False) as sensor:
                runs[f"{name}/{key}"] = fly_fpv(
                    model.action, TARGETS[side], sensor=sensor,
                    initial_xy=pose, initial_yaw=yaw, visible_gate=side,
                    frame_input=True)
            model.reset(seed + 600_000 + episode_seed(seed, scenario, side, trial))
            open_loop[key][name] = replay(model.action, frame_archive[key],
                                          runs[f"visual/{key}"]["control_observations"])
        del model
        gc.collect()
    summary = {name: summarize(runs, open_loop, name) for name in ("visual",) + names}
    result = {"protocol": {"issue": 2, "source": "MaleCNS v1.0 via flybrain 0.1.0",
                           "data_sha256": hashes, "map_sha256": sha256(map_path),
                           "train_seed": seed, "validation_seed": seed + 100_000,
                           "eval_start_seed": seed + 200_000,
                           "wiring_seed": wiring_seed, "train_count": train_count,
                           "validation_count": validation_count, "eval_seeds_per_side": eval_seeds,
                           "ticks_per_frame": ticks, "decision_budget_ms": 400,
                           "scenarios": list(SCENARIOS), "camera": "320x180 RGB, one target gate visible",
                           "target": "gate centre, commanded by scene colour and position",
                           "neural_input": "2D source-column luminance, no detector angle",
                           "neural_sensor": "RGB render and dropout only; colour detector disabled",
                           "readout": "ridge on 16 T4/T5 and 2 DNa02 group counts",
                           "labels": "simulator bearing clipped to geometric yaw command, training only",
                           "baseline": "fixed colour segmentation to bearing and proportional yaw",
                           "open_loop": "exact reference-flight RGB frames and dropout markers",
                           "closed_loop": "matched starts/dropout seeds, controller-specific later frames",
                           "limitation": "retina-camera optical orientation uncalibrated; altitude and speed assisted; offline latency measured but not enforced"},
              "corpus": {"train": train_records, "validation": validation_records},
              "fits": fits, "summary": summary, "runs": runs, "open_loop": open_loop}
    (output / "results.json").write_text(json.dumps(result, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/raw-fpv"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--train-count", type=int, default=96)
    parser.add_argument("--validation-count", type=int, default=32)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--ticks", type=int, default=20)
    parser.add_argument("--seed", type=int, default=71)
    args = parser.parse_args()
    run(args.output, map_path=args.map, train_count=args.train_count,
        validation_count=args.validation_count, eval_seeds=args.eval_seeds,
        ticks=args.ticks, seed=args.seed)
