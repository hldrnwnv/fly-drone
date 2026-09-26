"""Run the connectome controller in a 3D MuJoCo FPV quad simulator."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from .brain import FlySteering
from .benchmark import benchmark
from .chase import fly_chase, fly_wing_direct
from .pursuit_brain import FlyPursuit, teacher_commands
from .cli import geometric_controller, verify_data
from .fpv import FPVQuad, TARGETS, fly_fpv
from .wing_motors import WingMotorDrive


def simulate(args) -> None:
    fly = FlySteering(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    runs = {}
    flights = {}
    for target_index, (side, target) in enumerate(TARGETS.items()):
        print(f"FPV flight to {side} gate", flush=True)
        trials = []
        for trial in range(args.eval_seeds):
            fly.reset(args.seed + 100 + 100 * target_index + trial)
            trials.append(fly_fpv(fly.action, target))
        runs[f"brain_{side}"] = trials[0]
        for number, run in enumerate(trials[1:], 2):
            runs[f"brain_{side}_{number}"] = run
        runs[f"geometric_{side}"] = fly_fpv(geometric_controller, target)
        runs[f"straight_{side}"] = fly_fpv(lambda _bearing: 0.0, target)
        flights[side] = {"brain_successes": sum(x["reached"] for x in trials),
                         "brain_trials": len(trials),
                         "brain_final_distances": [x["final_distance"] for x in trials],
                         "brain_mean_controller_ms": sum(x["controller_mean_ms"] for x in trials) / len(trials),
                         "geometric_reached": runs[f"geometric_{side}"]["reached"],
                         "geometric_mean_controller_ms": runs[f"geometric_{side}"]["controller_mean_ms"],
                         "straight_reached": runs[f"straight_{side}"]["reached"]}
    metrics = {"simulator": "MuJoCo 3.14.0", "model": "assets/fpv_quad.xml",
               "physics_timestep_s": 0.005, "brain_interval_s": 0.4,
               "brain": "MaleCNS v1.0 via flybrain 0.1.0", "data_sha256": hashes,
               "seed": args.seed, "train": train, "validation": validation, "flights": flights}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps({"metrics": metrics, "runs": runs,
                   "samples": {"train": fly.training_records, "validation": fly.validation_records}},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {(args.output / 'results.json').resolve()}", flush=True)
    if not args.no_video:
        for side in TARGETS:
            output = args.output / f"fpv_{side}.mp4"
            render_video(runs[f"brain_{side}"], output)
            print(f"Video: {output.resolve()}", flush=True)


def chase(args) -> None:
    fly = FlySteering(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    runs = {}
    captured_frames = {} if args.depth_flow_video else None
    for trial in range(args.eval_seeds):
        fly.reset(args.seed + 100 + trial)
        name = f"connectome_{trial + 1}"
        frames = []
        runs[name] = fly_chase(
            fly.action, duration=args.duration,
            frame_sink=(lambda t, image: frames.append((t, image)))
            if captured_frames is not None else None)
        if captured_frames is not None:
            captured_frames[name] = frames
        print(f"Chase trial {trial + 1}: tracked={runs[f'connectome_{trial + 1}']['tracked']}",
              flush=True)
    runs["geometric"] = fly_chase(geometric_controller, duration=args.duration)
    demo_run = next((name for name, run in runs.items()
                     if name.startswith("connectome_") and run["tracked"]), "connectome_1")
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "model": "MuJoCo FPV quad with scripted visual cow target",
               "control_roles": "connectome chooses yaw; camera tag size controls speed; stabilizer controls four motors",
               "seed": args.seed, "duration_s": args.duration,
               "demo_run": demo_run,
               "data_sha256": hashes, "train": train, "validation": validation,
               "connectome_tracked": sum(run["tracked"] for key, run in runs.items()
                                          if key.startswith("connectome_")),
               "connectome_trials": args.eval_seeds,
               "connectome_follow_fractions": [run["follow_fraction_after_4s"]
                                                for key, run in runs.items()
                                                if key.startswith("connectome_")],
               "geometric_tracked": runs["geometric"]["tracked"],
               "geometric_follow_fraction": runs["geometric"]["follow_fraction_after_4s"]}
    args.output.mkdir(parents=True, exist_ok=True)
    sensory_maps = None
    if captured_frames is not None:
        from .perception import NeuralPerception
        frames = captured_frames[demo_run]
        observations = runs[demo_run]["control_observations"]
        if len(frames) != len(observations) or any(
                abs(t - obs["t"]) > 1e-5 or
                hashlib.sha256(image.tobytes()).hexdigest() != obs["frame_sha256"]
                for (t, image), obs in zip(frames, observations)):
            raise RuntimeError("captured FPV frames differ from control observations")
        perception = NeuralPerception()
        for index, (t, image) in enumerate(frames, 1):
            perception.observe(image, time_s=t)
            if index % 10 == 0:
                print(f"Depth and flow frames {index}/{len(frames)}", flush=True)
        sensory_maps = args.output / "sensory_maps.npz"
        perception.save(sensory_maps)
        metrics["depth_flow_display"] = {
            "mode": "diagnostic_only; does not affect steering or rotor commands",
            "frames": len(frames), "source_run": demo_run,
            "depth_model": perception.history[0][2].depth_model,
            "flow_model": perception.history[0][2].flow_model,
            "mean_inference_ms": float(np.mean([item[2].inference_ms
                                                for item in perception.history]))}
    (args.output / "results.json").write_text(json.dumps({"metrics": metrics, "runs": runs,
                        "samples": {"train": fly.training_records,
                                    "validation": fly.validation_records}},
                        ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if not args.no_video:
        for name, filename in ((demo_run, "follow_demo.mp4"),
                               ("geometric", "geometric.mp4")):
            output = args.output / filename
            render_video(runs[name], output)
            print(f"Video: {output.resolve()}", flush=True)
        if sensory_maps is not None:
            output = args.output / "follow_depth_flow.mp4"
            render_video(runs[demo_run], output, sensory_maps=sensory_maps,
                         diagnostic_only=True)
            print(f"Video: {output.resolve()}", flush=True)


def chase_vision(args) -> None:
    """Compare visual-cell stimulation with a locked bearing-only readout."""
    from .perception import NeuralPerception

    fly = FlySteering(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    perception = NeuralPerception()
    args.output.mkdir(parents=True, exist_ok=True)
    runs = {}
    open_loop = {}
    map_files = {}
    conditions = ("bearing", "flow", "flow_depth")
    for trial in range(args.eval_seeds):
        seed = args.seed + 100 + trial
        for mode in conditions:
            name = f"{mode}_{trial + 1}"
            fly.reset(seed)
            perception.reset()

            def visual_action(bearing, features, selected_mode=mode):
                command = fly.action(bearing, visual_features=features,
                                     visual_mode=("none" if selected_mode == "bearing"
                                                  else selected_mode))
                channels = ({"mode": "bearing", "flow_available": 0.0}
                            if selected_mode == "bearing" else fly.last_visual_input.copy())
                return command, channels

            runs[name] = fly_chase(visual_action, duration=args.duration,
                                  visual_perception=perception)
            observations = runs[name]["control_observations"]
            if len(observations) != len(perception.history) or any(
                    abs(t - obs["t"]) > 1e-5 or
                    hashlib.sha256(image.tobytes()).hexdigest() != obs["frame_sha256"]
                    for (t, image, _features), obs in zip(perception.history, observations)):
                raise RuntimeError(f"visual features do not match decision frames: {name}")
            maps = args.output / f"{name}_maps.npz"
            perception.save(maps)
            map_files[name] = maps.name
            if mode == "bearing":
                # Replay identical observations through each neural input mode.
                recorded = list(zip(observations,
                                    [item[2] for item in perception.history]))
                open_loop[str(trial + 1)] = {}
                for replay_mode in conditions:
                    fly.reset(seed)
                    decisions = []
                    for obs, features in recorded:
                        command = fly.action(obs["bearing"], visual_features=features,
                                             visual_mode=("none" if replay_mode == "bearing"
                                                          else replay_mode))
                        decisions.append({"t": obs["t"], "frame_sha256": obs["frame_sha256"],
                                          "bearing": obs["bearing"], "command": command,
                                          "visual_input": fly.last_visual_input.copy()})
                    open_loop[str(trial + 1)][replay_mode] = decisions
            print(f"Trial {trial + 1}, {mode}: tracked={runs[name]['tracked']}; "
                  f"follow={runs[name]['follow_fraction_after_4s']:.3f}", flush=True)
    runs["geometric"] = fly_chase(geometric_controller, duration=args.duration)
    demos = {mode: next((f"{mode}_{i + 1}" for i in range(args.eval_seeds)
                         if runs[f"{mode}_{i + 1}"]["tracked"]), f"{mode}_1")
             for mode in conditions}
    summaries = {mode: {
        "tracked": sum(runs[f"{mode}_{i + 1}"]["tracked"] for i in range(args.eval_seeds)),
        "follow_fractions": [runs[f"{mode}_{i + 1}"]["follow_fraction_after_4s"]
                             for i in range(args.eval_seeds)],
        "mean_decision_ms": float(np.mean([
            runs[f"{mode}_{i + 1}"]["decision_mean_ms"]
            for i in range(args.eval_seeds)])),
        "deadline_misses": sum(runs[f"{mode}_{i + 1}"]["decision_deadline_misses"]
                               for i in range(args.eval_seeds)),
    } for mode in conditions}
    comparisons = {}
    for mode in ("flow", "flow_depth"):
        changes = 0
        direction_matches = 0
        direction_total = 0
        for trial in range(args.eval_seeds):
            original = open_loop[str(trial + 1)]["bearing"]
            altered = open_loop[str(trial + 1)][mode]
            for baseline, candidate in zip(original, altered):
                changes += abs(candidate["command"] - baseline["command"]) > 1e-6
                if abs(candidate["bearing"]) >= 0.08:
                    direction_total += 1
                    direction_matches += np.sign(candidate["command"]) == np.sign(
                        candidate["bearing"])
        comparisons[mode] = {"changed_open_loop_commands": int(changes),
                             "open_loop_decisions": sum(len(open_loop[str(i + 1)]["bearing"])
                                                        for i in range(args.eval_seeds)),
                             "direction_matches": int(direction_matches),
                             "direction_decisions": direction_total}
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "design": "paired initial seeds, same camera/model budget, locked readout; "
                         "open-loop replay uses exactly the bearing-only FPV frames",
               "inputs": {"bearing": "tag bearing -> LC10a",
                          "flow": "bearing + RAFT flow -> T4/T5",
                          "flow_depth": "flow + learned depth-weighted looming -> LPLC2 and "
                                        "near/far motion contrast -> LC15"},
               "readout": "same fitted DNa02 linear readout in all conditions",
               "stabilizer": "same conventional four-motor controller",
               "seed": args.seed, "duration_s": args.duration,
               "data_sha256": hashes, "train": train, "validation": validation,
               "summaries": summaries, "open_loop_comparison": comparisons,
               "geometric_tracked": runs["geometric"]["tracked"],
               "geometric_follow_fraction": runs["geometric"]["follow_fraction_after_4s"],
               "demo_runs": demos, "map_files": map_files}
    output = args.output / "results.json"
    output.write_text(json.dumps({"metrics": metrics, "runs": runs,
                                  "open_loop": open_loop,
                                  "samples": {"train": fly.training_records,
                                              "validation": fly.validation_records}},
                                 ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    print(json.dumps({"summaries": summaries,
                      "open_loop_comparison": comparisons,
                      "geometric_tracked": metrics["geometric_tracked"]},
                     indent=2), flush=True)
    if not args.no_video:
        for mode, name in demos.items():
            video = args.output / f"{mode}_demo.mp4"
            render_video(runs[name], video,
                         sensory_maps=args.output / map_files[name],
                         diagnostic_only=mode == "bearing",
                         neural_input=mode != "bearing")
            print(f"Video: {video.resolve()}", flush=True)


def chase_retina(args) -> None:
    """Compare timed photoreceptor frames with external flow on matched flights."""
    from .perception import NeuralPerception

    fly = FlySteering(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    perception = NeuralPerception()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = ("bearing", "flow", "retina")
    runs, maps, frame_files, open_loop = {}, {}, {}, {}
    for trial in range(args.eval_seeds):
        seed = args.seed + 100 + trial
        for mode in conditions:
            name = f"{mode}_{trial + 1}"
            fly.reset(seed)
            perception.reset()
            fast_history = []
            baseline_sequences = []

            def visual_action(bearing, features, frames, selected_mode=mode):
                if selected_mode == "bearing":
                    baseline_sequences.append([frame.copy() for frame in frames])
                command = fly.action(bearing, visual_features=features,
                                     visual_mode=("none" if selected_mode == "bearing"
                                                  else selected_mode),
                                     retina_frames=frames if selected_mode == "retina" else None)
                channels = ({"mode": "bearing", "flow_available": 0.0}
                            if selected_mode == "bearing" else fly.last_visual_input.copy())
                return command, channels

            runs[name] = fly_chase(
                visual_action, duration=args.duration, visual_perception=perception,
                fast_frame_interval=0.04,
                fast_frame_sink=lambda t, image: fast_history.append((t, image)))
            observations = runs[name]["control_observations"]
            if len(observations) != len(perception.history) or any(
                    abs(t - obs["t"]) > 1e-5 or
                    hashlib.sha256(frame.tobytes()).hexdigest() != obs["frame_sha256"]
                    for (t, frame, _features), obs in zip(perception.history, observations)):
                raise RuntimeError(f"decision images differ from neural inputs: {name}")
            sampled = {round(t, 6): frame for t, frame in fast_history}
            if any(hashlib.sha256(sampled[round(obs["t"], 6)].tobytes()).hexdigest()
                   != obs["frame_sha256"] for obs in observations):
                raise RuntimeError(f"fast camera sequence differs from decisions: {name}")
            fast_path = args.output / f"{name}_frames.npz"
            np.savez_compressed(fast_path,
                                t_s=np.array([t for t, _frame in fast_history]),
                                rgb=np.stack([frame for _t, frame in fast_history]))
            frame_files[name] = fast_path.name
            map_path = args.output / f"{name}_maps.npz"
            perception.save(map_path)
            maps[name] = map_path.name
            if mode == "bearing":
                recorded = list(zip(observations, [item[2] for item in perception.history],
                                    baseline_sequences))
                open_loop[str(trial + 1)] = {}
                for replay_mode in (*conditions, "retina_static", "retina_reverse",
                                    "retina_only"):
                    fly.reset(seed)
                    decisions = []
                    for obs, features, frames in recorded:
                        if replay_mode == "retina_static":
                            eye_frames = [frames[-1]] * len(frames)
                        elif replay_mode == "retina_reverse":
                            eye_frames = list(reversed(frames))
                        else:
                            eye_frames = frames
                        effective_mode = ("retina_only" if replay_mode == "retina_only"
                                          else "retina" if replay_mode.startswith("retina")
                                          else "none" if replay_mode == "bearing"
                                          else replay_mode)
                        command = fly.action(obs["bearing"], visual_features=features,
                                             visual_mode=effective_mode,
                                             retina_frames=eye_frames if effective_mode.startswith("retina")
                                             else None)
                        decisions.append({"t": obs["t"],
                                          "frame_sha256": obs["frame_sha256"],
                                          "bearing": obs["bearing"],
                                          "command": command,
                                          "DNa02_activity": fly.trace.features().tolist(),
                                          "visual_input": fly.last_visual_input.copy()})
                    open_loop[str(trial + 1)][replay_mode] = decisions
            print(f"Trial {trial + 1}, {mode}: tracked={runs[name]['tracked']}; "
                  f"follow={runs[name]['follow_fraction_after_4s']:.3f}", flush=True)

    summaries = {mode: {
        "tracked": sum(runs[f"{mode}_{i + 1}"]["tracked"]
                       for i in range(args.eval_seeds)),
        "follow_fractions": [runs[f"{mode}_{i + 1}"]["follow_fraction_after_4s"]
                             for i in range(args.eval_seeds)],
        "mean_decision_ms": float(np.mean([
            runs[f"{mode}_{i + 1}"]["decision_mean_ms"]
            for i in range(args.eval_seeds)])),
        "deadline_misses": sum(runs[f"{mode}_{i + 1}"]["decision_deadline_misses"]
                               for i in range(args.eval_seeds))}
        for mode in conditions}
    comparisons = {}
    for mode in ("flow", "retina", "retina_static", "retina_reverse", "retina_only"):
        pairs = [(base, other)
                 for trial in range(args.eval_seeds)
                 for base, other in zip(open_loop[str(trial + 1)]["bearing"],
                                        open_loop[str(trial + 1)][mode])]
        comparisons[mode] = {
            "decisions": len(pairs),
            "changed_commands": sum(abs(a["command"] - b["command"]) > 1e-6
                                    for a, b in pairs),
            "mean_abs_command_delta": float(np.mean([
                abs(a["command"] - b["command"]) for a, b in pairs])),
            "T4_T5_spikes": int(sum(
                value for _a, b in pairs for key, value in b["visual_input"].items()
                if key.endswith("_spikes")))}
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "design": "same FPV camera every 40 ms, same neural ticks, same "
                         "decision-frame RAFT/depth compute, same readout and quad stabilizer",
               "retina_limit": "1D luminance projection from middle half of image; "
                               "azimuth estimates only, no elevation or measured photoreceptor tuning",
               "inputs": {"bearing": "tag angle -> LC10a",
                          "flow": "tag angle + RAFT -> T4/T5",
                          "retina": "tag angle + timed FPV luminance -> photoreceptors"},
               "open_loop_controls": {
                   "retina_static": "last frame repeated with the same frame count",
                   "retina_reverse": "same frames in reverse temporal order",
                   "retina_only": "LC10a input omitted; output gain still uses tag angle"},
               "readout": "same bearing-trained DNa02 linear readout in all conditions",
               "frame_interval_s": 0.04, "decision_interval_s": 0.4,
               "seed": args.seed, "duration_s": args.duration,
               "data_sha256": hashes, "train": train, "validation": validation,
               "summaries": summaries, "open_loop_comparison": comparisons,
               "map_files": maps, "frame_files": frame_files}
    (args.output / "results.json").write_text(json.dumps({"metrics": metrics,
        "runs": runs, "open_loop": open_loop,
        "samples": {"train": fly.training_records, "validation": fly.validation_records}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    print(json.dumps({"summaries": summaries, "open_loop_comparison": comparisons},
                     indent=2), flush=True)
    if not args.no_video:
        demos = {mode: next((f"{mode}_{i + 1}" for i in range(args.eval_seeds)
                             if runs[f"{mode}_{i + 1}"]["tracked"]), f"{mode}_1")
                 for mode in conditions}
        for mode, name in demos.items():
            path = args.output / f"{mode}_demo.mp4"
            render_video(runs[name], path,
                         sensory_maps=args.output / maps[name],
                         diagnostic_only=mode == "bearing",
                         neural_input=mode != "bearing",
                         retina_frames=(args.output / frame_files[name]
                                        if mode == "retina" else None),
                         retina_azimuth=(fly.brain.azimuth if mode == "retina" else None))
            print(f"Video: {path.resolve()}", flush=True)


def chase_odor(args) -> None:
    """Test a bilateral virtual food-odor stimulus against matched flights."""
    from .olfaction import FoodOdorPlume

    fly = FlySteering(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    plume = FoodOdorPlume()
    args.output.mkdir(parents=True, exist_ok=True)
    conditions = ("tag", "tag_odor", "odor_only")
    starts = (-0.45, 0.0, 0.45)
    runs, open_loop = {}, {}
    for trial in range(args.eval_seeds):
        seed = args.seed + 100 + trial
        initial_xy = (0.0, starts[trial % len(starts)])
        for mode in conditions:
            name = f"{mode}_{trial + 1}"
            fly.reset(seed)

            def odor_action(bearing, sample, selected_mode=mode):
                delivered = ({"L": 0.0, "R": 0.0}
                             if selected_mode == "tag" else sample)
                command = fly.action(
                    bearing, odor=delivered,
                    visual_mode="odor_only" if selected_mode == "odor_only" else "none")
                return command, {"mode": selected_mode, **fly.last_odor_input}

            runs[name] = fly_chase(
                odor_action, duration=args.duration, odor_perception=plume,
                speed_source="fixed" if mode == "odor_only" else None,
                initial_xy=initial_xy)
            if mode == "tag":
                observations = runs[name]["control_observations"]
                open_loop[str(trial + 1)] = {}
                for replay_mode in ("tag", "tag_odor", "odor_swapped",
                                    "odor_isotropic", "odor_only"):
                    fly.reset(seed)
                    decisions = []
                    for obs in observations:
                        sample = obs["odor_sensor"]
                        if replay_mode == "tag":
                            delivered = {"L": 0.0, "R": 0.0}
                        elif replay_mode == "odor_swapped":
                            delivered = {"L": sample["R"], "R": sample["L"]}
                        elif replay_mode == "odor_isotropic":
                            mean = (sample["L"] + sample["R"]) / 2
                            delivered = {"L": mean, "R": mean}
                        else:
                            delivered = sample
                        command = fly.action(
                            obs["bearing"], odor=delivered,
                            visual_mode="odor_only" if replay_mode == "odor_only"
                            else "none")
                        decisions.append({"t": obs["t"],
                                          "frame_sha256": obs["frame_sha256"],
                                          "bearing": obs["bearing"],
                                          "sample": sample,
                                          "command": command,
                                          "DNa02_activity": fly.trace.features().tolist(),
                                          "odor_input": fly.last_odor_input.copy()})
                    open_loop[str(trial + 1)][replay_mode] = decisions
                replay = open_loop[str(trial + 1)]["tag"]
                if any(abs(a["steering_command"] - b["command"]) > 1e-9 or
                       a["frame_sha256"] != b["frame_sha256"]
                       for a, b in zip(observations, replay)):
                    raise RuntimeError("odor-free replay did not reproduce flight decisions")
            print(f"Trial {trial + 1}, {mode}: status={runs[name]['status']}; "
                  f"follow={runs[name]['follow_fraction_after_4s']:.3f}", flush=True)

    summaries = {mode: {
        "tracked": sum(runs[f"{mode}_{i + 1}"]["tracked"]
                       for i in range(args.eval_seeds)),
        "follow_fractions": [runs[f"{mode}_{i + 1}"]["follow_fraction_after_4s"]
                             for i in range(args.eval_seeds)],
        "closest_cow_m": [min(frame["cow_distance_m"]
                              for frame in runs[f"{mode}_{i + 1}"]["trace"])
                           for i in range(args.eval_seeds)],
        "mean_decision_ms": float(np.mean([
            runs[f"{mode}_{i + 1}"]["decision_mean_ms"]
            for i in range(args.eval_seeds)]))}
        for mode in conditions}
    comparisons = {}
    for mode in ("tag_odor", "odor_swapped", "odor_isotropic", "odor_only"):
        pairs = [(base, other)
                 for trial in range(args.eval_seeds)
                 for base, other in zip(open_loop[str(trial + 1)]["tag"],
                                        open_loop[str(trial + 1)][mode])]
        comparisons[mode] = {
            "decisions": len(pairs),
            "changed_commands": int(sum(abs(a["command"] - b["command"]) > 1e-6
                                        for a, b in pairs)),
            "mean_abs_command_delta": float(np.mean([
                abs(a["command"] - b["command"]) for a, b in pairs])),
            "DM1_PN_spikes": int(sum(b["odor_input"]["DM1_PN_spikes"]
                                      for _a, b in pairs)),
            "VA2_PN_spikes": int(sum(b["odor_input"]["VA2_PN_spikes"]
                                      for _a, b in pairs))}
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "odor_identity": "virtual low-dose food-odor surrogate; not measured cow odor",
               "odor_source": "moving cow position, hidden from controller",
               "plume": "deterministic downwind Gaussian with meander and intermittency; "
                        "wind=(-1,0) in world XY",
               "sensor": "bilateral virtual antennae, +/-0.12 m in body lateral axis",
               "neural_input": "ORN_DM1 and ORN_VA2 on each side; 0.25*concentration "
                               "extra voltage per 20 ms tick; no connectome weight changes",
               "readout": "same bearing-trained DNa02 readout, not odor-trained",
               "conditions": {"tag": "tag bearing and image-size speed rule; odor withheld",
                              "tag_odor": "same visual inputs plus bilateral ORN stimulus",
                              "odor_only": "no tag bearing injected and fixed forward speed; "
                                           "conventional quad stabilizer remains"},
               "seed": args.seed, "starts_y_m": list(starts),
               "duration_s": args.duration, "data_sha256": hashes,
               "train": train, "validation": validation,
               "summaries": summaries, "open_loop_comparison": comparisons}
    (args.output / "results.json").write_text(json.dumps({
        "metrics": metrics, "runs": runs, "open_loop": open_loop,
        "samples": {"train": fly.training_records,
                    "validation": fly.validation_records}},
        ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    print(json.dumps({"summaries": summaries, "open_loop_comparison": comparisons},
                     indent=2), flush=True)
    if not args.no_video:
        for mode in conditions:
            demo = next((f"{mode}_{i + 1}" for i in range(args.eval_seeds)
                         if runs[f"{mode}_{i + 1}"]["tracked"]), f"{mode}_1")
            path = args.output / f"{mode}_demo.mp4"
            render_video(runs[demo], path)
            print(f"Video: {path.resolve()}", flush=True)


def chase_odor_navigation(args) -> None:
    """Test an odor-trained receptor readout with explicit upwind guidance."""
    from .odor_navigation import (MaleCNSOdorReadout, OdorWindController,
                                  WindFoodOdorPlume)

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = verify_data()
    readout = MaleCNSOdorReadout()
    calibration = readout.fit()
    plume = WindFoodOdorPlume()
    modes = ("wind_only", "raw_odor", "neural_zero", "neural_odor",
             "neural_swapped")
    static_cases = [(x, y) for x in (3.5, 4.5)
                    for y in (-0.8, 0.8, -1.2, 1.2)]
    runs = {}
    static_results = {}
    for case_index, source in enumerate(static_cases, 1):
        static_results[str(case_index)] = {"source_xy_m": list(source)}
        for mode in modes:
            if mode.startswith("neural"):
                readout.reset(1100 + case_index)
            controller = OdorWindController(
                mode=mode, readout=readout if mode.startswith("neural") else None)
            name = f"static_{case_index}_{mode}"
            run = fly_chase(controller, duration=12.0, odor_perception=plume,
                            speed_source="fixed", fixed_speed_m_s=0.65,
                            target_pose=lambda _t, xy=source: (xy, 0.0))
            distances = [frame["cow_distance_m"] for frame in run["trace"]]
            first_hit = next((frame["t"] for frame in run["trace"]
                              if frame["cow_distance_m"] < 0.75), None)
            static_results[str(case_index)][mode] = {
                "reached_0p75_m": first_hit is not None and run["status"] == "completed",
                "first_hit_s": first_hit,
                "closest_distance_m": min(distances),
                "status": run["status"]}
            runs[name] = run
        print(f"Static source {case_index}/{len(static_cases)}: "
              f"{static_results[str(case_index)]['neural_odor']}", flush=True)

    moving_results = {}
    for trial, offset in enumerate((-0.45, 0.0, 0.45), 1):
        moving_results[str(trial)] = {"initial_y_m": offset}
        for mode in modes:
            if mode.startswith("neural"):
                readout.reset(1200 + trial)
            controller = OdorWindController(
                mode=mode, readout=readout if mode.startswith("neural") else None)
            name = f"moving_{trial}_{mode}"
            run = fly_chase(controller, duration=18.0, odor_perception=plume,
                            speed_source="fixed", fixed_speed_m_s=0.65,
                            initial_xy=(0.0, offset))
            moving_results[str(trial)][mode] = {
                "tracked": run["tracked"],
                "follow_fraction_after_4s": run["follow_fraction_after_4s"],
                "closest_distance_m": min(frame["cow_distance_m"]
                                          for frame in run["trace"]),
                "status": run["status"]}
            runs[name] = run
        print(f"Moving source {trial}/3: "
              f"{moving_results[str(trial)]['neural_odor']}", flush=True)

    summary = {
        "static_reached": {mode: sum(static_results[str(i)][mode]["reached_0p75_m"]
                                     for i in range(1, len(static_cases) + 1))
                           for mode in modes},
        "moving_tracked": {mode: sum(moving_results[str(i)][mode]["tracked"]
                                     for i in range(1, 4)) for mode in modes},
        "moving_follow_fractions": {
            mode: [moving_results[str(i)][mode]["follow_fraction_after_4s"]
                   for i in range(1, 4)] for mode in modes}}
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "data_sha256": hashes,
               "design": "static discovery used x=4,y=+/-1 with raw sensor gain 2; "
                         "evaluation uses new static source locations and moving target",
               "readout": "trained only on synthetic L/R ORN input doses; "
                          "no source poses or flight outcomes in training",
               "neural_output": "bilateral ORN_DM1/VA2 spikes, not DNa02 motor pathway",
               "wind_sensor": "ideal upwind bearing relative to drone, no target bearing",
               "policy": "yaw=clip(0.6*upwind_bearing+2.0*odor_L_minus_R,-1,1); "
                         "fixed forward speed 0.65 m/s; ordinary quad stabilizer",
               "static_success": "completed flight comes within 0.75 m of stationary source",
               "static_duration_s": 12.0, "moving_duration_s": 18.0,
               "calibration": calibration,
               "summary": summary,
               "static_results": static_results,
               "moving_results": moving_results}
    (args.output / "results.json").write_text(
        json.dumps({"metrics": metrics, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    if not args.no_video:
        for name, filename in (("static_1_neural_odor", "static_neural_odor.mp4"),
                               ("static_1_wind_only", "static_wind_only.mp4"),
                               ("moving_1_neural_odor", "moving_neural_odor.mp4")):
            path = args.output / filename
            render_video(runs[name], path)
            print(f"Video: {path.resolve()}", flush=True)


def chase_odor_fusion(args) -> None:
    """Compare visual steering with wind/odor fallback during known tag gaps."""
    from .odor_navigation import (MaleCNSOdorReadout, VisualOdorFusion,
                                  WindFoodOdorPlume)

    args.output.mkdir(parents=True, exist_ok=True)
    hashes = verify_data()
    vision = FlySteering(seed=7)
    visual_training = vision.fit(96)
    odor = MaleCNSOdorReadout()
    odor_calibration = odor.fit()
    plume = WindFoodOdorPlume()
    modes = ("tag_only", "tag_wind_only", "tag_raw_odor",
             "tag_neural_odor", "tag_neural_swapped")
    dropout = lambda t: 4.0 <= t < 6.0 or 10.0 <= t < 12.0 or 15.0 <= t < 17.0
    runs = {}
    summaries = {mode: [] for mode in ("full_vision",) + modes}
    for trial, offset in enumerate((-0.45, 0.0, 0.45), 1):
        for mode in ("full_vision",) + modes:
            vision.reset(106 + trial)
            if mode.startswith("tag_neural"):
                odor.reset(1300 + trial)
            controller = VisualOdorFusion(
                mode="tag_only" if mode == "full_vision" else mode,
                visual_readout=vision,
                odor_readout=odor if mode.startswith("tag_neural") else None)
            name = f"{mode}_{trial}"
            run = fly_chase(
                controller, duration=18.0, odor_perception=plume,
                odor_visual_context=True, initial_xy=(0.0, offset),
                visual_dropout=None if mode == "full_vision" else dropout)
            runs[name] = run
            summaries[mode].append({
                "tracked": run["tracked"],
                "follow_fraction_after_4s": run["follow_fraction_after_4s"],
                "masked_decisions": sum(obs["reason"] == "scripted_visual_dropout"
                                        for obs in run["control_observations"]),
                "status": run["status"]})
        print(f"Fusion trial {trial}/3: "
              f"{ {mode: round(summaries[mode][-1]['follow_fraction_after_4s'], 3) for mode in summaries} }",
              flush=True)
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "data_sha256": hashes,
               "visual_readout": visual_training,
               "odor_calibration": odor_calibration,
               "design": "same moving cow, starts, visual readout, speed rule and tag masking; "
                         "two separate MaleCNS instances for visual and olfactory readouts",
               "visual_dropout_s": [[4, 6], [10, 12], [15, 17]],
               "fallback": "when tag visible use visual DNa02 readout; when masked use "
                           "wind plus raw or ORN-decoded bilateral odor; full_vision has no mask",
               "wind_sensor": "idealized upwind bearing, no target coordinates",
               "summaries": summaries,
               "tracked_counts": {mode: sum(x["tracked"] for x in summaries[mode])
                                  for mode in summaries}}
    (args.output / "results.json").write_text(
        json.dumps({"metrics": metrics, "runs": runs}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    print(json.dumps({"tracked_counts": metrics["tracked_counts"],
                      "follow_fractions": {mode: [x["follow_fraction_after_4s"]
                                                for x in summaries[mode]]
                                           for mode in summaries}}, indent=2), flush=True)
    if not args.no_video:
        for name, filename in (("tag_only_1", "masked_tag_only.mp4"),
                               ("tag_neural_odor_1", "masked_tag_neural_odor.mp4")):
            path = args.output / filename
            render_video(runs[name], path)
            print(f"Video: {path.resolve()}", flush=True)


def chase_brain(args) -> None:
    fly = FlyPursuit(seed=args.seed)
    hashes = verify_data()
    train = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    runs = {}
    for trial in range(args.eval_seeds):
        fly.reset(args.seed + 100 + trial)
        runs[f"brain_speed_{trial + 1}"] = fly_chase(fly.action, duration=args.duration,
                                                      joint_control=True)
        fly.reset(args.seed + 100 + trial)
        def rule_speed(bearing: float, range_m: float) -> tuple[float, float]:
            yaw, _speed = fly.action(bearing, range_m)
            return yaw, float(teacher_commands(bearing, range_m)[1])
        runs[f"rule_speed_{trial + 1}"] = fly_chase(rule_speed, duration=args.duration,
                                                    joint_control=True,
                                                    speed_source="image_range_rule")
        print(f"Trial {trial + 1}: brain speed {runs[f'brain_speed_{trial + 1}']['tracked']}, "
              f"rule speed {runs[f'rule_speed_{trial + 1}']['tracked']}", flush=True)
    runs["geometric"] = fly_chase(geometric_controller, duration=args.duration)
    demo_trial = next((trial + 1 for trial in range(args.eval_seeds)
                       if runs[f"brain_speed_{trial + 1}"]["tracked"]), 1)
    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0",
               "input": "FPV tag bearing to LC10a; apparent range to LC11",
               "outputs": "DNa02 linear readout for turn direction; 1314 descending-neuron linear readout for forward speed",
               "teacher": "supervised geometric yaw and range-based speed targets; connectome weights frozen",
               "seed": args.seed, "duration_s": args.duration, "demo_trial": demo_trial,
               "data_sha256": hashes, "train": train, "validation": validation,
               "brain_speed_tracked": sum(runs[f"brain_speed_{i+1}"]["tracked"]
                                          for i in range(args.eval_seeds)),
               "rule_speed_tracked": sum(runs[f"rule_speed_{i+1}"]["tracked"]
                                         for i in range(args.eval_seeds)),
               "trials": args.eval_seeds,
               "geometric_tracked": runs["geometric"]["tracked"]}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps({"metrics": metrics, "runs": runs,
                        "samples": {"train": fly.training_records,
                                    "validation": fly.validation_records}},
                        ensure_ascii=False, indent=2), encoding="utf-8")
    fly.yaw_readout.save(args.output / "yaw_readout.npz")
    fly.speed_readout.save(args.output / "speed_readout.npz")
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if not args.no_video:
        for name, filename in ((f"brain_speed_{demo_trial}", "brain_speed_demo.mp4"),
                               (f"rule_speed_{demo_trial}", "rule_speed_demo.mp4")):
            output = args.output / filename
            render_video(runs[name], output)
            print(f"Video: {output.resolve()}", flush=True)


def wing_direct(args) -> None:
    if args.neural_vision:
        from .perception import NeuralPerception
        perception = NeuralPerception()
    else:
        perception = None
    ticks_per_action = max(1, round(args.decision_interval / 0.020))
    fly = WingMotorDrive(seed=args.seed, ticks_per_action=ticks_per_action,
                         warmup_actions=max(1, round(240 / ticks_per_action)),
                         haltere_feedback=args.haltere_feedback,
                         retina_input=args.retina,
                         mapping="wing_wrench" if args.wing_wrench else "raw",
                         haltere_3d=args.gyro_3d,
                         neural_vision=args.neural_vision,
                         premotor_roll=args.premotor_roll,
                         sensory_roll=args.sensory_roll,
                         sensory_roll_types=(("SNpp37", "SNpp38", "SNpp06")
                                             if args.sensory_triplet else
                                             ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")),
                         premotor_gain=args.premotor_gain,
                         premotor_cap=args.premotor_cap,
                         premotor_reversed=args.premotor_reverse)
    hashes = verify_data()
    run = fly_wing_direct(fly, duration=args.duration,
                          decision_interval=args.decision_interval,
                          altitude_assist=args.altitude_assist,
                          perception=perception)
    result = {"metrics": {"brain": "MaleCNS v1.0 via flybrain 0.1.0",
                          "brain_dynamics": "flybrain, not Shiu or Eon",
                          "data_sha256": hashes,
                          "interface": ("DLM/DVM power; b1/b2/b3/i1/i2 steering MN to wing intent, "
                                        "body wrench, and rotor mixer" if args.wing_wrench else
                                        "DLM/DVM left/right power and b1/b2 steering MN spikes"),
                          "mapping": fly.mapping,
                          "calibration": ("resting power is hover ratio; neutral wrench is mass * gravity"
                                          if args.wing_wrench else
                                          "resting power spikes scaled to 0.75 kg hover thrust"),
                          "haltere_feedback": args.haltere_feedback,
                          "gyro_3d": args.gyro_3d,
                          "neural_vision": args.neural_vision,
                          "premotor_roll": args.premotor_roll,
                          "sensory_roll": args.sensory_roll,
                          "sensory_triplet": args.sensory_triplet,
                          "premotor_gain": args.premotor_gain,
                          "premotor_cap": args.premotor_cap,
                          "premotor_reverse": args.premotor_reverse,
                          "decision_interval_s": args.decision_interval,
                          "retina_input": args.retina,
                          "altitude_assist": args.altitude_assist,
                          "newtons_per_power_spike": fly.newtons_per_power_spike,
                          "seed": args.seed, "duration_s": args.duration,
                          "status": run["status"], "tracked": run["tracked"],
                          "spike_totals": {name: sum(obs["motor_spikes"][name]
                                                      for obs in run["control_observations"])
                                           for name in fly.groups}},
              "warmup_spikes": fly.warmup_record, "runs": {"wing_direct": run}}
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "results.json"
    path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if perception is not None:
        perception.save(args.output / "sensory_maps.npz")
    print(json.dumps(result["metrics"], ensure_ascii=False, indent=2), flush=True)
    print(f"Results: {path.resolve()}", flush=True)
    if not args.no_video:
        video = args.output / "wing_direct.mp4"
        render_video(run, video,
                     sensory_maps=(args.output / "sensory_maps.npz")
                     if perception is not None else None)
        print(f"Video: {video.resolve()}", flush=True)


def load_run(path: Path, name: str) -> dict:
    result = json.loads(path.read_text(encoding="utf-8"))
    try:
        return result["runs"][name]
    except KeyError as exc:
        raise SystemExit(f"Unknown run {name!r}; choose one of {list(result['runs'])}") from exc


def set_pose(quad: FPVQuad, frame: dict) -> None:
    if "cow_position" in frame:
        quad.set_cow_pose(frame["cow_position"][:2], frame["cow_yaw"])
    quad.data.qpos[:3] = frame["position"]
    quad.data.qpos[3:7] = frame["quaternion"]
    quad.data.qvel[:3] = frame["velocity"]
    quad.data.qvel[3:6] = 0
    mujoco.mj_forward(quad.model, quad.data)


def render_video(run: dict, path: Path, *, sensory_maps: Path | None = None,
                 diagnostic_only: bool = False, neural_input: bool = False,
                 retina_frames: Path | None = None,
                 retina_azimuth: np.ndarray | None = None) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for MP4 output; use --no-video or install ffmpeg")
    is_chase = run.get("kind") == "chase"
    quad = FPVQuad(show_cow=is_chase, show_gates=not is_chase)
    width, height = 640, 360
    if sensory_maps is not None:
        from .sensory_overlay import sensory_screen
        with np.load(sensory_maps) as saved:
            sensor_times = saved["t_s"].copy()
            depths = saved["relative_inverse_depth"].copy()
            flows = saved["flow_xy_px"].copy()
            flow_valid = saved["flow_valid"].copy()
        observations = run["control_observations"]
        if len(sensor_times) != len(observations) or any(
                abs(t - obs["t"]) > 1e-5 for t, obs in zip(sensor_times, observations)):
            raise ValueError("sensory maps do not match this flight's observations")
        video_height = 2 * height
    else:
        video_height = height
    odor_observations = (run.get("control_observations") or [])
    odor_overlay = bool(odor_observations and
                        "odor_sensor" in odor_observations[0])
    if odor_overlay:
        from .sensory_overlay import odor_screen
        odor_times = np.array([obs["t"] for obs in odor_observations])
    if retina_frames is not None:
        if sensory_maps is None or retina_azimuth is None:
            raise ValueError("retina video requires maps and receptor azimuths")
        from .retina import RetinaProjector
        with np.load(retina_frames) as saved:
            fast_times = saved["t_s"].copy()
            fast_rgb = saved["rgb"].copy()
        projector = RetinaProjector(retina_azimuth)
        receptor_bins = np.histogram(projector.azimuth, bins=160,
                                     range=(-1, 1))[0]
    process = subprocess.Popen([ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo",
                                "-pixel_format", "rgb24", "-video_size", f"{2 * width}x{video_height}",
                                "-framerate", "25", "-i", "-", "-c:v", "libx264",
                                "-pix_fmt", "yuv420p", str(path)], stdin=subprocess.PIPE)
    try:
        with mujoco.Renderer(quad.model, width=width, height=height) as renderer:
            for frame in run["trace"]:
                set_pose(quad, frame)
                renderer.update_scene(quad.data, camera="fpv")
                fpv = renderer.render().copy()
                renderer.update_scene(quad.data, camera="overview" if is_chase else "chase")
                chase = renderer.render().copy()
                if sensory_maps is None:
                    if odor_overlay:
                        odor_index = int(np.clip(np.searchsorted(
                            odor_times, frame["t"], side="right") - 1,
                            0, len(odor_times) - 1))
                        screen = odor_screen(fpv, chase,
                                            odor_observations[odor_index])
                    else:
                        screen = np.concatenate((fpv, chase), axis=1)
                else:
                    index = int(np.clip(np.searchsorted(sensor_times, frame["t"], side="right") - 1,
                                        0, len(sensor_times) - 1))
                    display_observation = {**observations[index],
                                           "cow_xy": frame.get("cow_position", [])[:2],
                                           "drone_xy": frame["position"][:2]}
                    profile = None
                    if retina_frames is not None:
                        fast_index = int(np.clip(np.searchsorted(
                            fast_times, frame["t"], side="right") - 1,
                            0, len(fast_times) - 1))
                        eye = projector.encode(fast_rgb[fast_index])
                        sums = np.histogram(projector.azimuth, bins=160,
                                            range=(-1, 1), weights=eye)[0]
                        profile = sums / np.maximum(receptor_bins, 1)
                    screen = sensory_screen(fpv, chase, depths[index], flows[index],
                                            bool(flow_valid[index]), display_observation,
                                            diagnostic_only=diagnostic_only,
                                            neural_input=neural_input,
                                            retina_profile=profile)
                process.stdin.write(screen.tobytes())
    finally:
        process.stdin.close()
        returncode = process.wait()
    if returncode:
        raise RuntimeError(f"ffmpeg exited with status {returncode}")


def replay_gui(run: dict) -> None:
    import mujoco.viewer

    is_chase = run.get("kind") == "chase"
    quad = FPVQuad(show_cow=is_chase, show_gates=not is_chase)
    frames = run["trace"]
    with mujoco.viewer.launch_passive(quad.model, quad.data) as viewer:
        start = time.monotonic()
        for frame in frames:
            if not viewer.is_running():
                break
            wait = frame["t"] - (time.monotonic() - start)
            if wait > 0:
                time.sleep(wait)
            set_pose(quad, frame)
            viewer.sync()
        while viewer.is_running():
            time.sleep(0.1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sim = sub.add_parser("simulate", help="train and run FPV physics episodes")
    sim.add_argument("--output", type=Path, default=Path("runs/fpv"))
    sim.add_argument("--train-samples", type=int, default=96)
    sim.add_argument("--validation-samples", type=int, default=32)
    sim.add_argument("--eval-seeds", type=int, default=5)
    sim.add_argument("--seed", type=int, default=7)
    sim.add_argument("--no-video", action="store_true")
    comparison = sub.add_parser("benchmark", help="compare camera-based gate flight controllers")
    comparison.add_argument("--output", type=Path, default=Path("runs/vision-benchmark"))
    comparison.add_argument("--train-samples", type=int, default=96)
    comparison.add_argument("--validation-samples", type=int, default=32)
    comparison.add_argument("--eval-seeds", type=int, default=5)
    comparison.add_argument("--seed", type=int, default=7)
    comparison.add_argument("--wiring-seed", type=int, default=2026)
    comparison.add_argument("--wiring-replicates", type=int, default=3)
    comparison.add_argument("--ticks-per-action", type=int, default=20)
    pursuit = sub.add_parser("chase", help="follow a moving visual target in 3D")
    pursuit.add_argument("--output", type=Path, default=Path("runs/chase"))
    pursuit.add_argument("--duration", type=float, default=18.0)
    pursuit.add_argument("--train-samples", type=int, default=96)
    pursuit.add_argument("--validation-samples", type=int, default=32)
    pursuit.add_argument("--eval-seeds", type=int, default=3)
    pursuit.add_argument("--seed", type=int, default=7)
    pursuit.add_argument("--no-video", action="store_true")
    pursuit.add_argument("--depth-flow-video", action="store_true",
                         help="show depth and optical flow from the same FPV frames; display only")
    vision_pursuit = sub.add_parser("chase-vision",
                                    help="compare tagged bearing with flow and distance-cell inputs")
    vision_pursuit.add_argument("--output", type=Path, default=Path("runs/chase-neural-vision"))
    vision_pursuit.add_argument("--duration", type=float, default=18.0)
    vision_pursuit.add_argument("--train-samples", type=int, default=96)
    vision_pursuit.add_argument("--validation-samples", type=int, default=32)
    vision_pursuit.add_argument("--eval-seeds", type=int, default=3)
    vision_pursuit.add_argument("--seed", type=int, default=7)
    vision_pursuit.add_argument("--no-video", action="store_true")
    retina_pursuit = sub.add_parser("chase-retina",
                                    help="compare timed raw photoreceptor frames with external flow")
    retina_pursuit.add_argument("--output", type=Path, default=Path("runs/chase-retina"))
    retina_pursuit.add_argument("--duration", type=float, default=18.0)
    retina_pursuit.add_argument("--train-samples", type=int, default=96)
    retina_pursuit.add_argument("--validation-samples", type=int, default=32)
    retina_pursuit.add_argument("--eval-seeds", type=int, default=3)
    retina_pursuit.add_argument("--seed", type=int, default=7)
    retina_pursuit.add_argument("--no-video", action="store_true")
    odor_pursuit = sub.add_parser("chase-odor",
                                   help="probe virtual food odor through bilateral ORNs")
    odor_pursuit.add_argument("--output", type=Path, default=Path("runs/chase-odor"))
    odor_pursuit.add_argument("--duration", type=float, default=18.0)
    odor_pursuit.add_argument("--train-samples", type=int, default=96)
    odor_pursuit.add_argument("--validation-samples", type=int, default=32)
    odor_pursuit.add_argument("--eval-seeds", type=int, default=3)
    odor_pursuit.add_argument("--seed", type=int, default=7)
    odor_pursuit.add_argument("--no-video", action="store_true")
    odor_navigation = sub.add_parser("chase-odor-nav",
                                      help="calibrate bilateral ORN readout and test wind-guided odor search")
    odor_navigation.add_argument("--output", type=Path,
                                  default=Path("runs/chase-odor-navigation"))
    odor_navigation.add_argument("--no-video", action="store_true")
    odor_fusion = sub.add_parser("chase-odor-fusion",
                                  help="test odor/wind fallback when the visual tag is masked")
    odor_fusion.add_argument("--output", type=Path,
                              default=Path("runs/chase-odor-fusion"))
    odor_fusion.add_argument("--no-video", action="store_true")
    odor_history = sub.add_parser("chase-odor-history",
                                  help="evaluate temporal plume navigation on held-out cases")
    odor_history.add_argument("--output", type=Path,
                              default=Path("runs/chase-odor-history"))
    neural_pursuit = sub.add_parser("chase-brain", help="let connectome readouts command yaw and speed")
    neural_pursuit.add_argument("--output", type=Path, default=Path("runs/chase-brain"))
    neural_pursuit.add_argument("--duration", type=float, default=18.0)
    neural_pursuit.add_argument("--train-samples", type=int, default=192)
    neural_pursuit.add_argument("--validation-samples", type=int, default=64)
    neural_pursuit.add_argument("--eval-seeds", type=int, default=5)
    neural_pursuit.add_argument("--seed", type=int, default=7)
    neural_pursuit.add_argument("--no-video", action="store_true")
    wings = sub.add_parser("wing-direct", help="raw wing motor spikes drive four rotors")
    wings.add_argument("--output", type=Path, default=Path("runs/wing-direct"))
    wings.add_argument("--duration", type=float, default=8.0)
    wings.add_argument("--decision-interval", type=float, default=0.4)
    wings.add_argument("--seed", type=int, default=7)
    wings.add_argument("--gyro-feedback", "--haltere-feedback",
                       dest="haltere_feedback", action="store_true",
                       help="inject simulated gyro roll rate into MaleCNS SApp08 cells; synthetic encoding")
    wings.add_argument("--gyro-3d", action="store_true",
                       help="encode all three gyro axes into separate SApp08 cell pools")
    wings.add_argument("--neural-vision", action="store_true",
                       help="run RAFT Small flow and Depth Anything V2 Small; inject into T4/T5/LPLC2")
    wings.add_argument("--premotor-roll", action="store_true",
                       help="inject corrective roll-rate stimulus into IN08B051_d")
    wings.add_argument("--sensory-roll", action="store_true",
                       help="inject corrective roll-rate stimulus into SNpp26/27/37/38/06")
    wings.add_argument("--sensory-triplet", action="store_true",
                       help="with --sensory-roll, stimulate only SNpp37/38/06")
    wings.add_argument("--premotor-gain", type=float, default=1.0)
    wings.add_argument("--premotor-cap", type=float, default=0.5)
    wings.add_argument("--premotor-reverse", "--roll-reverse", action="store_true",
                       help="negative control: swap stimulated roll-feedback sides")
    wings.add_argument("--retina", action="store_true",
                       help="project FPV image brightness onto MaleCNS photoreceptors instead of LC10a")
    wings.add_argument("--wing-wrench", action="store_true",
                       help="map relative wing signals to body wrench, then mix four rotors")
    wings.add_argument("--altitude-assist", action="store_true",
                       help="diagnostic external vertical feedback; neural differential unchanged")
    wings.add_argument("--no-video", action="store_true")
    for command in ("video", "replay"):
        p = sub.add_parser(command, help="render a saved episode as MP4 or in the native 3D viewer")
        p.add_argument("--results", type=Path, default=Path("runs/fpv/results.json"))
        p.add_argument("--run", default="brain_left")
        if command == "video":
            p.add_argument("--output", type=Path, default=Path("runs/fpv/fpv_left.mp4"))
            p.add_argument("--sensory-maps", type=Path,
                           help="saved sensory_maps.npz for a four-panel FPV video")
            p.add_argument("--diagnostic-only", action="store_true",
                           help="label depth and flow as display-only for a chase video")
    args = parser.parse_args()
    if args.command == "simulate":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1:
            parser.error("sample counts and eval-seeds must be positive")
        simulate(args)
    elif args.command == "benchmark":
        if min(args.train_samples, args.validation_samples, args.eval_seeds,
               args.ticks_per_action, args.wiring_replicates) < 1:
            parser.error("sample counts, eval-seeds, wiring-replicates and ticks-per-action must be positive")
        benchmark(output=args.output, seed=args.seed, wiring_seed=args.wiring_seed,
                  train_samples=args.train_samples,
                  validation_samples=args.validation_samples, eval_seeds=args.eval_seeds,
                  ticks_per_action=args.ticks_per_action,
                  wiring_replicates=args.wiring_replicates)
    elif args.command == "chase":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1 or args.duration <= 0:
            parser.error("sample counts, eval-seeds and duration must be positive")
        chase(args)
    elif args.command == "chase-vision":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1 or args.duration <= 0:
            parser.error("sample counts, eval-seeds and duration must be positive")
        chase_vision(args)
    elif args.command == "chase-retina":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1 or args.duration <= 0:
            parser.error("sample counts, eval-seeds and duration must be positive")
        chase_retina(args)
    elif args.command == "chase-odor":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1 or args.duration <= 0:
            parser.error("sample counts, eval-seeds and duration must be positive")
        chase_odor(args)
    elif args.command == "chase-odor-nav":
        chase_odor_navigation(args)
    elif args.command == "chase-odor-fusion":
        chase_odor_fusion(args)
    elif args.command == "chase-odor-history":
        from .odor_experiment import run_experiment
        run_experiment(args.output)
    elif args.command == "chase-brain":
        if min(args.train_samples, args.validation_samples, args.eval_seeds) < 1 or args.duration <= 0:
            parser.error("sample counts, eval-seeds and duration must be positive")
        chase_brain(args)
    elif args.command == "wing-direct":
        if min(args.duration, args.decision_interval, args.premotor_gain,
               args.premotor_cap) <= 0:
            parser.error("duration, decision interval, premotor gain and cap must be positive")
        if args.premotor_roll and args.sensory_roll:
            parser.error("choose --premotor-roll or --sensory-roll")
        if args.sensory_triplet and not args.sensory_roll:
            parser.error("--sensory-triplet requires --sensory-roll")
        wing_direct(args)
    elif args.command == "video":
        maps = args.sensory_maps or args.results.parent / "sensory_maps.npz"
        render_video(load_run(args.results, args.run), args.output,
                     sensory_maps=maps if maps.exists() else None,
                     diagnostic_only=args.diagnostic_only)
    else:
        replay_gui(load_run(args.results, args.run))


if __name__ == "__main__":
    main()
