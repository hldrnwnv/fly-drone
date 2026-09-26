"""Train a readout, fly a virtual waypoint task, and save a replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from flybrain.data import DATA, FILES, RELEASE_URL

from .brain import FlySteering
from .render import write_replay
from .sim import fly_path


def geometric_controller(bearing: float) -> float:
    return max(-1.0, min(1.0, bearing / 0.7))


def verify_data() -> dict[str, str]:
    digests = {}
    for name, expected in FILES.items():
        digest = hashlib.sha256()
        with (DATA / name).open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        actual = digest.hexdigest()
        if actual != expected:
            raise RuntimeError(f"{DATA / name} does not match flybrain's published checksum")
        digests[name] = actual
    return digests


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/demo"))
    parser.add_argument("--train-samples", type=int, default=96)
    parser.add_argument("--validation-samples", type=int, default=32)
    parser.add_argument("--ticks-per-action", type=int, default=20)
    parser.add_argument("--eval-seeds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if min(args.train_samples, args.validation_samples, args.ticks_per_action) < 2 or args.eval_seeds < 1:
        parser.error("sample counts and ticks per action must be at least 2; eval-seeds at least 1")

    print("Loading MaleCNS connectome…", flush=True)
    fly = FlySteering(seed=args.seed, ticks_per_action=args.ticks_per_action)
    data_hashes = verify_data()
    print(f"Loaded {fly.brain.n:,} neurons; LC10a L/R: {len(fly.left)}/{len(fly.right)}; "
          f"DNa02 steering: {len(fly.trace.idx)}", flush=True)
    print("Training the linear descending-neuron readout…", flush=True)
    training = fly.fit(args.train_samples)
    validation = fly.validate(args.validation_samples)
    print(f"Held-out direction accuracy: {validation['direction_accuracy']:.1%}", flush=True)

    runs = {}
    summary = {}
    for target_id, (name, target) in enumerate((("цель слева", (5.0, 4.0)),
                                                ("цель справа", (5.0, -4.0)))):
        print(f"Flying: {name}…", flush=True)
        trials = []
        for trial in range(args.eval_seeds):
            fly.reset(args.seed + 10 + 100 * target_id + trial)
            trials.append(fly_path(fly.action, target))
        model = trials[0]
        geometry = fly_path(geometric_controller, target)
        straight = fly_path(lambda _bearing: 0.0, target)
        for label, run in (("мозг — " + name, model),
                           ("геометрический — " + name, geometry),
                           ("прямо — " + name, straight)):
            runs[label] = run
        for trial, run in enumerate(trials[1:], start=2):
            runs[f"мозг — {name}, повтор {trial}"] = run
        summary[name] = {"brain": {"reached": model["reached"], "final_distance": model["final_distance"],
                                   "successes": sum(run["reached"] for run in trials),
                                   "trials": len(trials),
                                   "mean_final_distance": sum(run["final_distance"] for run in trials) / len(trials),
                                   "trial_final_distances": [run["final_distance"] for run in trials]},
                         "geometric": {"reached": geometry["reached"], "final_distance": geometry["final_distance"]},
                         "straight": {"reached": straight["reached"], "final_distance": straight["final_distance"]}}

    metrics = {"source": "MaleCNS v1.0 via flybrain 0.1.0", "seed": args.seed,
               "data_release": RELEASE_URL, "data_sha256": data_hashes,
               "train": training, "validation": validation, "flights": summary,
               "neuron_counts": {"all": fly.brain.n, "LC10a_left": len(fly.left),
                                 "LC10a_right": len(fly.right), "DNa02_steering": len(fly.trace.idx)}}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "results.json").write_text(json.dumps({"metrics": metrics, "runs": runs,
                                                     "samples": {"train": fly.training_records,
                                                                 "validation": fly.validation_records}},
                                                     ensure_ascii=False, indent=2), encoding="utf-8")
    fly.readout.save(args.output / "readout.npz")
    write_replay(args.output / "replay.html", runs, metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print(f"Replay: {(args.output / 'replay.html').resolve()}", flush=True)


if __name__ == "__main__":
    main()
