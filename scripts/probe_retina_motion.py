"""Probe whether timed photoreceptor movies change T4/T5 direction responses.

Run: FLY_DATA="$PWD/data/flybrain" .venv/bin/python scripts/probe_retina_motion.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain
from fly_drone.retina import RetinaProjector


def movie(position: float, *, width: int = 320, height: int = 180) -> np.ndarray:
    frame = np.full((height, width, 3), 30, dtype=np.uint8)
    center = int(round(position * (width - 1)))
    frame[:, max(0, center - 12):min(width, center + 13)] = 220
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/chase-retina/motion_probe.json"))
    parser.add_argument("--seeds", type=int, default=3)
    args = parser.parse_args()
    brain = FlyBrain(seed=7, device="cpu", sensory_input=False)
    projector = RetinaProjector(brain.azimuth)
    labels = np.full(brain.n, -1, dtype=np.int8)
    names = []
    for side in "LR":
        for prefix in ("T4", "T5"):
            for suffix in "abcd":
                names.append(f"{prefix}{suffix}_{side}")
                labels[brain.cells([prefix + suffix], side=side)] = len(names) - 1
    positions = np.linspace(0.1, 0.9, 50)
    sequences = {"left_to_right": positions,
                 "right_to_left": positions[::-1],
                 "stationary": np.full_like(positions, 0.5)}
    blank = projector.encode(np.full((180, 320, 3), 30, dtype=np.uint8))
    drives = {key: [projector.encode(movie(float(p))) for p in trajectory]
              for key, trajectory in sequences.items()}
    trials = []
    for seed in range(7, 7 + args.seeds):
        for condition, eye_drives in drives.items():
            brain.reset(seed)
            for _ in range(10):
                brain.step(eye_drive=blank)
            counts = np.zeros(len(names), dtype=np.int64)
            for eye in eye_drives:
                fired = brain.step(eye_drive=eye)
                observed = labels[fired]
                counts += np.bincount(observed[observed >= 0], minlength=len(names))
            trials.append({"seed": seed, "condition": condition,
                           "T4_T5_spikes": dict(zip(names, counts.tolist()))})
            print(f"seed={seed} {condition} spikes={counts.sum()}", flush=True)
    output = {"protocol": "same horizontal bright bar and 20 ms eye frames; "
                          "reverse order vs stationary; 10 blank warmup ticks",
              "limitations": "1D azimuth approximation and uncalibrated MaleCNS spiking dynamics",
              "neural_ticks_per_condition": 60,
              "photoreceptors": len(brain.visual), "trials": trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
