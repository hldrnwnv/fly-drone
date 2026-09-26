"""Paired 2-D/1-D moving-bar probe on saved MaleCNS visual inputs.

Run after build_retina_map.py. No FPV flight or learned vision model is used.
Every trial resets the same frozen brain and noise seed before 10 blank ticks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

from fly_drone.retina import RetinaProjector
from fly_drone.retina_2d import RetinaProjector2D

N_FRAMES = 50
HEIGHT = WIDTH = 64
BACKGROUND = 30
BAR = 220
STIMULI = ("blank", "bright", "stationary", "left_to_right", "right_to_left")


def frame(center: float | None = None, *, bright: bool = False) -> np.ndarray:
    image = np.full((HEIGHT, WIDTH, 3), BAR if bright else BACKGROUND, dtype=np.uint8)
    if center is not None:
        x = round(center * (WIDTH - 1))
        image[HEIGHT // 4:3 * HEIGHT // 4, max(0, x - 3):min(WIDTH, x + 4)] = BAR
    return image


def stimuli() -> dict[str, np.ndarray]:
    positions = np.linspace(0.15, 0.85, N_FRAMES)
    return {
        "blank": np.repeat(frame()[None], N_FRAMES, axis=0),
        "bright": np.repeat(frame(bright=True)[None], N_FRAMES, axis=0),
        "stationary": np.repeat(frame(0.5)[None], N_FRAMES, axis=0),
        "left_to_right": np.stack([frame(float(x)) for x in positions]),
        "right_to_left": np.stack([frame(float(x)) for x in positions[::-1]]),
    }


def run(output: Path, map_path: Path, data: Path, first_seed: int, seeds: int,
        movies: dict[str, np.ndarray] | None = None) -> None:
    if seeds <= 0:
        raise ValueError("seeds must be positive")
    with np.load(map_path) as saved:
        xy = saved["xy"]
        side = saved["side"]
        ids = saved["ids"]
        azimuth = saved["legacy_azimuth"]
    brain = FlyBrain(seed=first_seed, device="cpu", sensory_input=False)
    with np.load(data / "brain.npz") as meta:
        body_ids = meta["ids"]
    if not np.array_equal(ids, body_ids[brain.visual]):
        raise ValueError("saved map and loaded brain neuron IDs differ")
    two_d = RetinaProjector2D(xy, side)
    one_d = RetinaProjector(azimuth)
    movies = stimuli() if movies is None else movies
    names_of_stimuli = tuple(movies)
    if any(images.shape != (N_FRAMES, HEIGHT, WIDTH, 3) or images.dtype != np.uint8
           for images in movies.values()):
        raise ValueError("each stimulus must be 50 RGB uint8 frames of size 64x64")
    input_arrays = {}
    background = np.full(len(brain.visual), BACKGROUND / 255, dtype=np.float32)
    for mode in ("2d", "1d"):
        for eye in "LR":
            eye_mask = side == eye
            for name, images in movies.items():
                drives = []
                for image in images:
                    drive = two_d.encode(image, eye=eye) if mode == "2d" else one_d.encode(image)
                    if mode == "1d":
                        drive = np.where(eye_mask, drive, background).astype(np.float32)
                    drives.append(drive)
                input_arrays[f"{mode}_{eye}_{name}"] = np.stack(drives)

    names = [f"{kind}_{eye}" for eye in "LR" for kind in
             ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d")]
    group = np.full(brain.n, -1, dtype=np.int16)
    for i, name in enumerate(names):
        kind, eye = name.split("_")
        group[brain.cells([kind], side=eye)] = i
    descending = brain.cells(["descending_neuron"])
    descending_mask = np.zeros(brain.n, dtype=bool)
    descending_mask[descending] = True
    dnas = {eye: brain.cells(["DNa02"], side=eye) for eye in "LR"}
    if not all(len(x) for x in dnas.values()):
        raise ValueError("DNa02 group is missing")
    selected = (group >= 0)
    selected[descending] = True
    selected_indices = np.flatnonzero(selected)
    tick_offsets = [0]
    spike_chunks = []
    trials = []
    for seed in range(first_seed, first_seed + seeds):
        for mode in ("2d", "1d"):
            for eye in "LR":
                for name in names_of_stimuli:
                    drives = input_arrays[f"{mode}_{eye}_{name}"]
                    brain.reset(seed)
                    for _ in range(10):
                        brain.step(eye_drive=background)
                    counts = np.zeros(len(names), dtype=np.int64)
                    dn_counts = {s: 0 for s in "LR"}
                    dna_counts = {s: 0 for s in "LR"}
                    first_tick = len(tick_offsets) - 1
                    for drive in drives:
                        fired = brain.step(eye_drive=drive)
                        g = group[fired]
                        counts += np.bincount(g[g >= 0], minlength=len(names))
                        for s in "LR":
                            dn_counts[s] += int(np.sum((brain.side[fired] == s) & descending_mask[fired]))
                            dna_counts[s] += int(np.isin(fired, dnas[s]).sum())
                        retained = fired[selected[fired]].astype(np.int32)
                        spike_chunks.append(retained)
                        tick_offsets.append(tick_offsets[-1] + len(retained))
                    trials.append({"seed": seed, "mode": mode, "eye": eye,
                                   "stimulus": name, "first_tick": first_tick,
                                   "ticks": N_FRAMES,
                                   "T4_T5_spikes": dict(zip(names, counts.tolist())),
                                   "descending_spikes": dn_counts,
                                   "DNa02_spikes": dna_counts})
        print(f"completed seed {seed}", flush=True)
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "stimulus_frames.npz", **movies)
    np.savez_compressed(output / "inputs.npz", **input_arrays)
    np.savez_compressed(output / "spikes.npz",
                        tick_offsets=np.asarray(tick_offsets, dtype=np.int64),
                        neuron_indices=np.concatenate(spike_chunks),
                        observed_indices=selected_indices,
                        observed_body_ids=body_ids[selected_indices])
    (output / "results.json").write_text(json.dumps({
        "protocol": "50 20 ms movie ticks after 10 blank ticks; same reset noise seed per condition",
        "first_seed": first_seed, "seeds": seeds,
        "controls": list(names_of_stimuli),
        "modes": {"2d": "source located columns; unlocated receptors held at background",
                  "1d": "original flybrain azimuth projector on identical RGB frames"},
        "eyes": "only named eye stimulated; other eye receives constant background",
        "spike_array": "tick_offsets delimit neuron_indices for each movie tick; see trial first_tick",
        "limits": "image-to-eye orientation and angle uncalibrated; toy LIF dynamics, no flight claim",
        "trials": trials}, indent=2), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/retina-2d"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--first-seed", type=int, default=19)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    run(args.output, args.map, args.data, args.first_seed, args.seeds)
