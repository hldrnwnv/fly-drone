"""Pre-registered issue #8 retina map-by-speed probe on MaleCNS v1.0."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

from fly_drone.retina import RetinaProjector
from fly_drone.retina_2d import RetinaProjector2D

FRAME_COUNT = 50
HEIGHT = WIDTH = 64
BACKGROUND = np.float32(30 / 255)
CONTRASTS = (95, 40)
SPEEDS = (1, 2, 4)
MAPS = ("original", "direct_only", "shuffle_801", "shuffle_802")
READOUTS = ("L1", "Mi1", "Tm3", "T4a", "T4b", "T4c", "T4d",
            "T5a", "T5b", "T5c", "T5d", "DNa02")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def map_variants(xy: np.ndarray, side: np.ndarray,
                 provenance: np.ndarray) -> dict[str, np.ndarray]:
    if not set(provenance).issubset({"direct", "inferred_l1", "unmapped"}):
        raise ValueError("unexpected receptor provenance")
    maps = {"original": xy.copy()}
    direct = xy.copy()
    direct[provenance == "inferred_l1"] = np.nan
    maps["direct_only"] = direct
    for seed in (801, 802):
        shuffled = xy.copy()
        rng = np.random.default_rng(seed)
        for eye in "LR":
            indices = np.flatnonzero((provenance == "inferred_l1") & (side == eye))
            shuffled[indices] = xy[rng.permutation(indices)]
            if not np.array_equal(np.sort(shuffled[indices], axis=0),
                                  np.sort(xy[indices], axis=0)):
                raise AssertionError("permutation changed the coordinate multiset")
        maps[f"shuffle_{seed}"] = shuffled
    return maps


def grating(periods: int, phase: float, contrast: int) -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.float64) / WIDTH
    values = np.rint(125 + contrast * np.sin(2 * np.pi * (periods * x - phase)))
    image = np.full((HEIGHT, WIDTH, 3), 125, dtype=np.uint8)
    image[HEIGHT // 4:3 * HEIGHT // 4] = values.astype(np.uint8)[None, :, None]
    return image


def movie(periods: int, phase: float, contrast: int, speed: int,
          direction: int) -> np.ndarray:
    frames = np.stack([grating(periods, phase + (direction * speed * tick % FRAME_COUNT)
                               / FRAME_COUNT, contrast) for tick in range(FRAME_COUNT)])
    return frames


def build_movies() -> dict[str, np.ndarray]:
    movies = {}
    for block, periods, phase in (("main", 4, 0.0), ("holdout", 3, 0.25),
                                  ("phase_only", 4, 0.25), ("period_only", 3, 0.0)):
        movies[f"{block}_blank"] = np.full((FRAME_COUNT, HEIGHT, WIDTH, 3), 30,
                                             dtype=np.uint8)
        for contrast in CONTRASTS:
            movies[f"{block}_{contrast}_static"] = np.repeat(
                grating(periods, phase, contrast)[None], FRAME_COUNT, axis=0)
            for speed in SPEEDS:
                right = movie(periods, phase, contrast, speed, 1)
                left = movie(periods, phase, contrast, speed, -1)
                if not np.array_equal(right[0], left[0]):
                    raise AssertionError("direction pair has different first frames")
                if sorted(frame.tobytes() for frame in right) != sorted(
                        frame.tobytes() for frame in left):
                    raise AssertionError("direction pair has different frame multisets")
                for direction, frames in (("right", right), ("left", left)):
                    movies[f"{block}_{contrast}_{speed}_{direction}"] = frames
    return movies


def specifications(seed: int, *, extension: bool = False) -> list[dict]:
    specs = []
    if extension and seed < 35:
        raise ValueError("extension uses only held-out seeds 35..42")
    blocks = (("phase_only", "period_only") if extension else
              ("main", "holdout") if seed >= 35 else ("main",))
    for block in blocks:
        for mode in MAPS:
            for eye in "LR":
                for contrast in CONTRASTS:
                    for speed in SPEEDS:
                        for direction in ("right", "left"):
                            specs.append({"block": block, "map": mode, "eye": eye,
                                          "contrast": contrast, "speed": speed,
                                          "condition": direction,
                                          "movie": f"{block}_{contrast}_{speed}_{direction}"})
                    if seed >= 35 and not extension:
                        specs.append({"block": block, "map": mode, "eye": eye,
                                      "contrast": contrast, "speed": 0,
                                      "condition": "static",
                                      "movie": f"{block}_{contrast}_static"})
                if seed >= 35 and not extension:
                    specs.append({"block": block, "map": mode, "eye": eye,
                                  "contrast": 0, "speed": 0, "condition": "blank",
                                  "movie": f"{block}_blank"})
    if seed >= 35 and not extension:
        for eye in "LR":
            for speed in SPEEDS:
                for direction in ("right", "left"):
                    specs.append({"block": "main", "map": "1d", "eye": eye,
                                  "contrast": 95, "speed": speed,
                                  "condition": direction,
                                  "movie": f"main_95_{speed}_{direction}"})
    return specs


def prepare_inputs(movies: dict[str, np.ndarray], maps: dict[str, np.ndarray],
                   side: np.ndarray, azimuth: np.ndarray,
                   specs: list[dict]) -> dict[str, np.ndarray]:
    projectors = {name: RetinaProjector2D(xy, side) for name, xy in maps.items()}
    legacy = RetinaProjector(azimuth)
    inputs = {}
    for spec in specs:
        key = "_".join(str(spec[k]) for k in ("block", "map", "eye", "contrast",
                                                "speed", "condition"))
        if key in inputs:
            continue
        frames = movies[spec["movie"]]
        if spec["map"] == "1d":
            eye_mask = side == spec["eye"]
            drive = np.stack([np.where(eye_mask, legacy.encode(frame), BACKGROUND)
                              for frame in frames]).astype(np.float32)
        else:
            projector = projectors[spec["map"]]
            drive = np.stack([projector.encode(frame, eye=spec["eye"])
                              for frame in frames])
        if drive.shape != (FRAME_COUNT, len(side)) or not np.isfinite(drive).all():
            raise ValueError(f"invalid input {key}")
        inputs[key] = drive
        spec["input_key"] = key
    return inputs


def run(output: Path, map_path: Path, data: Path, seed: int,
        *, extension: bool = False) -> None:
    if seed < 31 or seed > 42:
        raise ValueError("protocol requires seeds 31..42")
    output.mkdir(parents=True, exist_ok=True)
    seed_file = output / f"seed_{seed}.json"
    spike_file = output / f"seed_{seed}_spikes.npz"
    if seed_file.exists() or spike_file.exists():
        raise FileExistsError(f"seed {seed} already has saved output")
    with np.load(map_path) as saved:
        ids, visual = saved["ids"], saved["visual"]
        xy, side = saved["xy"], saved["side"]
        provenance, azimuth = saved["provenance"], saved["legacy_azimuth"]
    brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
    with np.load(data / "brain.npz") as meta:
        body_ids = meta["ids"]
    if not np.array_equal(ids, body_ids[brain.visual]) or not np.array_equal(
            visual, brain.visual):
        raise ValueError("saved map does not match loaded MaleCNS graph")
    maps = map_variants(xy, side, provenance)
    movies = build_movies()
    specs = specifications(seed, extension=extension)
    inputs = prepare_inputs(movies, maps, side, azimuth, specs)
    if not (output / "maps.npz").exists():
        np.savez_compressed(output / "maps.npz", **maps, side=side, provenance=provenance,
                            ids=ids)
    if not (output / "frames.npz").exists():
        np.savez_compressed(output / "frames.npz", **movies)
    # Stimuli and drives are independent of the noise seed. One shared file per
    # study split avoids storing identical receptor arrays eight times.
    input_path = output / ("extension_inputs.npz" if extension else
                           "held_out_inputs.npz" if seed >= 35 else "discovery_inputs.npz")
    if not input_path.exists():
        np.savez_compressed(input_path, **inputs)
    else:
        with np.load(input_path) as saved:
            if set(saved.files) != set(inputs) or any(
                    not np.array_equal(saved[key], value) for key, value in inputs.items()):
                raise ValueError("existing shared inputs differ from newly generated inputs")
    names = [f"{kind}_{eye}" for eye in "LR" for kind in READOUTS]
    group = np.full(brain.n, -1, dtype=np.int16)
    for index, name in enumerate(names):
        kind, eye = name.rsplit("_", 1)
        group[brain.cells([kind], side=eye)] = index
    selected = group >= 0
    descending = brain.cells(["descending_neuron"])
    selected[descending] = True
    selected_indices = np.flatnonzero(selected)
    tick_offsets = [0]
    chunks = []
    trials = []
    background = np.full(len(brain.visual), BACKGROUND, dtype=np.float32)
    for index, spec in enumerate(specs):
        brain.reset(seed)
        for _ in range(10):
            brain.step(eye_drive=background)
        first_tick = len(tick_offsets) - 1
        counts = np.zeros(len(names), dtype=np.int64)
        for drive in inputs[spec["input_key"]]:
            fired = brain.step(eye_drive=drive)
            g = group[fired]
            counts += np.bincount(g[g >= 0], minlength=len(names))
            retained = fired[selected[fired]].astype(np.int32)
            chunks.append(retained)
            tick_offsets.append(tick_offsets[-1] + len(retained))
        trials.append({**spec, "seed": seed, "first_tick": first_tick,
                       "ticks": FRAME_COUNT,
                       "spikes": dict(zip(names, counts.tolist()))})
        if (index + 1) % 50 == 0:
            print(f"seed {seed}: {index + 1}/{len(specs)} trials", flush=True)
    np.savez_compressed(spike_file,
                        tick_offsets=np.asarray(tick_offsets, dtype=np.int64),
                        neuron_indices=np.concatenate(chunks),
                        observed_indices=selected_indices,
                        observed_body_ids=body_ids[selected_indices])
    seed_file.write_text(json.dumps({"seed": seed, "trials": trials}, indent=2),
                         encoding="utf-8")
    print(f"seed {seed}: saved {len(trials)} trials", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/retina-factorial"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--extension", action="store_true")
    args = parser.parse_args()
    run(args.output, args.map, args.data, args.seed, extension=args.extension)
