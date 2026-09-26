"""Audit issue #8 tick receipts and summarize all pre-registered comparisons."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from probe_retina_factors import CONTRASTS, MAPS, READOUTS, SPEEDS, sha256


def audit(run_dir: Path, data: Path, map_path: Path) -> dict:
    names = [f"{kind}_{eye}" for eye in "LR" for kind in READOUTS]
    with np.load(data / "brain.npz") as meta:
        cell_type, side = meta["cell_type"], meta["side"]
    label_index = np.full(len(cell_type), -1, dtype=np.int16)
    for index, label in enumerate(names):
        kind, eye = label.rsplit("_", 1)
        label_index[(cell_type == kind) & (side == eye)] = index
    with np.load(run_dir / "maps.npz") as maps:
        original = maps["original"]
        provenance = maps["provenance"]
        receptor_side = maps["side"]
        for name in MAPS:
            xy = maps[name]
            direct = provenance == "direct"
            np.testing.assert_array_equal(xy[direct], original[direct])
            if name.startswith("shuffle"):
                for eye in "LR":
                    inferred = np.flatnonzero((provenance == "inferred_l1") &
                                          (receptor_side == eye))
                    np.testing.assert_array_equal(np.sort(xy[inferred], axis=0),
                                                  np.sort(original[inferred], axis=0))
    for source_dir, blocks in ((run_dir, ("main", "holdout")),
                               (run_dir / "extension", ("phase_only", "period_only"))):
        with np.load(source_dir / "maps.npz") as maps:
            np.testing.assert_array_equal(maps["original"], original)
        with np.load(source_dir / "frames.npz") as frames:
            for block in blocks:
                for contrast in CONTRASTS:
                    for speed in SPEEDS:
                        right = frames[f"{block}_{contrast}_{speed}_right"]
                        left = frames[f"{block}_{contrast}_{speed}_left"]
                        if not np.array_equal(right[0], left[0]):
                            raise ValueError("direction pair starts with different frames")
                        if sorted(x.tobytes() for x in right) != sorted(
                                x.tobytes() for x in left):
                            raise ValueError("direction pair has different frame multisets")
    trials = []
    input_stats = {}
    studies = ([(run_dir, seed) for seed in range(31, 43)] +
               [(run_dir / "extension", seed) for seed in range(35, 43)])
    for source_dir, seed in studies:
        seed_data = json.loads((source_dir / f"seed_{seed}.json").read_text())
        seed_trials = seed_data["trials"]
        with np.load(source_dir / f"seed_{seed}_spikes.npz") as spikes:
            offsets = spikes["tick_offsets"]
            fired = spikes["neuron_indices"]
        input_file = ("extension_inputs.npz" if source_dir != run_dir else
                      "held_out_inputs.npz" if seed >= 35 else "discovery_inputs.npz")
        with np.load(source_dir / input_file) as inputs:
            if len(offsets) - 1 != 50 * len(seed_trials) or offsets[-1] != len(fired):
                raise ValueError(f"invalid spike offsets for seed {seed}")
            for trial in seed_trials:
                if trial["seed"] != seed or trial["ticks"] != 50:
                    raise ValueError(f"invalid trial header for seed {seed}")
                start, stop = offsets[trial["first_tick"]], offsets[trial["first_tick"] + 50]
                observed = fired[start:stop]
                if np.any((observed < 0) | (observed >= len(label_index))):
                    raise ValueError("invalid neuron index")
                labels = label_index[observed]
                counts = np.bincount(labels[labels >= 0], minlength=len(names))
                for index, label in enumerate(names):
                    if int(counts[index]) != trial["spikes"][label]:
                        raise ValueError(f"spike receipt mismatch {seed} {label}")
                key = trial["input_key"]
                drive = inputs[key]
                if drive.shape != (50, len(receptor_side)) or not np.isfinite(drive).all():
                    raise ValueError(f"invalid input {seed} {key}")
                if np.any((drive < 0) | (drive > 1)):
                    raise ValueError(f"input outside [0,1]: {seed} {key}")
                if np.any(drive[:, receptor_side != trial["eye"]] != np.float32(30 / 255)):
                    raise ValueError(f"other eye stimulated: {seed} {key}")
                if key not in input_stats:
                    input_stats[key] = {"mean_drive": float(drive.mean()),
                                        "active_receptors": int(np.sum(
                                            np.any(drive != np.float32(30 / 255), axis=0)))}
                trials.append(trial)
            # Shuffling coordinates must preserve the exact input distribution
            # at every tick, not merely the average input.
            for trial in seed_trials:
                if trial["map"] != "original":
                    continue
                for variant in ("shuffle_801", "shuffle_802"):
                    variant_key = trial["input_key"].replace("_original_", f"_{variant}_")
                    if variant_key not in inputs:
                        continue
                    inferred = (provenance == "inferred_l1") & (receptor_side == trial["eye"])
                    a = np.sort(inputs[trial["input_key"]][:, inferred], axis=1)
                    b = np.sort(inputs[variant_key][:, inferred], axis=1)
                    if not np.array_equal(a, b):
                        raise ValueError(f"shuffle changed input distribution: {seed} {variant_key}")
    lookup = {(t["seed"], t["block"], t["map"], t["eye"], t["contrast"],
               t["speed"], t["condition"]): t for t in trials}
    if len(lookup) != len(trials):
        raise ValueError("duplicate trial key")
    comparisons = []
    for key, right in lookup.items():
        seed, block, mode, eye, contrast, speed, condition = key
        if condition != "right":
            continue
        left = lookup[(seed, block, mode, eye, contrast, speed, "left")]
        comparisons.append({"seed": seed, "split": "discovery" if seed <= 34 else "held_out",
                            "block": block, "map": mode, "eye": eye,
                            "contrast": contrast, "speed": speed,
                            "delta_spikes": {name: right["spikes"][name] - left["spikes"][name]
                                             for name in names}})
    groups = defaultdict(list)
    for row in comparisons:
        key = (row["split"], row["block"], row["map"], row["eye"],
               row["contrast"], row["speed"])
        groups[key].append(row)
    summary = []
    for key, rows in sorted(groups.items()):
        summary.append({"split": key[0], "block": key[1], "map": key[2],
                        "eye": key[3], "contrast": key[4], "speed": key[5],
                        "n_seeds": len(rows),
                        "mean_delta": {name: float(np.mean([
                            row["delta_spikes"][name] for row in rows])) for name in names},
                        "positive": {name: int(sum(
                            row["delta_spikes"][name] > 0 for row in rows)) for name in names}})
    raw = data / "raw" / "optic-columns.xlsx"
    hashes = {str(p): sha256(p) for p in (map_path, data / "brain.npz",
                                            data / "weights.npz", raw,
                                            Path("scripts/probe_retina_factors.py"),
                                            Path("runs/retina-factorial/protocol.md"))}
    result = {"dataset": "male-cns:v1.0; flybrain==0.1.0", "hashes_sha256": hashes,
              "trial_count": len(trials), "movie_ticks": len(trials) * 50,
              "seeds": {"discovery": list(range(31, 35)),
                        "held_out": list(range(35, 43))},
              "inputs": input_stats, "comparisons": comparisons, "summary": summary}
    (run_dir / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (run_dir / "group_summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream, lineterminator="\n")
        writer.writerow(["split", "block", "map", "eye", "contrast_rgb", "speed_cycles_per_s",
                         "group", "n_seeds", "mean_right_minus_left_spikes_per_50_ticks",
                         "positive_seeds"])
        for row in summary:
            for name in names:
                writer.writerow([row["split"], row["block"], row["map"], row["eye"],
                                 row["contrast"], row["speed"], name, row["n_seeds"],
                                 row["mean_delta"][name], row["positive"][name]])
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retina-factorial"))
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    args = parser.parse_args()
    result = audit(args.run_dir, args.data, args.map)
    print(f"verified {result['trial_count']} trials, {result['movie_ticks']} movie ticks")
