"""Verify saved retina trial counters against tick-level spikes and drives."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def audit(data: Path, run_dir: Path) -> tuple[int, int]:
    with np.load(data / "brain.npz") as brain:
        cell_type, side, superclass = (brain[k] for k in ("cell_type", "side", "superclass"))
    with np.load(run_dir / "spikes.npz") as saved:
        offsets = saved["tick_offsets"]
        fired = saved["neuron_indices"]
    with np.load(run_dir / "inputs.npz") as saved:
        for key in saved.files:
            drive = saved[key]
            if drive.shape != (50, 6006) or not np.isfinite(drive).all() or drive.min() < 0 or drive.max() > 1:
                raise ValueError(f"invalid drive: {key}")
    trials = json.loads((run_dir / "results.json").read_text(encoding="utf-8"))["trials"]
    if len(offsets) - 1 != sum(t["ticks"] for t in trials):
        raise ValueError("tick offsets do not match trials")
    for trial in trials:
        start = offsets[trial["first_tick"]]
        stop = offsets[trial["first_tick"] + trial["ticks"]]
        spikes = fired[start:stop]
        for label, expected in trial["T4_T5_spikes"].items():
            kind, eye = label.split("_")
            observed = int(np.sum((cell_type[spikes] == kind) & (side[spikes] == eye)))
            if observed != expected:
                raise ValueError(f"T4/T5 mismatch in {trial['seed']} {label}")
        for eye in "LR":
            dn = int(np.sum((superclass[spikes] == "descending_neuron") & (side[spikes] == eye)))
            dna = int(np.sum((cell_type[spikes] == "DNa02") & (side[spikes] == eye)))
            if dn != trial["descending_spikes"][eye] or dna != trial["DNa02_spikes"][eye]:
                raise ValueError(f"descending mismatch in {trial['seed']} {eye}")
    return len(trials), len(offsets) - 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--run-dir", type=Path, default=Path("runs/retina-2d"))
    args = parser.parse_args()
    trials, ticks = audit(args.data, args.run_dir)
    print(f"verified {trials} trials and {ticks} ticks in {args.run_dir}")
