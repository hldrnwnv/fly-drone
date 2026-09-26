"""Calibrate virtual food-odor doses against MaleCNS ORNs and projection cells.

Run: FLY_DATA="$PWD/data/flybrain" .venv/bin/python scripts/probe_odor_input.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/chase-odor/odor_calibration.json"))
    parser.add_argument("--seeds", type=int, default=8)
    args = parser.parse_args()
    brain = FlyBrain(seed=7, device="cpu", sensory_input=False)
    pools = {"ORN_DM1": brain.cells(["ORN_DM1"]),
             "ORN_VA2": brain.cells(["ORN_VA2"]),
             "DM1_lPN": brain.cells(["DM1_lPN"]),
             "VA2_adPN": brain.cells(["VA2_adPN"]),
             "DNa02": brain.cells(["DNa02"])}
    if any(not len(cells) for cells in pools.values()):
        raise RuntimeError("Required olfactory or descending cells absent")
    orn = np.concatenate((pools["ORN_DM1"], pools["ORN_VA2"]))
    trials = []
    for seed in range(107, 107 + args.seeds):
        for dose in (0.0, 0.03, 0.05, 0.08, 0.10, 0.15):
            brain.reset(seed)
            counts = {name: 0 for name in pools}
            for _ in range(20):
                fired = brain.step(inject=[(orn, dose)] if dose else [])
                for name, cells in pools.items():
                    counts[name] += int(np.isin(fired, cells).sum())
            trials.append({"seed": seed, "dose_per_tick": dose,
                           "spike_counts": counts})
        print(f"seed={seed} done", flush=True)
    result = {"protocol": "20 MaleCNS ticks, 20 ms each, bilateral ORN_DM1/VA2; "
                          "same noise seeds for every dose, sensory_input=False",
              "cell_counts": {name: len(cells) for name, cells in pools.items()},
              "trials": trials}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
