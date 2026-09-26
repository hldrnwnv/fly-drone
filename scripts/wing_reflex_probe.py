"""Check whether symmetric haltere stimuli evoke opposite wing MN asymmetry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain


CELL_TYPES = ("b1 MN", "b2 MN", "DLMn a, b", "DLMn c-f", "DVMn 1a-c")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=12)
    parser.add_argument("--ticks", type=int, default=100)
    parser.add_argument("--amount", type=float, default=0.9)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/wing-reflex-probe/results.json"))
    args = parser.parse_args()
    if args.seeds < 1 or args.ticks < 1 or args.amount <= 0:
        parser.error("seeds, ticks and amount must be positive")
    brain = FlyBrain(seed=0, device="cpu", sensory_input=False)
    sensory = {side: brain.cells(["SApp08"], side=side) for side in "LR"}
    groups = {f"{name}_{side}": brain.cells([name], side=side)
              for name in CELL_TYPES for side in "LR"}
    records = []
    for seed in range(args.seeds):
        for condition in ("rest", "L", "R"):
            brain.reset(seed)
            injection = [] if condition == "rest" else [(sensory[condition], args.amount)]
            counts = {name: 0 for name in groups}
            for _ in range(args.ticks):
                fired = brain.step(inject=injection)
                for name, cells in groups.items():
                    counts[name] += int(np.isin(cells, fired).sum())
            steering = {side: counts[f"b1 MN_{side}"] + counts[f"b2 MN_{side}"]
                        for side in "LR"}
            records.append({"seed": seed, "condition": condition,
                            "motor_spikes": counts, "steering_spikes": steering,
                            "steering_left_minus_right": steering["L"] - steering["R"]})
    paired = []
    for seed in range(args.seeds):
        by_condition = {r["condition"]: r for r in records if r["seed"] == seed}
        paired.append({"seed": seed,
                       "left_stimulus_effect": (by_condition["L"]["steering_left_minus_right"] -
                                                by_condition["rest"]["steering_left_minus_right"]),
                       "right_stimulus_effect": (by_condition["R"]["steering_left_minus_right"] -
                                                 by_condition["rest"]["steering_left_minus_right"])})
    result = {"input": "constant synthetic voltage to MaleCNS SApp08 on one side",
              "amount_per_20ms_tick": args.amount, "ticks": args.ticks,
              "seeds": args.seeds,
              "hypothesis": "left stimulus increases left-minus-right b1/b2 spikes; "
                            "right stimulus decreases them",
              "mean_left_effect": float(np.mean([x["left_stimulus_effect"] for x in paired])),
              "mean_right_effect": float(np.mean([x["right_stimulus_effect"] for x in paired])),
              "sign_consistent_seeds": sum(x["left_stimulus_effect"] > 0 and
                                           x["right_stimulus_effect"] < 0 for x in paired),
              "paired": paired, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({key: result[key] for key in
                      ("ticks", "seeds", "mean_left_effect", "mean_right_effect",
                       "sign_consistent_seeds")}, indent=2))


if __name__ == "__main__":
    main()
