"""Test whether IN08B051_d mediates the SNpp-bundle wing-MN response."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain


SENSORY_TYPES = ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")
MOTOR_TYPES = ("i1 MN", "i2 MN")
SEEDS = tuple(range(600, 612))


def trial(brain: FlyBrain, seed: int, sensory: np.ndarray,
          relay: np.ndarray, *, stimulate: bool, silence: bool) -> dict:
    brain.reset(seed)
    for _ in range(20):
        brain.step()
    groups = {f"{name}_{side}": brain.cells([name], side=side)
              for name in MOTOR_TYPES for side in "LR"}
    groups["relay_L"] = brain.cells(["IN08B051_d"], side="L")
    groups["relay_R"] = brain.cells(["IN08B051_d"], side="R")
    counts = {key: 0 for key in groups}
    inject = []
    if stimulate:
        inject.append((sensory, 0.5))
    if silence:
        inject.append((relay, -2.0))
    for _ in range(30):
        fired = brain.step(inject=inject)
        for key, cells in groups.items():
            counts[key] += int(np.isin(cells, fired).sum())
    proxy = counts["i1 MN_R"] + counts["i2 MN_R"] - counts["i1 MN_L"] - counts["i2 MN_L"]
    return {"seed": seed, "motor_spikes": {key: value for key, value in counts.items()
                                            if key.startswith(MOTOR_TYPES)},
            "relay_spikes": {key: value for key, value in counts.items()
                             if key.startswith("relay_")},
            "roll_proxy": proxy}


def main() -> None:
    brain = FlyBrain(seed=0, device="cpu", sensory_input=False)
    conditions = {}
    for side in "LR":
        sensory = brain.cells(list(SENSORY_TYPES), side=side)
        relay = brain.cells(["IN08B051_d"], side=side)
        conditions[side] = {
            name: [trial(brain, seed, sensory, relay,
                         stimulate=name in ("sensory", "sensory_plus_relay_silenced"),
                         silence=name in ("relay_silenced", "sensory_plus_relay_silenced"))
                   for seed in SEEDS]
            for name in ("rest", "sensory", "relay_silenced",
                         "sensory_plus_relay_silenced")}
    summary = {}
    for side, groups in conditions.items():
        normal = [a["roll_proxy"] - b["roll_proxy"]
                  for a, b in zip(groups["sensory"], groups["rest"])]
        blocked = [a["roll_proxy"] - b["roll_proxy"]
                   for a, b in zip(groups["sensory_plus_relay_silenced"],
                                   groups["relay_silenced"])]
        summary[side] = {"mean_sensory_effect": float(np.mean(normal)),
                         "mean_effect_with_relay_silenced": float(np.mean(blocked)),
                         "sensory_effects": normal,
                         "blocked_effects": blocked,
                         "mean_relay_spikes_sensory": float(np.mean(
                             [r["relay_spikes"][f"relay_{side}"]
                              for r in groups["sensory"]])),
                         "mean_relay_spikes_blocked": float(np.mean(
                             [r["relay_spikes"][f"relay_{side}"]
                              for r in groups["sensory_plus_relay_silenced"]]))}
    result = {"method": "paired stimulation and within-side hyperpolarization",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; weights frozen",
              "sensory_types": SENSORY_TYPES, "relay_type": "IN08B051_d",
              "sensory_extra_voltage_per_tick": 0.5,
              "relay_extra_voltage_per_tick_when_silenced": -2.0,
              "warmup_ticks": 20, "stimulus_ticks": 30, "seeds": SEEDS,
              "readout": "(i1+i2)_R - (i1+i2)_L spike count",
              "summary": summary, "conditions": conditions}
    path = Path("runs/wing-premotor-probe/relay-mediation.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {path.resolve()}", flush=True)


if __name__ == "__main__":
    main()
