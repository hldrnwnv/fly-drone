"""Paired, open-loop probes of MaleCNS premotor candidates for roll control.

The connectome weights stay frozen. A candidate is useful only if stimulation
changes the wing MN spikes consistently on held-out seeds; structural edges
alone do not establish a functional reflex or flight behavior.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from flybrain import FlyBrain
from scipy import sparse


SENSORY_TYPES = ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")
VIRTUAL_GROUPS = {
    "SNpp_bundle": SENSORY_TYPES,
    "SNpp37_38": ("SNpp37", "SNpp38"),
    "SNpp26_27": ("SNpp26", "SNpp27"),
    "SNpp37_38_06": ("SNpp37", "SNpp38", "SNpp06"),
    **{f"SNpp_bundle_without_{name}": tuple(n for n in SENSORY_TYPES if n != name)
       for name in SENSORY_TYPES},
}
CANDIDATES = ("SApp08", "w-cHIN", "IN06A003", "IN08B003",
              "IN08B051_d", "IN11A001", "IN17A033", "IN19A026",
              "SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06",
              "DNp07", "DNpe021", "DNg08_a,DNg08_b", "DNg24", *VIRTUAL_GROUPS)
MOTORS = ("b1 MN", "b2 MN", "b3 MN", "i1 MN", "i2 MN", "iii3 MN")


def candidate_cells(brain: FlyBrain, name: str, side: str) -> np.ndarray:
    return brain.cells(list(VIRTUAL_GROUPS.get(name, (name,))), side=side)


def roll_proxy(counts: dict[str, int]) -> int:
    """Positive means the current wing adapter would request positive roll."""
    return (counts["i1 MN_R"] + counts["i2 MN_R"] -
            counts["i1 MN_L"] - counts["i2 MN_L"])


def run_condition(brain: FlyBrain, seed: int, stimulus: np.ndarray,
                  amount: float, warmup_ticks: int, ticks: int,
                  groups: dict[str, np.ndarray]) -> dict:
    brain.reset(seed)
    for _ in range(warmup_ticks):
        brain.step()
    counts = {key: 0 for key in groups}
    inject = [(stimulus, amount)] if len(stimulus) else []
    for _ in range(ticks):
        fired = brain.step(inject=inject)
        for name, cells in groups.items():
            counts[name] += int(np.isin(cells, fired).sum())
    return {"motor_spikes": {key: counts[key] for key in counts if key.startswith(MOTORS)},
            "stimulated_spikes": counts["stimulus"],
            "roll_proxy": roll_proxy(counts)}


def structural_receipt(brain: FlyBrain, candidates: tuple[str, ...]) -> dict:
    W = sparse.csc_matrix((brain.weights, brain.indices, brain.indptr),
                          shape=(brain.n, brain.n)).tocsr()
    sensor = {side: brain.cells(["SApp08"], side=side) for side in "LR"}
    receipt = {}
    for name in candidates:
        for side in "LR":
            cells = candidate_cells(brain, name, side)
            key = f"{name}_{side}"
            direct = {f"{motor}_{target_side}": float(W[brain.cells([motor], side=target_side)]
                                                      [:, cells].sum())
                      for motor in MOTORS for target_side in "LR"}
            sensor_paths = {}
            for sensor_side in "LR":
                source = sensor[sensor_side]
                sensor_paths[sensor_side] = {
                    "direct": float(W[cells][:, source].sum()),
                    "two_synapse_linearized": float((W[cells, :] @ W[:, source]).sum())}
            receipt[key] = {"cells": int(len(cells)),
                            "direct_signed_weight_to_motors": direct,
                            "SApp08_to_candidate": sensor_paths}
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--candidates", nargs="+", choices=CANDIDATES,
                        default=CANDIDATES)
    parser.add_argument("--warmup-ticks", type=int, default=20)
    parser.add_argument("--ticks", type=int, default=30)
    parser.add_argument("--amount", type=float, default=1.0)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/wing-premotor-probe/results.json"))
    args = parser.parse_args()
    if args.seeds < 2 or min(args.warmup_ticks, args.ticks) < 1 or args.amount <= 0:
        parser.error("seeds >= 2; warmup-ticks, ticks, and amount must be positive")
    brain = FlyBrain(seed=0, device="cpu", sensory_input=False)
    candidates = tuple(args.candidates)
    structural = structural_receipt(brain, candidates)
    motor_groups = {f"{name}_{side}": brain.cells([name], side=side)
                    for name in MOTORS for side in "LR"}
    records = []
    seeds = list(range(args.seed_start, args.seed_start + args.seeds))
    for ordinal, seed in enumerate(seeds, 1):
        rest = run_condition(brain, seed, np.empty(0, dtype=int), args.amount,
                             args.warmup_ticks, args.ticks,
                             {**motor_groups, "stimulus": np.empty(0, dtype=int)})
        records.append({"seed": seed, "candidate": "rest", "side": None, **rest})
        for name in candidates:
            for side in "LR":
                cells = candidate_cells(brain, name, side)
                response = run_condition(brain, seed, cells, args.amount,
                                         args.warmup_ticks, args.ticks,
                                         {**motor_groups, "stimulus": cells})
                records.append({"seed": seed, "candidate": name, "side": side,
                                **response})
        print(f"seed {ordinal}/{args.seeds}", flush=True)
    summary = {}
    discovery = seeds[:args.seeds // 2]
    heldout = seeds[args.seeds // 2:]
    by_key = {(r["seed"], r["candidate"], r["side"]): r for r in records}
    for name in candidates:
        effects = {}
        for side in "LR":
            effects[side] = {seed: (by_key[seed, name, side]["roll_proxy"] -
                                    by_key[seed, "rest", None]["roll_proxy"])
                             for seed in seeds}
        oriented = [effects["R"][seed] - effects["L"][seed]
                    for seed in discovery]
        orientation = 1 if np.mean(oriented) >= 0 else -1
        summary[name] = {
            "orientation_from_discovery": orientation,
            "mean_effect_L_discovery": float(np.mean([effects["L"][s] for s in discovery])),
            "mean_effect_R_discovery": float(np.mean([effects["R"][s] for s in discovery])),
            "mean_effect_L_heldout": float(np.mean([effects["L"][s] for s in heldout])),
            "mean_effect_R_heldout": float(np.mean([effects["R"][s] for s in heldout])),
            "heldout_opposite_sign_pairs": sum(
                orientation * effects["R"][s] > 0 and
                orientation * effects["L"][s] < 0 for s in heldout),
            "heldout_seeds": len(heldout),
            "effects": {side: list(effects[side].values()) for side in "LR"},
        }
    result = {"method": "paired reset, fixed-current neuron stimulation",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; connectome weights frozen",
              "brain_dynamics": "hand-calibrated flybrain, not Shiu or Eon",
              "seed_split": {"discovery": discovery, "heldout": heldout},
              "warmup_ticks": args.warmup_ticks, "stimulus_ticks": args.ticks,
              "candidates": candidates,
              "step_s": brain.dt, "amount_extra_voltage_per_tick": args.amount,
              "readout": "(i1+i2)_R - (i1+i2)_L spike count, proxy for current wing adapter roll sign",
              "limitation": "Open-loop motor spikes only; no body dynamics or validated fly reflex",
              "structural": structural, "summary": summary, "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({name: {key: value for key, value in item.items()
                             if key != "effects"} for name, item in summary.items()},
                     indent=2), flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
