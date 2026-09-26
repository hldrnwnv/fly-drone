"""Targeted MaleCNS probes based on published fly flight-circuit roles.

Separately measure power, steering, and premotor outputs. Connectome weights
stay fixed. Stimulation is synthetic and does not reproduce natural wing or
haltere spike timing.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from flybrain import FlyBrain
import numpy as np


GROUPS = {
    "DNg02_population": tuple(f"DNg02_{letter}" for letter in "abcdefg"),
    "DNa04": ("DNa04",),
    "DNa05": ("DNa05",),
    "SApp08_haltere": ("SApp08",),
    "SNpp37_38_tegula": ("SNpp37", "SNpp38"),
    "SNpp06_26_wing": ("SNpp06", "SNpp26"),
    "DNa02_comparison": ("DNa02",),
}
POWER = ("DLMn a, b", "DLMn c-f", "DVMn 1a-c")
STEERING = ("b1 MN", "b2 MN", "b3 MN", "i1 MN", "i2 MN")
READOUT_TYPES = (*POWER, *STEERING, "w-cHIN")
SEEDS = tuple(range(1600, 1612))
DOSES = (0.5, 1.0, -0.5)
DUTIES = (1.0, 0.33)
SIDES = ("L", "R", "LR")


def summarize(counts: dict[str, int]) -> dict[str, int]:
    def group(names: tuple[str, ...], side: str) -> int:
        return sum(counts[f"{name}_{side}"] for name in names)

    power_l, power_r = (group(POWER, side) for side in "LR")
    i_l, i_r = (group(("i1 MN", "i2 MN"), side) for side in "LR")
    b_l, b_r = (group(("b1 MN", "b2 MN"), side) for side in "LR")
    b13_l, b13_r = (group(("b1 MN", "b3 MN"), side) for side in "LR")
    relay_l, relay_r = (group(("w-cHIN",), side) for side in "LR")
    return {"power_total": power_l + power_r,
            "power_left_minus_right": power_l - power_r,
            "i1_i2_right_minus_left": i_r - i_l,
            "b1_b2_left_minus_right": b_l - b_r,
            "b1_b3_left_minus_right": b13_l - b13_r,
            "w_cHIN_total": relay_l + relay_r,
            "w_cHIN_left_minus_right": relay_l - relay_r}


class CircuitProbe:
    def __init__(self):
        self.brain = FlyBrain(seed=0, device="cpu", sensory_input=False)
        self.readouts = {f"{name}_{side}": self.brain.cells([name], side=side)
                         for name in READOUT_TYPES for side in "LR"}
        self.readout_names = tuple(self.readouts)
        self.readout_lookup = np.full(self.brain.n, -1, dtype=np.int16)
        for index, cells in enumerate(self.readouts.values()):
            self.readout_lookup[cells] = index
        self.inputs = {name: {side: self.brain.cells(list(types), side=side)
                              for side in "LR"} for name, types in GROUPS.items()}
        self.input_masks = {}
        for name, sides in self.inputs.items():
            self.input_masks[name] = {}
            for side, cells in sides.items():
                mask = np.zeros(self.brain.n, dtype=bool)
                mask[cells] = True
                self.input_masks[name][side] = mask
        missing = [f"{name}_{side}" for name, sides in self.inputs.items()
                   for side, cells in sides.items() if not len(cells)]
        if missing:
            raise RuntimeError(f"Missing named input cells: {missing}")

    def run(self, seed: int, group: str | None, sides: str,
            dose: float, duty: float) -> dict:
        self.brain.reset(seed)
        for _ in range(20):
            self.brain.step()
        counts_array = np.zeros(len(self.readout_names), dtype=int)
        pulse_ticks = round(30 * duty)
        inject = ([(self.inputs[group][side], dose) for side in sides]
                  if group is not None else [])
        masks = [self.input_masks[group][side] for side in sides] if group else []
        stimulated = 0
        for tick in range(30):
            fired = self.brain.step(inject=inject if tick < pulse_ticks else [])
            labels = self.readout_lookup[fired]
            labels = labels[labels >= 0]
            counts_array += np.bincount(labels, minlength=len(counts_array))
            if tick < pulse_ticks:
                stimulated += sum(int(mask[fired].sum()) for mask in masks)
        counts = dict(zip(self.readout_names, map(int, counts_array)))
        return {"seed": seed, "motor_spikes": counts,
                "features": summarize(counts), "stimulated_spikes": stimulated}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/directed-flight-circuits/results.json"))
    args = parser.parse_args()
    probe = CircuitProbe()
    baseline = {seed: probe.run(seed, None, "", 0.0, 0.0) for seed in SEEDS}
    records = []
    configurations = [(group, sides, dose, duty)
                      for group in GROUPS for sides in SIDES
                      for dose in DOSES for duty in DUTIES]
    for i, (group, sides, dose, duty) in enumerate(configurations, 1):
        episodes = [probe.run(seed, group, sides, dose, duty) for seed in SEEDS]
        effects = [{feature: episode["features"][feature] -
                    baseline[episode["seed"]]["features"][feature]
                    for feature in episode["features"]} for episode in episodes]
        discovery = effects[:6]
        heldout = effects[6:]
        records.append({"group": group, "types": GROUPS[group],
                        "stimulated_sides": sides, "dose": dose, "duty": duty,
                        "discovery_mean_effect": {key: float(np.mean([e[key] for e in discovery]))
                                                  for key in effects[0]},
                        "heldout_mean_effect": {key: float(np.mean([e[key] for e in heldout]))
                                                for key in effects[0]},
                        "episodes": episodes, "paired_effects": effects})
        if i % 14 == 0:
            print(f"configurations {i}/{len(configurations)}", flush=True)
    targets = {
        "DNg02_population": "power_total",
        "DNa04": "b1_b3_left_minus_right",
        "DNa05": "b1_b3_left_minus_right",
        "SApp08_haltere": "w_cHIN_total",
        "SNpp37_38_tegula": "i1_i2_right_minus_left",
        "SNpp06_26_wing": "i1_i2_right_minus_left",
        "DNa02_comparison": "i1_i2_right_minus_left",
    }
    selected = {}
    for group, feature in targets.items():
        group_records = [r for r in records if r["group"] == group]
        # Power and relay expect excitation; steering could have either sign.
        sign_free = feature in ("i1_i2_right_minus_left", "b1_b3_left_minus_right")
        best = max(group_records, key=lambda r: abs(r["discovery_mean_effect"][feature])
                   if sign_free else r["discovery_mean_effect"][feature])
        selected[group] = {"readout": feature,
                           "discovery_effect": best["discovery_mean_effect"][feature],
                           "heldout_effect": best["heldout_mean_effect"][feature],
                           "configuration": {key: best[key] for key in
                                             ("stimulated_sides", "dose", "duty")},
                           "heldout_effects": [e[feature] for e in best["paired_effects"][6:]]}
    result = {"method": "literature-directed paired neural interventions",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; synapse weights frozen",
              "neural_timestep_s": probe.brain.dt,
              "warmup_ticks": 20, "measurement_ticks": 30,
              "stimulation": "constant signed extra voltage for first duty fraction of ticks",
              "biological_scope": "SNpp37/38 and SNpp06/26 are annotated wing proprioceptors; SApp08 is annotated haltere sensory; DNg02 controls flight power in live-fly studies",
              "source_urls": ["https://elifesciences.org/articles/107867/figures",
                              "https://www.virtualflybrain.org/term/sapp08_r-malecns54772-vfb_jrmc1b82/",
                              "https://pmc.ncbi.nlm.nih.gov/articles/PMC9206711/",
                              "https://elifesciences.org/articles/96084/figures"],
              "groups": GROUPS,
              "input_cell_counts": {name: {side: int(len(cells)) for side, cells in sides.items()}
                                    for name, sides in probe.inputs.items()},
              "configurations": len(configurations),
              "neural_runs": len(SEEDS) * (1 + len(configurations)),
              "seeds": {"discovery": SEEDS[:6], "heldout": SEEDS[6:]},
              "baseline": baseline, "targets": targets, "selected": selected,
              "records": records}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"neural_runs": result["neural_runs"],
                      "selected": selected}, indent=2), flush=True)
    print(f"Results: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
