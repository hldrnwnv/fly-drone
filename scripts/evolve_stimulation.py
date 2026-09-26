"""Resumable mixed differential evolution of MaleCNS roll stimulation.

Searches cell groups, excitation/inhibition, side rule, gain, and pulse duty.
The connectome and wing-to-rotor adapter are fixed. Neural probes select
candidates; separate MuJoCo scenarios test whether their effects transfer.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
from time import perf_counter

from flybrain import FlyBrain
import numpy as np

from evaluate_premotor_roll import episode
from fly_drone.chase import fly_wing_direct
from fly_drone.wing_motors import WingMotorDrive
from report_stimulation import render_report


GROUPS = {
    "IN08B051_d": ("IN08B051_d",),
    **{name: (name,) for name in ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")},
    "SNpp37_38": ("SNpp37", "SNpp38"),
    "SNpp37_38_06": ("SNpp37", "SNpp38", "SNpp06"),
    "SNpp_bundle": ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06"),
    **{name: (name,) for name in ("SApp08", "w-cHIN", "DNa04", "DNa05",
                                   "DNb01", "DNp04", "DNa02", "DNp20", "DNp07",
                                   "DNpe021", "DNg08_a,DNg08_b", "DNg24")},
    "DNg02_all": tuple(f"DNg02_{letter}" for letter in "abcdefg"),
}
GROUP_NAMES = tuple(GROUPS)
BOUNDS = np.array([[0, len(GROUP_NAMES) - 1], [0, 1], [0, 1],
                   [0.2, 1.3], [0.2, 1.0]], dtype=float)
MOTOR_NAMES = tuple(f"{name}_{side}" for name in ("i1 MN", "i2 MN") for side in "LR")


def decode(vector: np.ndarray) -> dict:
    group = GROUP_NAMES[int(np.clip(round(vector[0]), 0, len(GROUP_NAMES) - 1))]
    return {"group": group, "types": GROUPS[group],
            "polarity": 1 if vector[1] >= 0.5 else -1,
            "orientation": 1 if vector[2] >= 0.5 else -1,
            "gain": float(vector[3]), "duty": float(vector[4])}


class NeuralProbe:
    def __init__(self, seeds: tuple[int, ...], heldout: tuple[int, ...]):
        self.brain = FlyBrain(seed=0, device="cpu", sensory_input=False)
        self.motors = {name: self.brain.cells([name.rsplit("_", 1)[0]],
                                             side=name[-1]) for name in MOTOR_NAMES}
        self.cells = {group: {side: self.brain.cells(list(types), side=side)
                              for side in "LR"} for group, types in GROUPS.items()}
        missing = [f"{group}_{side}" for group, sides in self.cells.items()
                   for side, cells in sides.items() if not len(cells)]
        if missing:
            raise RuntimeError(f"Candidate cells missing: {missing}")
        self.rest = {seed: self.run(seed, None, None, 0, 0)
                     for seed in (*seeds, *heldout)}

    def run(self, seed: int, group: str | None, side: str | None,
            amount: float, duty: float) -> dict:
        self.brain.reset(seed)
        for _ in range(20):
            self.brain.step()
        counts = {name: 0 for name in MOTOR_NAMES}
        cells = self.cells[group][side] if group else np.empty(0, dtype=int)
        pulse_ticks = max(1, round(30 * duty)) if group else 0
        stimulated_spikes = 0
        for tick in range(30):
            fired = self.brain.step(inject=[(cells, amount)] if tick < pulse_ticks else [])
            for name, motor_cells in self.motors.items():
                counts[name] += int(np.isin(motor_cells, fired).sum())
            if tick < pulse_ticks:
                stimulated_spikes += int(np.isin(cells, fired).sum())
        proxy = counts["i1 MN_R"] + counts["i2 MN_R"] - counts["i1 MN_L"] - counts["i2 MN_L"]
        return {"roll_proxy": proxy, "motor_spikes": counts,
                "stimulated_spikes": stimulated_spikes}

    def evaluate(self, decoded: dict, seeds: tuple[int, ...]) -> dict:
        amount = decoded["polarity"] * min(0.6 * decoded["gain"], 0.8)
        records = []
        for seed in seeds:
            rest = self.rest[seed]
            sides = {side: self.run(seed, decoded["group"], side, amount, decoded["duty"])
                     for side in "LR"}
            effect = {side: sides[side]["roll_proxy"] - rest["roll_proxy"]
                      for side in "LR"}
            left = -decoded["orientation"] * effect["L"]
            right = decoded["orientation"] * effect["R"]
            records.append({"seed": seed, "rest": rest, "stimulated": sides,
                            "effect": effect, "corrected_left": left,
                            "corrected_right": right, "pair_margin": min(left, right)})
        margins = np.asarray([r["pair_margin"] for r in records], dtype=float)
        successes = sum(r["corrected_left"] > 0 and r["corrected_right"] > 0
                        for r in records)
        score = float(np.mean(margins) - 0.25 * np.std(margins) +
                      0.3 * successes / len(records))
        return {"score": score, "opposite_sign_pairs": successes,
                "seeds": len(seeds), "records": records}


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def initial_population(rng: np.random.Generator, size: int) -> np.ndarray:
    pop = rng.uniform(BOUNDS[:, 0], BOUNDS[:, 1], size=(size, len(BOUNDS)))
    anchors = (("IN08B051_d", 1, 1, 1.0, 1.0),
               ("SNpp37_38_06", 1, 1, 1.0, 1.0),
               ("SNpp_bundle", 1, 1, 1.0, 1.0),
               ("SApp08", 1, 1, 1.0, 1.0),
               ("DNa04", 1, 1, 1.0, 1.0))
    for i, (group, polarity, orientation, gain, duty) in enumerate(anchors[:size]):
        pop[i] = [GROUP_NAMES.index(group), float(polarity == 1),
                  float(orientation == 1), gain, duty]
    return pop


def make_trial(rng: np.random.Generator, population: np.ndarray, slot: int) -> np.ndarray:
    choices = [i for i in range(len(population)) if i != slot]
    a, b, c = rng.choice(choices, size=3, replace=False)
    mutant = np.clip(population[a] + 0.8 * (population[b] - population[c]),
                     BOUNDS[:, 0], BOUNDS[:, 1])
    mask = rng.random(population.shape[1]) < 0.7
    mask[int(rng.integers(0, len(mask)))] = True
    trial = np.where(mask, mutant, population[slot])
    # Category mutation keeps all cell groups reachable after DE crossover.
    if rng.random() < 0.25:
        trial[0] = int(rng.integers(0, len(GROUP_NAMES)))
    return trial


def configure(fly: WingMotorDrive, decoded: dict) -> None:
    fly.configure_roll_probe(decoded["types"], gain=decoded["gain"],
                             polarity=decoded["polarity"],
                             orientation=decoded["orientation"],
                             duty=decoded["duty"])


def physics(fly: WingMotorDrive, decoded: dict | None,
            scenarios: tuple[tuple[int, float], ...]) -> dict:
    fly.probe_feedback = None
    if decoded is not None:
        configure(fly, decoded)
    runs = [episode(fly, seed, rate) for seed, rate in scenarios]
    return {"configuration": decoded,
            "survived": sum(r["status"] == "completed" for r in runs),
            "mean_duration_s": float(np.mean([r["duration_s"] for r in runs])),
            "mean_abs_roll_rad": float(np.mean([
                r["abs_roll_integral_rad_s"] / r["duration_s"] for r in runs])),
            "runs": runs}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--population", type=int, default=24)
    parser.add_argument("--generations", type=int, default=10)
    parser.add_argument("--train-seeds", type=int, default=4)
    parser.add_argument("--output", type=Path,
                        default=Path("runs/neural-stimulation-search"))
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.population < 5 or args.generations < 1 or args.train_seeds < 2:
        parser.error("population >= 5, generations >= 1, train-seeds >= 2 required")
    args.output.mkdir(parents=True, exist_ok=True)
    lock_handle = (args.output / "run.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        parser.error(f"another search is writing to {args.output}")
    checkpoint = args.output / "checkpoint.json"
    output = args.output / "results.json"
    if output.exists():
        parser.error(f"completed results already exist: {output}")
    train = tuple(range(800, 800 + args.train_seeds))
    heldout = tuple(range(1200, 1212))
    config = {"population": args.population, "generations": args.generations,
              "train_seeds": train, "heldout_seeds": heldout,
              "groups": GROUPS, "bounds": BOUNDS.tolist(), "rng_seed": 2026}
    if args.resume:
        if not checkpoint.exists():
            parser.error(f"checkpoint missing: {checkpoint}")
        state = json.loads(checkpoint.read_text())
        if state["config"] != json.loads(json.dumps(config)):
            parser.error("checkpoint configuration differs from requested run")
        rng = np.random.default_rng()
        rng.bit_generator.state = state["rng_state"]
        population = np.asarray(state["population"], dtype=float)
        energies = state["energies"]
    else:
        if checkpoint.exists():
            parser.error(f"checkpoint exists; pass --resume or choose another output: {checkpoint}")
        rng = np.random.default_rng(2026)
        population = initial_population(rng, args.population)
        energies = [None] * args.population
        state = {"config": config, "generation": 0, "slot": 0,
                 "evaluations": 0, "neural_runs": 0, "history": []}
    probe = NeuralProbe(train, heldout)
    started = perf_counter()
    while state["generation"] <= args.generations:
        generation, slot = state["generation"], state["slot"]
        if slot >= args.population:
            state["generation"] += 1
            state["slot"] = 0
            atomic_json(checkpoint, {**state, "population": population.tolist(),
                                     "energies": energies, "rng_state": rng.bit_generator.state})
            best = max(v for v in energies if v is not None)
            print(f"generation {generation}: best neural score {best:.3f}; "
                  f"{state['neural_runs']} stochastic neural runs", flush=True)
            continue
        candidate = population[slot].copy() if generation == 0 else make_trial(rng, population, slot)
        decoded = decode(candidate)
        result = probe.evaluate(decoded, train)
        accepted = energies[slot] is None or result["score"] > energies[slot]
        if accepted:
            population[slot] = candidate
            energies[slot] = result["score"]
        state["history"].append({"evaluation": state["evaluations"],
                                 "generation": generation, "slot": slot,
                                 "vector": candidate.tolist(), "candidate": decoded,
                                 "accepted": accepted, "train": result})
        state["evaluations"] += 1
        state["neural_runs"] += 2 * len(train)
        state["slot"] += 1
        atomic_json(checkpoint, {**state, "population": population.tolist(),
                                 "energies": energies, "rng_state": rng.bit_generator.state})
    ranked = sorted(range(args.population), key=lambda i: energies[i], reverse=True)
    finalists = []
    seen_groups = set()
    for i in ranked:
        decoded = decode(population[i])
        if decoded["group"] not in seen_groups:
            finalists.append(decoded)
            seen_groups.add(decoded["group"])
        if len(finalists) >= 6:
            break
    # Always compare the previously observed sensory route under the same new seeds.
    reference = decode(np.asarray([GROUP_NAMES.index("SNpp_bundle"), 1, 1, 1.0, 1.0]))
    if reference["group"] not in seen_groups:
        finalists.append(reference)
    neural_heldout = [{"candidate": item, "result": probe.evaluate(item, heldout)}
                      for item in finalists]
    fly = WingMotorDrive(seed=7, ticks_per_action=5, warmup_actions=48,
                         mapping="wing_wrench")
    selection_scenarios = tuple((1300 + i, (0.4, -0.4, 0.8, -0.8)[i % 4])
                                for i in range(4))
    physics_selection = [physics(fly, item, selection_scenarios) for item in finalists]
    winner = max(physics_selection,
                 key=lambda p: (p["survived"], -p["mean_abs_roll_rad"]))["configuration"]
    test_scenarios = tuple((1400 + i, (0.4, -0.4, 0.8, -0.8)[i % 4])
                           for i in range(8))
    physics_test = {
        "winner": physics(fly, winner, test_scenarios),
        "no_stimulation": physics(fly, None, test_scenarios),
        "reference_bundle": physics(fly, reference, test_scenarios),
        "reversed_winner": physics(fly, {**winner, "orientation": -winner["orientation"]},
                                   test_scenarios),
    }
    configure(fly, winner)
    fly.reset(1500)
    chase = fly_wing_direct(fly, duration=8.0, decision_interval=0.1)
    result = {"method": "mixed discrete/continuous differential evolution rand/1/bin",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; weights frozen",
              "counting": "one neural run is one stochastic reset of the same connectome, not a distinct fly connectome",
              "selection": "neural train -> neural heldout -> 4 physics selection starts -> 8 physics test starts -> moving-target transfer",
              "config": config, "evaluations": state["evaluations"],
              "neural_runs_train": state["neural_runs"],
              "neural_runs_heldout": 2 * len(heldout) * len(finalists),
              "rest_runs": len(probe.rest),
              "wall_s": perf_counter() - started,
              "finalists": finalists, "neural_heldout": neural_heldout,
              "physics_selection_scenarios": selection_scenarios,
              "physics_selection": physics_selection,
              "physics_test_scenarios": test_scenarios,
              "physics_test": physics_test,
              "winner": winner, "chase_transfer": chase,
              "checkpoint": str(checkpoint)}
    atomic_json(output, result)
    saved_checkpoint = json.loads(checkpoint.read_text())
    (args.output / "report.md").write_text(
        render_report(result, saved_checkpoint), encoding="utf-8")
    print(json.dumps({"evaluations": result["evaluations"],
                      "neural_runs_train": result["neural_runs_train"],
                      "winner": winner,
                      "test_survival": {k: v["survived"] for k, v in physics_test.items()},
                      "chase_status": chase["status"],
                      "chase_duration_s": chase["duration"],
                      "wall_s": result["wall_s"],
                      "results": str(output.resolve())}, indent=2), flush=True)


if __name__ == "__main__":
    main()
