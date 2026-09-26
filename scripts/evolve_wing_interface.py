"""Evolve only the sensor/wing-MN/rotor interface for roll recovery.

MaleCNS connections and flybrain dynamics remain frozen. This is a diagnostic
hover task with imposed roll-rate perturbations, not pursuit training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from time import perf_counter

import mujoco
import numpy as np

from fly_drone.fpv import FPVQuad
from fly_drone.wing_motors import OTHER_STEERING_TYPES, STEERING_TYPES, WingMotorDrive


NAMES = ("power_gain", "memory", "side_gain", "fore_gain", "haltere_gain")
LOW = np.array([0.7, 0.4, -2.0, -1.0, -3.0])
HIGH = np.array([1.4, 0.98, 2.0, 1.0, 3.0])
TRAIN = ((100, 0.6), (101, -0.6))
TEST = ((200, 0.4), (201, -0.4), (202, 0.9), (203, -0.9))


def set_genome(fly: WingMotorDrive, genome: np.ndarray) -> None:
    for name, value in zip(NAMES, genome):
        setattr(fly, name, float(value))


def episode(fly: WingMotorDrive, seed: int, initial_roll_rate: float,
            duration: float) -> dict:
    fly.reset(seed)
    quad = FPVQuad(show_cow=False, show_gates=False)
    quad.data.qvel[3] = initial_roll_rate
    mujoco.mj_forward(quad.model, quad.data)
    motors = np.zeros(4)
    total_roll = total_altitude = 0.0
    motor_decisions = 0
    steering_spikes = 0
    status = "completed"
    control_steps = round(fly.ticks_per_action * fly.brain.dt / quad.dt)
    for step in range(round(duration / quad.dt)):
        state = quad.state()
        if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
            status = "crashed"
            break
        if step % control_steps == 0:
            sample = fly.command(0.0, state.body_rates)
            motors = np.asarray(sample.motor_forces_n)
            motor_decisions += 1
            steering_spikes += sum(value for name, value in sample.spike_counts.items()
                                   if name.split("_")[0] in fly.steering_types)
        quad.data.ctrl[:] = motors
        mujoco.mj_step(quad.model, quad.data)
        total_roll += abs(state.roll) * quad.dt
        total_altitude += abs(state.position[2] - 1.2) * quad.dt
    state = quad.state()
    alive_s = float(quad.data.time)
    score = alive_s - 0.35 * total_roll - 0.25 * total_altitude
    return {"seed": seed, "initial_roll_rate_rad_s": initial_roll_rate,
            "status": status, "duration_s": alive_s, "score": score,
            "last_altitude_m": float(state.position[2]),
            "last_roll_rad": float(state.roll),
            "steering_spikes": steering_spikes, "motor_decisions": motor_decisions}


def evaluate(fly: WingMotorDrive, genome: np.ndarray, scenarios,
             duration: float) -> dict:
    set_genome(fly, genome)
    episodes = [episode(fly, seed, rate, duration) for seed, rate in scenarios]
    return {"parameters": dict(zip(NAMES, map(float, genome))),
            "mean_score": float(np.mean([run["score"] for run in episodes])),
            "survived": sum(run["status"] == "completed" for run in episodes),
            "episodes": episodes}


def classical_pd_episode(seed: int, initial_roll_rate: float,
                         duration: float) -> dict:
    """Conventional rate/attitude control on exactly the same quad and starts."""
    quad = FPVQuad(show_cow=False, show_gates=False)
    quad.data.qvel[3] = initial_roll_rate
    mujoco.mj_forward(quad.model, quad.data)
    total_roll = total_altitude = 0.0
    status = "completed"
    for _ in range(round(duration / quad.dt)):
        state = quad.state()
        if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
            status = "crashed"
            break
        vertical_accel = 4.0 * (1.2 - state.position[2]) - 2.8 * state.velocity[2]
        thrust = 0.75 * (9.81 + vertical_accel)
        roll_torque = -0.04 * state.roll - 0.025 * state.body_rates[0]
        pitch_torque = -0.04 * state.pitch - 0.025 * state.body_rates[1]
        yaw_torque = -0.03 * state.body_rates[2]
        quad.apply(thrust, roll_torque, pitch_torque, yaw_torque)
        total_roll += abs(state.roll) * quad.dt
        total_altitude += abs(state.position[2] - 1.2) * quad.dt
    state = quad.state()
    return {"seed": seed, "initial_roll_rate_rad_s": initial_roll_rate,
            "status": status, "duration_s": float(quad.data.time),
            "score": float(quad.data.time) - 0.35 * total_roll - 0.25 * total_altitude,
            "last_altitude_m": float(state.position[2]),
            "last_roll_rad": float(state.roll)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--duration", type=float, default=3.0)
    parser.add_argument("--baseline-only", action="store_true",
                        help="run the conventional PD check without neural search")
    parser.add_argument("--output", type=Path,
                        default=Path("runs/wing-evolution/results.json"))
    args = parser.parse_args()
    if args.generations < 1 or args.population < 4 or args.duration <= 0:
        parser.error("generations >= 1, population >= 4, duration > 0 required")
    if args.baseline_only:
        episodes = [classical_pd_episode(seed, rate, args.duration)
                    for seed, rate in TEST]
        result = {"controller": "conventional altitude and angular-rate/attitude PD",
                  "scenarios": TEST, "duration_limit_s": args.duration,
                  "survived": sum(e["status"] == "completed" for e in episodes),
                  "mean_score": float(np.mean([e["score"] for e in episodes])),
                  "episodes": episodes}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps({"survived": result["survived"],
                          "mean_score": result["mean_score"]}, indent=2))
        return
    rng = np.random.default_rng(2026)
    fly = WingMotorDrive(seed=7, ticks_per_action=5, warmup_actions=48,
                         haltere_feedback=True,
                         steering_types=STEERING_TYPES + OTHER_STEERING_TYPES)
    default = np.array([1.0, 0.8, 0.35, 0.25, 1.0])
    nearly_constant = np.array([1.0, 0.98, 0.0, 0.0, 0.0])
    population = [default, nearly_constant]
    population.extend(rng.uniform(LOW, HIGH) for _ in range(args.population - 2))
    history = []
    started = perf_counter()
    for generation in range(args.generations):
        evaluated = [evaluate(fly, genome, TRAIN, args.duration)
                     for genome in population]
        ranked = sorted(zip(population, evaluated),
                        key=lambda pair: pair[1]["mean_score"], reverse=True)
        history.append({"generation": generation,
                        "candidates": evaluated,
                        "best_score": ranked[0][1]["mean_score"]})
        print(f"generation {generation}: best={ranked[0][1]['mean_score']:.3f}, "
              f"survived={ranked[0][1]['survived']}/2", flush=True)
        if generation == args.generations - 1:
            break
        elites = [ranked[0][0].copy(), ranked[1][0].copy()]
        next_generation = elites.copy()
        while len(next_generation) < args.population:
            parent_a = ranked[int(rng.integers(0, min(4, len(ranked))))][0]
            parent_b = ranked[int(rng.integers(0, min(4, len(ranked))))][0]
            blend = rng.uniform(0.0, 1.0)
            child = blend * parent_a + (1 - blend) * parent_b
            child += rng.normal(0, 0.13, size=len(NAMES)) * (HIGH - LOW)
            next_generation.append(np.clip(child, LOW, HIGH))
        population = next_generation
    best_genome, best_train = ranked[0]
    result = {"method": "elitist real-valued genetic algorithm",
              "objective": "recover level attitude and hover after imposed roll rates",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; all graph weights frozen",
              "neural_timestep_s": fly.brain.dt,
              "decision_interval_s": fly.ticks_per_action * fly.brain.dt,
              "steering_types": list(fly.steering_types),
              "train_scenarios": TRAIN, "heldout_scenarios": TEST,
              "duration_limit_s": args.duration,
              "generations": args.generations, "population": args.population,
              "fitness": "alive seconds minus 0.35 integral abs(roll) minus 0.25 integral abs(height-1.2)",
              "history": history, "best_train": best_train,
              "best_heldout": evaluate(fly, best_genome, TEST, args.duration),
              "default_heldout": evaluate(fly, default, TEST, args.duration),
              "near_constant_heldout": evaluate(fly, nearly_constant, TEST, args.duration),
              "classical_pd_heldout": [classical_pd_episode(seed, rate, args.duration)
                                       for seed, rate in TEST],
              "wall_s": perf_counter() - started}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"best_train": best_train["mean_score"],
                      "best_heldout": result["best_heldout"]["mean_score"],
                      "best_heldout_survived": result["best_heldout"]["survived"],
                      "default_heldout": result["default_heldout"]["mean_score"],
                      "near_constant_heldout": result["near_constant_heldout"]["mean_score"],
                      "wall_s": result["wall_s"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
