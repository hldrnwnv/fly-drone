"""Matched MuJoCo roll perturbations for one connectome premotor candidate."""

from __future__ import annotations

import json
from pathlib import Path

import mujoco
import numpy as np

from fly_drone.fpv import FPVQuad
from fly_drone.wing_motors import WingMotorDrive


SCENARIOS = tuple((200 + i, (0.4, -0.4, 0.8, -0.8)[i % 4]) for i in range(8))
MODES = ("none", "premotor", "premotor_reversed", "sensory_bundle",
         "sensory_bundle_reversed", "sensory_triplet", "sensory_triplet_reversed")


def episode(controller: WingMotorDrive, seed: int, initial_rate: float,
            duration: float = 4.0, decision_interval: float = 0.1) -> dict:
    controller.reset(seed)
    quad = FPVQuad(show_cow=False, show_gates=False)
    quad.data.qvel[3] = initial_rate
    mujoco.mj_forward(quad.model, quad.data)
    motors = np.full(4, quad.mass * 9.81 / 4)
    interval_steps = round(decision_interval / quad.dt)
    abs_roll_integral = 0.0
    abs_height_integral = 0.0
    max_abs_roll = 0.0
    decisions = []
    status = "completed"
    for step in range(round(duration / quad.dt)):
        state = quad.state()
        if state.position[2] < 0.12 or abs(state.roll) > 1.3 or abs(state.pitch) > 1.3:
            status = "crashed"
            break
        if step % interval_steps == 0:
            sample = controller.command(0.0, state.body_rates,
                                        mass_kg=quad.mass, mixer=quad.mixer)
            motors = np.asarray(sample.motor_forces_n)
            decisions.append({"t": float(quad.data.time),
                              "roll_rad": state.roll,
                              "roll_rate_rad_s": float(state.body_rates[0]),
                              "roll_torque_nm": sample.achieved_wrench[1],
                              "premotor_drive": sample.sensory_drive,
                              "i1_i2_spikes": {key: value for key, value in
                                               sample.spike_counts.items()
                                               if key.startswith(("i1 MN", "i2 MN"))}})
        # Shared external altitude correction isolates the roll-control question.
        correction = quad.mass * (4.0 * (1.2 - state.position[2]) -
                                  2.8 * state.velocity[2])
        quad.data.ctrl[:] = np.clip(motors + correction / 4, 0, 8)
        mujoco.mj_step(quad.model, quad.data)
        abs_roll_integral += abs(state.roll) * quad.dt
        abs_height_integral += abs(state.position[2] - 1.2) * quad.dt
        max_abs_roll = max(max_abs_roll, abs(state.roll))
    final = quad.state()
    damping_decisions = [d for d in decisions if abs(d["roll_rate_rad_s"]) >= 0.1]
    damping_fraction = (sum(d["roll_rate_rad_s"] * d["roll_torque_nm"] < 0
                            for d in damping_decisions) / len(damping_decisions)
                        if damping_decisions else None)
    return {"seed": seed, "initial_roll_rate_rad_s": initial_rate,
            "status": status, "duration_s": float(quad.data.time),
            "abs_roll_integral_rad_s": abs_roll_integral,
            "abs_altitude_error_integral_m_s": abs_height_integral,
            "max_abs_roll_rad": max_abs_roll,
            "final_altitude_m": float(final.position[2]),
            "final_roll_rad": float(final.roll),
            "damping_fraction": damping_fraction,
            "decisions": decisions}


def main() -> None:
    controllers = {
        mode: WingMotorDrive(seed=7, ticks_per_action=5, warmup_actions=48,
                             mapping="wing_wrench",
                             premotor_roll=mode.startswith("premotor"),
                             sensory_roll=mode.startswith("sensory_"),
                             sensory_roll_types=(("SNpp37", "SNpp38", "SNpp06")
                                                 if mode.startswith("sensory_triplet") else
                                                 ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")),
                             premotor_reversed=mode.endswith("reversed"))
        for mode in MODES}
    runs = {mode: [episode(controller, seed, rate) for seed, rate in SCENARIOS]
            for mode, controller in controllers.items()}
    summary = {mode: {"survived": sum(r["status"] == "completed" for r in episodes),
                      "mean_duration_s": float(np.mean([r["duration_s"] for r in episodes])),
                      "mean_roll_integral_rad_s": float(np.mean(
                          [r["abs_roll_integral_rad_s"] for r in episodes])),
                      "mean_time_normalized_abs_roll_rad": float(np.mean(
                          [r["abs_roll_integral_rad_s"] / r["duration_s"]
                           for r in episodes])),
                      "mean_max_abs_roll_rad": float(np.mean(
                          [r["max_abs_roll_rad"] for r in episodes])),
                      "mean_damping_fraction": float(np.mean(
                          [r["damping_fraction"] for r in episodes
                           if r["damping_fraction"] is not None]))}
               for mode, episodes in runs.items()}
    result = {"hypothesis": "right-side stimulation for negative roll rate and left for positive",
              "brain": "MaleCNS v1.0 via flybrain 0.1.0; weights frozen",
              "candidate_selection": "from separate open-loop six-seed premotor probe",
              "conditions": {"none": "no stimulation",
                             "premotor": "IN08B051_d, predicted damping side",
                             "premotor_reversed": "IN08B051_d, opposite side",
                             "sensory_bundle": "SNpp26/27/37/38/06, predicted damping side",
                             "sensory_bundle_reversed": "SNpp26/27/37/38/06, opposite side",
                             "sensory_triplet": "SNpp37/38/06, predicted damping side",
                             "sensory_triplet_reversed": "SNpp37/38/06, opposite side"},
              "decision_interval_s": 0.1, "stimulus_cap": 0.5,
              "altitude_assist": "same external proportional-derivative altitude correction in all modes",
              "duration_limit_s": 4.0, "scenarios": SCENARIOS,
              "summary": summary, "runs": runs}
    output = Path("runs/wing-premotor-probe/roll-perturbations-triplet.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Results: {output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
