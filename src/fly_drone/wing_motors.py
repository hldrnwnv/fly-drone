"""Exploratory MaleCNS wing motor neuron to quad rotor interface.

This uses the existing flybrain dynamics, not the Shiu/Eon FlyWire model.
The rotor mapping is an engineering hypothesis, not a fly muscle model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from math import pi

import numpy as np
from flybrain import FlyBrain

from .perception import VisualFeatures
from .retina import RetinaProjector
from .sensory_adapter import FlySensoryEncoder
from .wing_adapter import WingIntent, WingRotorAdapter


POWER_TYPES = ("DLMn a, b", "DLMn c-f", "DVMn 1a-c")
STEERING_TYPES = ("b1 MN", "b2 MN")
OTHER_STEERING_TYPES = ("b3 MN", "i1 MN", "i2 MN", "iii1 MN",
                        "iii3 MN", "hg1 MN", "hg2 MN", "hg3 MN", "hg4 MN")
SENSORY_ROLL_TYPES = ("SNpp26", "SNpp27", "SNpp37", "SNpp38", "SNpp06")


@dataclass(frozen=True)
class WingSample:
    bearing_rad: float
    spike_counts: dict[str, int]
    motor_forces_n: list[float]
    haltere_drive: dict[str, float]
    visual_drive: dict
    wing_intent: dict | None = None
    desired_wrench: list[float] | None = None
    achieved_wrench: list[float] | None = None
    sensory_drive: dict[str, float] | None = None


class WingMotorDrive:
    """Observe named wing motor cells and map their activity to four thrusts.

    DLM/DVM activity supplies left/right collective thrust. b1 and b2
    activity supplies a front/rear difference. Optional roll feedback injects
    measured angular speed into selected cells. It does not control altitude,
    choose a target speed, or learn a behavioral readout.
    """

    def __init__(self, *, seed: int = 7, ticks_per_action: int = 20,
                 warmup_actions: int = 12, haltere_feedback: bool = False,
                 retina_input: bool = False,
                 steering_types: tuple[str, ...] = STEERING_TYPES,
                 mapping: str = "raw", haltere_3d: bool = False,
                 neural_vision: bool = False,
                 premotor_roll: bool = False,
                 sensory_roll: bool = False,
                 sensory_roll_types: tuple[str, ...] = SENSORY_ROLL_TYPES,
                 premotor_gain: float = 1.0,
                 premotor_cap: float = 0.5,
                 premotor_reversed: bool = False):
        if ticks_per_action < 1 or warmup_actions < 1:
            raise ValueError("ticks_per_action and warmup_actions must be positive")
        if mapping not in ("raw", "wing_wrench"):
            raise ValueError("mapping must be raw or wing_wrench")
        if mapping == "wing_wrench":
            steering_types = ("b1 MN", "b2 MN", "b3 MN", "i1 MN", "i2 MN")
        if premotor_roll and sensory_roll:
            raise ValueError("choose premotor_roll or sensory_roll")
        self.mapping = mapping
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        self.ticks_per_action = ticks_per_action
        self.left_visual = self.brain.cells(["LC10a"], side="L")
        self.right_visual = self.brain.cells(["LC10a"], side="R")
        self.haltere_feedback = haltere_feedback
        self.haltere_3d = haltere_3d
        self.neural_vision = neural_vision
        self.premotor_roll = premotor_roll
        self.sensory_roll = sensory_roll
        self.sensory_roll_types = sensory_roll_types
        self.premotor_gain = premotor_gain
        self.premotor_cap = premotor_cap
        self.premotor_reversed = premotor_reversed
        self.probe_feedback: dict | None = None
        self._probe_cells: dict[tuple[str, ...], dict[str, np.ndarray]] = {}
        if premotor_gain <= 0 or premotor_cap <= 0:
            raise ValueError("premotor gain and cap must be positive")
        self.retina_input = retina_input
        self.retina_projector = RetinaProjector(self.brain.azimuth) if retina_input else None
        self.steering_types = steering_types
        self.power_gain = 1.0
        self.memory = 0.8
        self.side_gain = 0.35
        self.fore_gain = 0.25
        self.haltere_gain = 1.0
        self.haltere = {side: self.brain.cells(["SApp08"], side=side)
                        for side in ("L", "R")}
        self.groups = {f"{name}_{side}": self.brain.cells([name], side=side)
                       for name in (*POWER_TYPES, *steering_types)
                       for side in ("L", "R")}
        self.sensory_encoder = (FlySensoryEncoder(self.brain)
                                if haltere_3d or neural_vision else None)
        self.premotor_cells = ({side: self.brain.cells(["IN08B051_d"], side=side)
                                for side in "LR"} if premotor_roll else {})
        self.sensory_roll_cells = ({side: self.brain.cells(
            list(sensory_roll_types), side=side)
            for side in "LR"} if sensory_roll else {})
        if premotor_roll and any(not len(cells) for cells in self.premotor_cells.values()):
            raise RuntimeError("IN08B051_d cells missing from MaleCNS")
        if sensory_roll and any(not len(cells) for cells in self.sensory_roll_cells.values()):
            raise RuntimeError("SNpp sensory cells missing from MaleCNS")
        if not len(self.left_visual) or not len(self.right_visual):
            raise RuntimeError("LC10a visual neurons absent from MaleCNS data")
        if any(not len(cells) for cells in self.groups.values()):
            missing = [name for name, cells in self.groups.items() if not len(cells)]
            raise RuntimeError(f"Wing motor cells absent from MaleCNS data: {missing}")
        if haltere_feedback and any(not len(cells) for cells in self.haltere.values()):
            raise RuntimeError("SApp08 haltere sensory cells absent from MaleCNS data")
        self.power = {side: np.concatenate([self.groups[f"{name}_{side}"]
                                            for name in POWER_TYPES]) for side in ("L", "R")}
        self.power_trace = {"L": 0.0, "R": 0.0}
        self.b1_trace = {"L": 0.0, "R": 0.0}
        self.b2_trace = {"L": 0.0, "R": 0.0}
        self.other_trace = {"L": 0.0, "R": 0.0}
        self.steering_trace = {f"{name}_{side}": 0.0 for name in steering_types
                               for side in ("L", "R")}
        self.newtons_per_power_spike = 0.0
        self.warmup_record: list[dict[str, int]] = []
        self.calibrate(warmup_actions)

    def _spike_counts(self, bearing: float,
                      haltere_drive: dict[str, float] | None = None,
                      eye_drive: np.ndarray | None = None,
                      extra_inject: list[tuple[np.ndarray, float]] | None = None,
                      pulse_inject: list[tuple[np.ndarray, float]] | None = None,
                      pulse_ticks: int = 0) -> dict[str, int]:
        counts = {name: 0 for name in self.groups}
        inject = []
        if not self.retina_input and abs(bearing) >= 0.08:
            amount = 0.55 + 0.55 * min(abs(bearing) / (pi / 2), 1.0)
            inject = [(self.left_visual if bearing > 0 else self.right_visual,
                       amount)]
        if haltere_drive:
            inject.extend((self.haltere[side], amount)
                          for side, amount in haltere_drive.items() if amount > 0)
        if extra_inject:
            inject.extend(extra_inject)
        for tick in range(self.ticks_per_action):
            tick_inject = (inject + pulse_inject
                           if pulse_inject and tick < pulse_ticks else inject)
            fired = self.brain.step(inject=tick_inject, eye_drive=eye_drive)
            for name, cells in self.groups.items():
                counts[name] += int(np.isin(cells, fired).sum())
        return counts

    def configure_roll_probe(self, types: tuple[str, ...], *, gain: float,
                             polarity: int, orientation: int, duty: float) -> None:
        """Set a temporary, signed gyro stimulus without changing connectome weights."""
        if not types or gain <= 0 or polarity not in (-1, 1) or orientation not in (-1, 1):
            raise ValueError("invalid roll probe types, gain, polarity, or orientation")
        if not 0 < duty <= 1:
            raise ValueError("roll probe duty must be in (0, 1]")
        if types not in self._probe_cells:
            cells = {side: self.brain.cells(list(types), side=side) for side in "LR"}
            if any(not len(group) for group in cells.values()):
                raise ValueError(f"roll probe cells absent: {types}")
            self._probe_cells[types] = cells
        self.probe_feedback = {"types": types, "gain": gain,
                               "polarity": polarity, "orientation": orientation,
                               "duty": duty}

    def calibrate(self, warmup_actions: int) -> None:
        """Scale resting power activity to hover thrust; no behavior labels used."""
        self.warmup_record = [self._spike_counts(0.0) for _ in range(warmup_actions)]
        mean = {side: float(np.mean([
            sum(record[f"{name}_{side}"] for name in POWER_TYPES)
            for record in self.warmup_record])) for side in ("L", "R")}
        total = mean["L"] + mean["R"]
        if total <= 0:
            raise RuntimeError("No power-neuron activity during calibration")
        self.newtons_per_power_spike = 0.75 * 9.81 / total
        self.baseline_power = mean
        self.power_trace = mean.copy()
        self.baseline_steering = {key: float(np.mean([record[key]
                                                    for record in self.warmup_record]))
                                  for key in self.steering_trace}
        self.steering_trace = self.baseline_steering.copy()

    def reset(self, seed: int) -> None:
        """Restart one episode using the same preflight thrust calibration."""
        self.brain.reset(seed)
        self.power_trace = self.baseline_power.copy()
        self.b1_trace = {"L": 0.0, "R": 0.0}
        self.b2_trace = {"L": 0.0, "R": 0.0}
        self.other_trace = {"L": 0.0, "R": 0.0}
        self.steering_trace = self.baseline_steering.copy()

    def command(self, bearing: float,
                body_rates: np.ndarray | None = None,
                frame: np.ndarray | None = None,
                *, mass_kg: float | None = None,
                mixer: np.ndarray | None = None,
                visual_features: VisualFeatures | None = None) -> WingSample:
        haltere_drive = {"L": 0.0, "R": 0.0}
        extra_inject: list[tuple[np.ndarray, float]] = []
        pulse_inject: list[tuple[np.ndarray, float]] = []
        pulse_ticks = 0
        sensory_drive: dict[str, float] = {}
        if self.haltere_3d:
            if body_rates is None:
                raise ValueError("body_rates required for 3-axis gyro encoding")
            encoded = self.sensory_encoder.gyro(body_rates)
            extra_inject.extend(encoded.inject)
            sensory_drive.update(encoded.channels)
        elif self.haltere_feedback:
            if body_rates is None:
                raise ValueError("body_rates required for haltere feedback")
            # A signed roll-rate approximation to Coriolis-sensitive input.
            # Which afferents respond to each rotation axis is not modeled.
            roll_rate = float(body_rates[0]) * self.haltere_gain
            haltere_drive["R" if roll_rate > 0 else "L"] = min(abs(roll_rate), 2.0)
        if self.premotor_roll or self.sensory_roll:
            if body_rates is None:
                raise ValueError("body_rates required for roll feedback")
            roll_rate = float(body_rates[0])
            side = "R" if roll_rate < 0 else "L"
            if self.premotor_reversed:
                side = "L" if side == "R" else "R"
            amount = float(np.clip(abs(roll_rate) * self.premotor_gain,
                                   0, self.premotor_cap))
            pathway = "premotor" if self.premotor_roll else "sensory_bundle"
            sensory_drive[f"{pathway}_{side}_drive"] = amount
            if amount:
                cells = (self.premotor_cells if self.premotor_roll
                         else self.sensory_roll_cells)
                extra_inject.append((cells[side], amount))
        if self.probe_feedback is not None:
            if body_rates is None:
                raise ValueError("body_rates required for roll probe feedback")
            probe = self.probe_feedback
            rate = float(body_rates[0])
            side = "R" if rate * probe["orientation"] < 0 else "L"
            amount = probe["polarity"] * min(abs(rate) * probe["gain"], 0.8)
            pulse_ticks = max(1, round(self.ticks_per_action * probe["duty"]))
            sensory_drive[f"probe_{side}_drive"] = amount
            sensory_drive["probe_ticks"] = pulse_ticks
            if amount:
                pulse_inject.append((self._probe_cells[probe["types"]][side], amount))
        eye_drive = None
        visual_drive: dict
        if self.retina_input:
            if frame is None:
                raise ValueError("FPV frame required for retina input")
            # Only horizontal azimuth is available for these prebuilt cells.
            # This 1D brightness projection omits the fly's real eye optics.
            eye_drive = self.retina_projector.encode(frame)
            visual_drive = {"kind": "photoreceptor_brightness_1d",
                            "sha256": hashlib.sha256(eye_drive.tobytes()).hexdigest(),
                            "cells": int(len(eye_drive)),
                            "min": float(eye_drive.min()),
                            "max": float(eye_drive.max()),
                            "mean": float(eye_drive.mean())}
        else:
            visual_drive = {"kind": "LC10a_synthetic",
                            "side": "L" if bearing > 0.08 else "R" if bearing < -0.08 else None,
                            "amount": (0.55 + 0.55 * min(abs(bearing) / (pi / 2), 1.0))
                            if abs(bearing) >= 0.08 else 0.0}
        if self.neural_vision:
            if visual_features is None:
                raise ValueError("neural_vision requires pretrained visual features")
            encoded = self.sensory_encoder.vision(visual_features)
            extra_inject.extend(encoded.inject)
            sensory_drive.update(encoded.channels)
            visual_drive["depth_model"] = visual_features.depth_model
            visual_drive["flow_model"] = visual_features.flow_model
            visual_drive["inference_ms"] = visual_features.inference_ms
        counts = self._spike_counts(bearing, haltere_drive, eye_drive, extra_inject,
                                    pulse_inject, pulse_ticks)
        # Smoothing approximates rotor inertia and a muscle-to-propeller adapter.
        for side in ("L", "R"):
            power = sum(counts[f"{name}_{side}"] for name in POWER_TYPES)
            new_weight = 1 - self.memory
            self.power_trace[side] = self.memory * self.power_trace[side] + new_weight * power
            self.b1_trace[side] = (self.memory * self.b1_trace[side] + new_weight *
                                   counts.get(f"b1 MN_{side}", 0))
            self.b2_trace[side] = (self.memory * self.b2_trace[side] + new_weight *
                                   counts.get(f"b2 MN_{side}", 0))
            other = sum(counts[f"{name}_{side}"] for name in self.steering_types
                        if name not in STEERING_TYPES)
            self.other_trace[side] = self.memory * self.other_trace[side] + new_weight * other
        if self.mapping == "wing_wrench":
            if mass_kg is None or mixer is None:
                raise ValueError("wing_wrench mapping requires quad mass and mixer")
            for key in self.steering_trace:
                self.steering_trace[key] = (self.memory * self.steering_trace[key] +
                                            (1 - self.memory) * counts[key])

            def relative(name: str, side: str) -> float:
                key = f"{name} MN_{side}"
                return self.steering_trace[key] - self.baseline_steering[key]

            amplitude = {side: (relative("b1", side) + relative("b2", side) -
                                relative("b3", side) - relative("i1", side) -
                                relative("i2", side)) for side in ("L", "R")}
            pitch = {side: -relative("i1", side) - relative("i2", side)
                     for side in ("L", "R")}
            baseline_total = sum(self.baseline_power.values())
            power_ratio = sum(self.power_trace.values()) / baseline_total
            intent = WingIntent(power_ratio, amplitude["L"], amplitude["R"],
                                pitch["L"], pitch["R"])
            mixed = WingRotorAdapter(mass_kg=mass_kg, mixer=mixer).mix(intent)
            return WingSample(float(bearing), counts, mixed.thrusts_n,
                              haltere_drive, visual_drive, intent.__dict__,
                              mixed.desired_wrench, mixed.achieved_wrench,
                              sensory_drive)
        left_n = self.power_gain * self.newtons_per_power_spike * self.power_trace["L"]
        right_n = self.power_gain * self.newtons_per_power_spike * self.power_trace["R"]
        side_bias = self.side_gain * ((self.b1_trace["L"] + self.b2_trace["L"] +
                                      self.other_trace["L"]) -
                                     (self.b1_trace["R"] + self.b2_trace["R"] +
                                      self.other_trace["R"]))
        left_n += side_bias
        right_n -= side_bias
        # Quad layout FL, FR, RR, RL. Each fly wing is split over two rotors.
        # The b1/b2 to fore/aft mapping is arbitrary and explicitly testable.
        fore_bias = self.fore_gain * (sum(self.b1_trace.values()) -
                                     sum(self.b2_trace.values()))
        motors = np.clip([left_n / 2 + fore_bias, right_n / 2 + fore_bias,
                          right_n / 2 - fore_bias, left_n / 2 - fore_bias], 0.0, 8.0)
        return WingSample(float(bearing), counts, motors.tolist(),
                          haltere_drive, visual_drive,
                          sensory_drive=sensory_drive)
