"""A calibrated MaleCNS odor-sensor readout with explicit wind guidance.

This is an engineering navigation adapter. It reads olfactory receptor spikes,
not an innate descending motor command from the connectome.
"""

from __future__ import annotations

from math import atan2

import numpy as np
from flybrain import FlyBrain

from .olfaction import FoodOdorPlume
from .sim import wrap_angle


class WindFoodOdorPlume(FoodOdorPlume):
    """Add an idealized wind vane; no source pose is delivered to the controller."""

    def sample(self, position_xy: np.ndarray, yaw: float,
               source_xy: np.ndarray, time_s: float) -> dict[str, float]:
        sample = super().sample(position_xy, yaw, source_xy, time_s)
        sample["upwind_bearing"] = wrap_angle(
            atan2(-self.wind[1], -self.wind[0]) - yaw)
        return sample


class MaleCNSOdorReadout:
    """Calibrate a left-minus-right ORN spike-rate readout on synthetic doses."""

    def __init__(self, *, seed: int = 7, ticks_per_decision: int = 20):
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        self.ticks_per_decision = ticks_per_decision
        self.orn = {side: np.concatenate((
            self.brain.cells(["ORN_DM1"], side=side),
            self.brain.cells(["ORN_VA2"], side=side))) for side in "LR"}
        self.pn = {side: np.concatenate((
            self.brain.cells(["DM1_lPN"], side=side),
            self.brain.cells(["VA2_adPN"], side=side))) for side in "LR"}
        self.dna02 = self.brain.cells(["DNa02"])
        if any(not len(cells) for cells in self.orn.values()):
            raise RuntimeError("MaleCNS bilateral DM1/VA2 ORNs are required")
        self.masks = {f"ORN_{side}": self._mask(self.orn[side]) for side in "LR"}
        self.masks.update({f"PN_{side}": self._mask(self.pn[side]) for side in "LR"})
        self.masks["DNa02"] = self._mask(self.dna02)
        self.gain: float | None = None
        self.last_counts: dict[str, int] = {}

    def _mask(self, cells: np.ndarray) -> np.ndarray:
        mask = np.zeros(self.brain.n, dtype=bool)
        mask[cells] = True
        return mask

    def reset(self, seed: int) -> None:
        self.brain.reset(seed)
        self.last_counts = {}

    def observe(self, left: float, right: float) -> float:
        if not 0 <= left <= 1 or not 0 <= right <= 1:
            raise ValueError("odor concentrations must be in [0, 1]")
        inject = [(self.orn[side], 0.25 * concentration)
                  for side, concentration in (("L", left), ("R", right))
                  if concentration > 0]
        counts = {name: 0 for name in self.masks}
        for _ in range(self.ticks_per_decision):
            fired = self.brain.step(inject=inject)
            for name, mask in self.masks.items():
                counts[name] += int(np.count_nonzero(mask[fired]))
        self.last_counts = counts
        return (counts["ORN_L"] / (len(self.orn["L"]) * self.ticks_per_decision)
                - counts["ORN_R"] / (len(self.orn["R"]) * self.ticks_per_decision))

    def fit(self, *, train_seeds: range = range(500, 508),
            validation_seeds: range = range(800, 804)) -> dict:
        pairs = ((0.0, 0.0), (0.1, 0.0), (0.0, 0.1),
                 (0.2, 0.05), (0.05, 0.2), (0.3, 0.1), (0.1, 0.3),
                 (0.5, 0.2), (0.2, 0.5), (0.7, 0.3), (0.3, 0.7))

        def collect(seeds: range) -> tuple[np.ndarray, np.ndarray, list[dict]]:
            features, labels, records = [], [], []
            for seed in seeds:
                for left, right in pairs:
                    self.reset(seed)
                    feature = self.observe(left, right)
                    features.append(feature)
                    labels.append(left - right)
                    records.append({"seed": seed, "L": left, "R": right,
                                    "ORN_rate_difference": feature,
                                    "spikes": self.last_counts.copy()})
            return np.asarray(features), np.asarray(labels), records

        train_x, train_y, training = collect(train_seeds)
        self.gain = float(np.dot(train_x, train_y) / (np.dot(train_x, train_x) + 1e-9))
        val_x, val_y, validation = collect(validation_seeds)
        prediction = np.clip(self.gain * val_x, -1, 1)
        active = np.abs(val_y) > 0.05
        return {"gain": self.gain,
                "features": "normalized ORN_DM1/VA2 left minus right spike count",
                "target": "injected left minus right concentration",
                "train_seeds": list(train_seeds),
                "validation_seeds": list(validation_seeds),
                "train_records": training,
                "validation_records": [
                    {**record, "prediction": float(estimate)}
                    for record, estimate in zip(validation, prediction)],
                "validation_mae": float(np.mean(np.abs(prediction - val_y))),
                "validation_sign_accuracy": float(np.mean(
                    np.sign(prediction[active]) == np.sign(val_y[active])))}

    def decode(self, left: float, right: float) -> float:
        if self.gain is None:
            raise RuntimeError("Call fit() before decode()")
        return float(np.clip(self.gain * self.observe(left, right), -1, 1))


class OdorWindController:
    """Upwind guidance plus a left/right odor correction at fixed flight speed."""

    def __init__(self, *, mode: str, readout: MaleCNSOdorReadout | None = None):
        if mode not in ("wind_only", "raw_odor", "neural_odor",
                        "neural_swapped", "neural_zero"):
            raise ValueError("unknown odor navigation mode")
        if mode.startswith("neural") and readout is None:
            raise ValueError("neural mode requires an olfactory readout")
        self.mode = mode
        self.readout = readout

    def __call__(self, _tag_bearing: float,
                 sample: dict[str, float]) -> tuple[float, dict]:
        left, right = sample["L"], sample["R"]
        if self.mode == "wind_only":
            difference = 0.0
        elif self.mode == "raw_odor":
            difference = left - right
        else:
            if self.mode == "neural_swapped":
                left, right = right, left
            elif self.mode == "neural_zero":
                left, right = 0.0, 0.0
            difference = self.readout.decode(left, right)
        command = float(np.clip(0.6 * sample["upwind_bearing"]
                                + 2.0 * difference, -1.0, 1.0))
        details = {"mode": self.mode, "decoded_L_minus_R": difference,
                   "upwind_bearing": sample["upwind_bearing"],
                   "wind_component": 0.6 * sample["upwind_bearing"],
                   "odor_component": 2.0 * difference}
        if self.readout is not None and self.mode.startswith("neural"):
            details["spikes"] = self.readout.last_counts.copy()
        return command, details


class VisualOdorFusion:
    """Use vision while present and a wind/odor policy during tag dropout."""

    def __init__(self, *, mode: str, visual_readout,
                 odor_readout: MaleCNSOdorReadout | None = None):
        if mode not in ("tag_only", "tag_wind_only", "tag_raw_odor",
                        "tag_neural_odor", "tag_neural_swapped"):
            raise ValueError("unknown fusion mode")
        self.mode = mode
        self.visual_readout = visual_readout
        fallback_mode = mode.removeprefix("tag_")
        self.fallback = (None if mode == "tag_only" else
                         OdorWindController(mode=fallback_mode, readout=odor_readout))

    def __call__(self, bearing: float,
                 sample: dict[str, float]) -> tuple[float, dict]:
        visible = bool(sample["tag_visible"])
        visual_command = self.visual_readout.action(bearing if visible else 0.0)
        fallback_command, fallback_details = (
            self.fallback(0.0, sample) if self.fallback is not None
            else (0.0, {"mode": "none"}))
        return (visual_command if visible else fallback_command,
                {"mode": "fusion_" + self.mode,
                 "used": "vision" if visible else "wind_odor_fallback",
                 "visual_command": visual_command,
                 "fallback_command": fallback_command,
                 "fallback": fallback_details})
