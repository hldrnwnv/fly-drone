"""A calibrated MaleCNS odor-sensor readout with explicit wind guidance.

This is an engineering navigation adapter. It reads olfactory receptor spikes,
not an innate descending motor command from the connectome.
"""

from __future__ import annotations

from collections import deque
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
        self.side_coefficients: tuple[float, float] | None = None
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
        side_rates = np.asarray([
            [record["spikes"][f"ORN_{side}"] /
             (len(self.orn[side]) * self.ticks_per_decision) for side in "LR"]
            for record in training]).reshape(-1)
        side_targets = np.asarray([[record[side] for side in "LR"]
                                   for record in training]).reshape(-1)
        zero_rates = np.asarray([
            record["spikes"][f"ORN_{side}"] /
            (len(self.orn[side]) * self.ticks_per_decision)
            for record in training if record["L"] == record["R"] == 0
            for side in "LR"])
        baseline_rate = float(np.mean(zero_rates))
        centered_rates = side_rates - baseline_rate
        gain = float(np.dot(centered_rates, side_targets) /
                     (np.dot(centered_rates, centered_rates) + 1e-9))
        offset = -gain * baseline_rate
        self.side_coefficients = (float(gain), float(offset))
        val_x, val_y, validation = collect(validation_seeds)
        prediction = np.clip(self.gain * val_x, -1, 1)
        validation_side = []
        for record in validation:
            estimated = [float(np.clip(
                gain * record["spikes"][f"ORN_{side}"] /
                (len(self.orn[side]) * self.ticks_per_decision) + offset, 0, 1))
                for side in "LR"]
            validation_side.append(estimated)
        active = np.abs(val_y) > 0.05
        return {"gain": self.gain,
                "side_coefficients": list(self.side_coefficients),
                "side_baseline_rate": baseline_rate,
                "features": "normalized ORN_DM1/VA2 left minus right spike count",
                "target": "injected left minus right concentration",
                "train_seeds": list(train_seeds),
                "validation_seeds": list(validation_seeds),
                "train_records": training,
                "validation_records": [
                    {**record, "prediction": float(estimate)}
                    for record, estimate in zip(validation, prediction)],
                "validation_mae": float(np.mean(np.abs(prediction - val_y))),
                "validation_side_mae": float(np.mean(np.abs(
                    np.asarray(validation_side) -
                    np.asarray([[record[side] for side in "LR"]
                                for record in validation])))),
                "validation_side_predictions": validation_side,
                "validation_sign_accuracy": float(np.mean(
                    np.sign(prediction[active]) == np.sign(val_y[active])))}

    def decode(self, left: float, right: float) -> float:
        if self.gain is None:
            raise RuntimeError("Call fit() before decode()")
        return float(np.clip(self.gain * self.observe(left, right), -1, 1))

    def decode_pair(self, left: float, right: float) -> tuple[float, float]:
        if self.side_coefficients is None:
            raise RuntimeError("Call fit() before decode_pair()")
        self.observe(left, right)
        gain, offset = self.side_coefficients
        return tuple(float(np.clip(
            gain * self.last_counts[f"ORN_{side}"] /
            (len(self.orn[side]) * self.ticks_per_decision) + offset, 0, 1))
            for side in "LR")


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


class OdorHistoryController:
    """Surge upwind on increasing odor; cast across wind after a decline.

    This is an external engineering policy. Bilateral temporal changes choose
    a casting side, rather than directly commanding yaw from current L-R.
    """

    def __init__(self, *, mode: str, readout: MaleCNSOdorReadout | None = None,
                 cast_angle: float = 0.65, detection_threshold: float = 0.035,
                 seed: int = 0):
        if mode not in ("wind_only", "raw_odor", "neural_odor", "neural_swapped"):
            raise ValueError("unknown odor history mode")
        if mode.startswith("neural") and readout is None:
            raise ValueError("neural mode requires an olfactory readout")
        if not 0 < cast_angle < 1.5 or not 0 <= detection_threshold < 1:
            raise ValueError("invalid history policy parameters")
        self.mode = mode
        self.readout = readout
        self.cast_angle = cast_angle
        self.detection_threshold = detection_threshold
        self.history: deque[tuple[float, float]] = deque(maxlen=4)
        self.cast_side = 1 if seed % 2 == 0 else -1
        self.lost_decisions = 0

    def __call__(self, _tag_bearing: float,
                 sample: dict[str, float]) -> tuple[float, dict]:
        left, right = sample["L"], sample["R"]
        if self.mode == "wind_only":
            sensed_left = sensed_right = 0.0
        elif self.mode == "raw_odor":
            sensed_left, sensed_right = left, right
        else:
            if self.mode == "neural_swapped":
                left, right = right, left
            sensed_left, sensed_right = self.readout.decode_pair(left, right)
        self.history.append((sensed_left, sensed_right))
        recent = list(self.history)
        current_total = sensed_left + sensed_right
        if len(recent) >= 4:
            previous_pair = np.mean(recent[-4:-2], axis=0)
            current_pair = np.mean(recent[-2:], axis=0)
            left_change, right_change = current_pair - previous_pair
            previous_total = float(np.sum(previous_pair))
            filtered_total = float(np.sum(current_pair))
        elif len(recent) > 1:
            previous_left, previous_right = recent[-2]
            previous_total = previous_left + previous_right
            left_change = sensed_left - previous_left
            right_change = sensed_right - previous_right
            filtered_total = current_total
        else:
            previous_total = current_total
            left_change = right_change = 0.0
            filtered_total = current_total
        temporal_change = filtered_total - previous_total
        detected = current_total >= self.detection_threshold
        if detected and temporal_change > 0.012:
            self.lost_decisions = 0
            offset = 0.0
            phase = "surge"
        else:
            self.lost_decisions += 1
            lateral_change = left_change - right_change
            if detected and abs(lateral_change) > 0.008:
                self.cast_side = int(np.sign(lateral_change))
            elif self.lost_decisions % 5 == 0:
                self.cast_side *= -1
            offset = self.cast_side * self.cast_angle
            phase = "cast" if detected else "search"
        desired_bearing = wrap_angle(sample["upwind_bearing"] + offset)
        command = float(np.clip(0.6 * desired_bearing, -1.0, 1.0))
        details = {"mode": self.mode, "phase": phase,
                   "sensed_L": sensed_left, "sensed_R": sensed_right,
                   "filtered_total": filtered_total,
                   "temporal_change": temporal_change,
                   "lateral_temporal_change": left_change - right_change,
                   "cast_side": self.cast_side, "lost_decisions": self.lost_decisions,
                   "history_LR": [list(pair) for pair in self.history],
                   "upwind_bearing": sample["upwind_bearing"],
                   "desired_bearing": desired_bearing, "cast_angle": self.cast_angle}
        if self.mode.startswith("neural"):
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
