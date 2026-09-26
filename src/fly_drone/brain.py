"""Task adapter around the published, frozen MaleCNS spiking connectome."""

from __future__ import annotations

from math import pi

import numpy as np
from flybrain import FlyBrain
from flybrain.reservoir import Readout, Trace

from .perception import VisualFeatures
from .retina import RetinaProjector


class FlySteering:
    """Encode target bearing into LC10a and decode descending neuron activity.

    Only the linear readout is fitted. Connectome weights never change.
    """

    def __init__(self, *, seed: int = 7, ticks_per_action: int = 20,
                 network: str = "real", wiring_seed: int = 2026):
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        if network == "degree_shuffled":
            # Every source keeps its number of outgoing edges and edge weights;
            # shuffling their destinations also preserves each target's in-degree.
            # The pairing of source and target neurons is destroyed.
            shuffled = self.brain.indices.copy()
            np.random.default_rng(wiring_seed).shuffle(shuffled)
            self.brain.indices = shuffled
        elif network != "real":
            raise ValueError("network must be 'real' or 'degree_shuffled'")
        self.network = network
        self.wiring_seed = wiring_seed if network == "degree_shuffled" else None
        self.left = self.brain.cells(["LC10a"], side="L")
        self.right = self.brain.cells(["LC10a"], side="R")
        if not len(self.left) or not len(self.right):
            raise RuntimeError("MaleCNS LC10a cells are missing from the loaded data")
        self.motor_left = self.brain.cells(["DNa02"], side="L")
        self.motor_right = self.brain.cells(["DNa02"], side="R")
        if not len(self.motor_left) or not len(self.motor_right):
            raise RuntimeError("MaleCNS DNa02 steering cells are missing from the loaded data")
        self.trace = Trace(self.brain, idx=np.concatenate((self.motor_left, self.motor_right)), tau=0.4)
        self.seed = seed
        self.ticks_per_action = ticks_per_action
        self.readout: Readout | None = None
        self.output_scale = 1.0
        self.training_records: list[dict] = []
        self.validation_records: list[dict] = []
        self.sensory_encoder = None
        self.retina_projector = None
        self.retina_motion_labels = None
        self.last_visual_input: dict[str, float | str] = {}
        self.odor_cells = None
        self.odor_pn_masks = None
        self.last_odor_input: dict[str, float | int] = {}

    def reset(self, seed: int | None = None) -> None:
        self.brain.reset(self.seed if seed is None else seed)
        self.trace.reset()
        self.last_visual_input = {}
        self.last_odor_input = {}

    def encode(self, bearing: float) -> list[tuple[np.ndarray, float]]:
        # Left/right LC10a cells are visual target detectors. Bearing is supplied
        # by the simulator; this is a synthetic sensory interface, not an eye.
        if abs(bearing) < 0.08:
            return []
        amount = 0.55 + 0.55 * min(abs(bearing) / (pi / 2), 1.0)
        return [(self.left if bearing > 0 else self.right, amount)]

    def observe(self, bearing: float, *, visual_features: VisualFeatures | None = None,
                visual_mode: str = "none",
                retina_frames: list[np.ndarray] | None = None,
                odor: dict[str, float] | None = None) -> np.ndarray:
        injection = ([] if visual_mode in ("retina_only", "odor_only")
                     else self.encode(bearing))
        if visual_mode not in ("none", "flow", "flow_depth", "retina",
                               "retina_only", "odor_only"):
            raise ValueError("invalid visual_mode")
        if visual_mode == "odor_only" and odor is None:
            raise ValueError("odor_only requires bilateral odor input")
        self.last_visual_input = {}
        self.last_odor_input = {}
        if odor is not None:
            if set(odor) != {"L", "R"} or any(
                    not np.isfinite(value) or not 0 <= value <= 1
                    for value in odor.values()):
                raise ValueError("odor must contain finite L/R concentrations in [0, 1]")
            if self.odor_cells is None:
                self.odor_cells = {side: np.concatenate((
                    self.brain.cells(["ORN_DM1"], side=side),
                    self.brain.cells(["ORN_VA2"], side=side))) for side in "LR"}
                if any(not len(cells) for cells in self.odor_cells.values()):
                    raise RuntimeError("MaleCNS ORN_DM1 or ORN_VA2 cells missing")
                self.odor_pn_masks = {}
                for label, cell_type in (("DM1", "DM1_lPN"),
                                         ("VA2", "VA2_adPN")):
                    mask = np.zeros(self.brain.n, dtype=bool)
                    mask[self.brain.cells([cell_type])] = True
                    self.odor_pn_masks[label] = mask
            for side in "LR":
                amount = 0.25 * float(odor[side])
                if amount:
                    injection.append((self.odor_cells[side], amount))
                self.last_odor_input[f"{side}_concentration"] = float(odor[side])
                self.last_odor_input[f"{side}_drive"] = amount
                self.last_odor_input[f"{side}_cells"] = len(self.odor_cells[side])
        if visual_mode in ("retina", "retina_only"):
            if not retina_frames:
                raise ValueError("retina mode requires real FPV frames")
            if self.retina_projector is None:
                self.retina_projector = RetinaProjector(self.brain.azimuth)
                labels = np.full(self.brain.n, -1, dtype=np.int8)
                for side_index, side in enumerate("LR"):
                    for direction_index, suffix in enumerate("abcd"):
                        label = 4 * side_index + direction_index
                        for prefix in ("T4", "T5"):
                            labels[self.brain.cells([prefix + suffix], side=side)] = label
                self.retina_motion_labels = labels
            drives = [self.retina_projector.encode(frame) for frame in retina_frames]
            motion_counts = np.zeros(8, dtype=np.int64)
            for tick in range(self.ticks_per_action):
                eye = drives[min(tick * len(drives) // self.ticks_per_action,
                                 len(drives) - 1)]
                fired = self.brain.step(inject=injection, eye_drive=eye)
                self.trace.observe(fired)
                labels = self.retina_motion_labels[fired]
                motion_counts += np.bincount(labels[labels >= 0], minlength=8)
            self.last_visual_input = {
                "mode": visual_mode, "frames": len(drives),
                "photoreceptors": len(self.brain.visual),
                "eye_mean": float(np.mean([drive.mean() for drive in drives])),
                "eye_temporal_change": float(np.mean([
                    np.mean(np.abs(right - left))
                    for left, right in zip(drives, drives[1:])])) if len(drives) > 1 else 0.0,
                **{f"T4_T5_{side}_{suffix}_spikes": int(motion_counts[4 * side_index + i])
                   for side_index, side in enumerate("LR")
                   for i, suffix in enumerate("abcd")}}
            return self.trace.features()
        if visual_mode in ("flow", "flow_depth"):
            if visual_features is None:
                raise ValueError("visual_features required for neural vision")
            if self.sensory_encoder is None:
                from .sensory_adapter import FlySensoryEncoder
                self.sensory_encoder = FlySensoryEncoder(self.brain)
            if visual_mode == "flow":
                visual_features = VisualFeatures(visual_features.flow_xy_px,
                                                 np.ones_like(visual_features.inverse_depth),
                                                 visual_features.inference_ms,
                                                 visual_features.depth_model,
                                                 visual_features.flow_model)
            encoded = self.sensory_encoder.vision(visual_features)
            injection.extend(encoded.inject)
            self.last_visual_input = {"mode": visual_mode, **encoded.channels}
        pn_counts = ({label: 0 for label in self.odor_pn_masks}
                     if odor is not None else {})
        for _ in range(self.ticks_per_action):
            fired = self.brain.step(inject=injection)
            self.trace.observe(fired)
            for label in pn_counts:
                pn_counts[label] += int(np.count_nonzero(
                    self.odor_pn_masks[label][fired]))
        self.last_odor_input.update({f"{label}_PN_spikes": count
                                     for label, count in pn_counts.items()})
        return self.trace.features()

    def collect(self, count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reset(seed)
        rng = np.random.default_rng(seed)
        activity, labels, bearings = [], [], []
        for _ in range(count):
            bearing = 0.0 if rng.random() < 0.16 else float(rng.uniform(-pi / 2, pi / 2))
            bearings.append(bearing)
            activity.append(self.observe(bearing))
            labels.append(0.0 if abs(bearing) < 0.08 else float(np.sign(bearing)))
        return (np.asarray(activity, dtype=np.float32), np.asarray(labels, dtype=np.float32),
                np.asarray(bearings, dtype=float))

    @staticmethod
    def records(bearings: np.ndarray, labels: np.ndarray, features: np.ndarray,
                predictions: np.ndarray) -> list[dict]:
        return [{"bearing_rad": float(angle), "label": float(label),
                 "DNa02_left": float(feature[0]), "DNa02_right": float(feature[1]),
                 "prediction": float(prediction)}
                for angle, label, feature, prediction in zip(bearings, labels, features, predictions)]

    def fit(self, count: int = 96) -> dict:
        features, labels, bearings = self.collect(count, seed=self.seed)
        self.readout = Readout.fit(features, labels, kind="ridge",
                                   components=(2,), lambdas=(0.01, 0.1, 1.0))
        raw = np.asarray(self.readout.predict(features), dtype=float)
        self.training_records = self.records(bearings, labels, features, raw)
        self.output_scale = max(float(np.quantile(np.abs(raw), 0.9)), 0.1)
        return {"samples": count, "cv_negative_mse": self.readout.cv_score,
                "components": self.readout.components, "regularization": self.readout.lam,
                "output_scale": self.output_scale}

    def validate(self, count: int = 32) -> dict:
        if self.readout is None:
            raise RuntimeError("Call fit() before validate()")
        features, labels, bearings = self.collect(count, seed=self.seed + 1)
        predictions = np.asarray(self.readout.predict(features), dtype=float)
        self.validation_records = self.records(bearings, labels, features, predictions)
        active = labels != 0
        return {"samples": count,
                "mse": float(np.mean((predictions - labels) ** 2)),
                "direction_samples": int(np.sum(active)),
                "direction_correct": int(np.sum(np.sign(predictions[active]) == labels[active])),
                "direction_accuracy": float(np.mean(np.sign(predictions[active]) == labels[active]))
                if np.any(active) else None,
                "neutral_mean_abs_output": float(np.mean(np.abs(predictions[~active])))
                if np.any(~active) else None}

    def action(self, bearing: float, *, visual_features: VisualFeatures | None = None,
               visual_mode: str = "none",
               retina_frames: list[np.ndarray] | None = None,
               odor: dict[str, float] | None = None) -> float:
        if self.readout is None:
            raise RuntimeError("Call fit() before action()")
        features = self.observe(bearing, visual_features=visual_features,
                                visual_mode=visual_mode, retina_frames=retina_frames,
                                odor=odor)
        direction = float(self.readout.predict(features)) / self.output_scale
        # Target distance to the centre sets gain; the network chooses the sign.
        gain = (1.0 if visual_mode == "odor_only"
                else min(abs(bearing) / 0.7, 1.0))
        return float(np.clip(direction, -1.0, 1.0) * gain)
