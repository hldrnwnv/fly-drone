"""RGB-only MaleCNS readout for the paired FPV experiment.

The source column map is a research projection, not calibrated fly optics.
The controller never receives the detector bearing or simulator state.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
from flybrain import FlyBrain

from .retina_2d import RetinaProjector2D


class RawFrameReadout:
    """Frozen graph, 2D photoreceptor drive and a fitted linear readout."""

    def __init__(self, map_path: Path, *, network: str = "real", wiring_seed: int = 2026,
                 ticks_per_frame: int = 20, seed: int = 7):
        if ticks_per_frame < 1:
            raise ValueError("ticks_per_frame must be positive")
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        if network == "degree_shuffled":
            shuffled = self.brain.indices.copy()
            np.random.default_rng(wiring_seed).shuffle(shuffled)
            self.brain.indices = shuffled
        elif network != "real":
            raise ValueError("invalid network")
        self.adjacency_sha256 = hashlib.sha256(self.brain.indices.tobytes()).hexdigest()
        with np.load(map_path) as saved:
            if not np.array_equal(saved["visual"], self.brain.visual):
                raise ValueError("retina map does not match the loaded brain")
            self.projector = RetinaProjector2D(saved["xy"], saved["side"])
        self.group_names = [f"{kind}_{side}" for side in "LR" for kind in
                            ("T4a", "T4b", "T4c", "T4d", "T5a", "T5b", "T5c", "T5d", "DNa02")]
        self.group = np.full(self.brain.n, -1, dtype=np.int16)
        for i, name in enumerate(self.group_names):
            kind, side = name.split("_")
            self.group[self.brain.cells([kind], side=side)] = i
        self.ticks_per_frame = ticks_per_frame
        self.seed = seed
        self.weights: np.ndarray | None = None
        self.output_scale = 1.0
        self.last_command = 0.0

    def reset(self, seed: int | None = None) -> None:
        self.brain.reset(self.seed if seed is None else seed)
        self.last_command = 0.0

    def features(self, frame: np.ndarray) -> np.ndarray:
        drive = self.projector.encode(frame)
        counts = np.zeros(len(self.group_names), dtype=np.float32)
        for _ in range(self.ticks_per_frame):
            fired = self.brain.step(eye_drive=drive)
            labels = self.group[fired]
            counts += np.bincount(labels[labels >= 0], minlength=len(counts))
        return counts

    def collect(self, frames: np.ndarray, *, first_seed: int) -> np.ndarray:
        features = []
        for i, frame in enumerate(frames):
            self.reset(first_seed + i)
            features.append(self.features(frame))
        return np.stack(features)

    def fit(self, train_x: np.ndarray, train_y: np.ndarray,
            validation_x: np.ndarray, validation_y: np.ndarray) -> dict:
        if train_x.shape[1] != len(self.group_names) or validation_x.shape[1] != len(self.group_names):
            raise ValueError("feature width does not match cell groups")
        mean = train_x.mean(axis=0)
        std = np.maximum(train_x.std(axis=0), 1.0)
        train = np.column_stack(((train_x - mean) / std, np.ones(len(train_x))))
        validation = np.column_stack(((validation_x - mean) / std,
                                      np.ones(len(validation_x))))
        candidates = []
        for lam in (0.1, 1.0, 10.0, 100.0):
            penalty = np.eye(train.shape[1]) * lam
            penalty[-1, -1] = 0
            beta = np.linalg.solve(train.T @ train + penalty, train.T @ train_y)
            mse = float(np.mean((validation @ beta - validation_y) ** 2))
            candidates.append((mse, lam, beta))
        mse, lam, beta = min(candidates, key=lambda item: item[0])
        self.weights = np.concatenate((mean, std, beta)).astype(np.float32)
        predictions = train @ beta
        self.output_scale = max(float(np.quantile(np.abs(predictions), 0.9)), 0.1)
        held = validation @ beta
        active = np.abs(validation_y) >= 0.08
        return {"regularization": lam, "validation_mse": mse,
                "validation_direction_correct": int(np.sum(np.sign(held[active]) ==
                                                         np.sign(validation_y[active]))),
                "validation_direction_total": int(active.sum()),
                "output_scale": self.output_scale,
                "train_predictions": predictions.tolist(),
                "validation_predictions": held.tolist()}

    def predict(self, feature: np.ndarray) -> float:
        if self.weights is None:
            raise RuntimeError("readout has not been fitted")
        n = len(self.group_names)
        mean, std, beta = self.weights[:n], self.weights[n:2 * n], self.weights[2 * n:]
        return float(np.r_[(feature - mean) / std, 1.0] @ beta)

    def action(self, frame: np.ndarray | None) -> float:
        if frame is None:
            return self.last_command
        prediction = self.predict(self.features(frame))
        self.last_command = float(np.clip(prediction / self.output_scale, -1, 1))
        return self.last_command

    def save(self, path: Path) -> None:
        if self.weights is None:
            raise RuntimeError("readout has not been fitted")
        np.savez(path, weights=self.weights, output_scale=self.output_scale,
                 group_names=np.asarray(self.group_names))
