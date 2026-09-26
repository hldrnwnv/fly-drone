"""Learn both yaw and speed readouts from a frozen MaleCNS reservoir."""

from __future__ import annotations

from math import pi

import numpy as np
from flybrain import FlyBrain
from flybrain.reservoir import Readout, Trace


def teacher_commands(bearing: float, range_m: float) -> np.ndarray:
    """Supervised pursuit targets used for fitting, never at flight inference."""
    yaw = float(np.clip(bearing / 0.7, -1.0, 1.0))
    speed = float(np.clip(0.65 + 0.75 * (range_m - 1.7), 0.0, 1.5))
    return np.array([yaw, speed], dtype=np.float32)


class FlyPursuit:
    """Inject target direction and apparent range into LC10a; decode two commands."""

    def __init__(self, *, seed: int = 7, ticks_per_action: int = 20):
        self.brain = FlyBrain(seed=seed, device="cpu", sensory_input=False)
        self.left = self.brain.cells(["LC10a"], side="L")
        self.right = self.brain.cells(["LC10a"], side="R")
        self.size_cells = self.brain.cells(["LC11"])
        self.descending = self.brain.cells("descending_neuron")
        self.trace = Trace(self.brain, idx=self.descending, tau=0.4)
        steering_cells = np.concatenate((self.brain.cells(["DNa02"], side="L"),
                                         self.brain.cells(["DNa02"], side="R")))
        self.steering_positions = np.flatnonzero(np.isin(self.descending, steering_cells))
        if len(self.steering_positions) != 2:
            raise RuntimeError("Expected two DNa02 cells in the descending readout")
        self.seed = seed
        self.ticks_per_action = ticks_per_action
        self.yaw_readout: Readout | None = None
        self.speed_readout: Readout | None = None
        self.training_records: list[dict] = []
        self.validation_records: list[dict] = []

    def reset(self, seed: int | None = None) -> None:
        self.brain.reset(self.seed if seed is None else seed)
        self.trace.reset()

    def encode(self, bearing: float, range_m: float) -> list[tuple[np.ndarray, float]]:
        # A synthetic sensory interface: the camera estimates target bearing
        # and size; this is not a reconstruction of the fly's retina.
        proximity = float(np.clip((4.0 - range_m) / 3.0, 0.0, 1.0))
        injection = [(self.size_cells, 0.20 + 0.70 * proximity)]
        if abs(bearing) >= 0.08:
            amount = 0.55 + 0.55 * min(abs(bearing) / (pi / 2), 1.0)
            injection.append((self.left if bearing > 0 else self.right, amount))
        return injection

    def observe(self, bearing: float, range_m: float) -> np.ndarray:
        injection = self.encode(bearing, range_m)
        for _ in range(self.ticks_per_action):
            self.trace.observe(self.brain.step(inject=injection))
        return self.trace.features()

    def collect(self, count: int, *, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        self.reset(seed)
        rng = np.random.default_rng(seed)
        features, labels, inputs = [], [], []
        for _ in range(count):
            bearing = float(rng.uniform(-1.2, 1.2))
            range_m = float(rng.uniform(0.9, 4.0))
            features.append(self.observe(bearing, range_m))
            labels.append(teacher_commands(bearing, range_m))
            inputs.append([bearing, range_m])
        return (np.asarray(features, dtype=np.float32),
                np.asarray(labels, dtype=np.float32), np.asarray(inputs, dtype=np.float32))

    @staticmethod
    def records(inputs: np.ndarray, labels: np.ndarray, predictions: np.ndarray) -> list[dict]:
        return [{"bearing_rad": float(inp[0]), "range_m": float(inp[1]),
                 "target_yaw": float(label[0]),
                 "target_direction": float(np.sign(inp[0])) if abs(inp[0]) >= 0.08 else 0.0,
                 "target_speed_m_s": float(label[1]),
                 "predicted_direction_score": float(prediction[0]),
                 "predicted_speed_m_s": float(prediction[1])}
                for inp, label, prediction in zip(inputs, labels, predictions)]

    def fit(self, count: int = 192) -> dict:
        features, labels, inputs = self.collect(count, seed=self.seed)
        yaw_labels = np.where(np.abs(inputs[:, 0]) < 0.08, 0,
                              np.sign(inputs[:, 0])).astype(np.float32)
        self.yaw_readout = Readout.fit(features[:, self.steering_positions], yaw_labels,
                                       kind="ridge", components=(2,),
                                       lambdas=(0.01, 0.1, 1.0))
        self.speed_readout = Readout.fit(features, labels[:, 1], kind="ridge",
                                         components=(5, 20, 60), lambdas=(0.01, 0.1, 1.0))
        yaw_predictions = np.asarray(self.yaw_readout.predict(features[:, self.steering_positions]),
                                     dtype=float)
        speed_predictions = np.asarray(self.speed_readout.predict(features), dtype=float)
        predictions = np.column_stack((yaw_predictions, speed_predictions))
        self.training_records = self.records(inputs, labels, predictions)
        return {"samples": count,
                "yaw_cv_negative_mse": self.yaw_readout.cv_score,
                "speed_cv_negative_mse": self.speed_readout.cv_score,
                "speed_components": self.speed_readout.components,
                "descending_features": len(self.descending),
                "size_input_cells": len(self.size_cells)}

    def validate(self, count: int = 64) -> dict:
        if self.yaw_readout is None or self.speed_readout is None:
            raise RuntimeError("Call fit() before validate()")
        features, labels, inputs = self.collect(count, seed=self.seed + 1)
        yaw_labels = np.where(np.abs(inputs[:, 0]) < 0.08, 0,
                              np.sign(inputs[:, 0])).astype(np.float32)
        yaw_predictions = np.asarray(self.yaw_readout.predict(features[:, self.steering_positions]),
                                     dtype=float)
        speed_predictions = np.asarray(self.speed_readout.predict(features), dtype=float)
        predictions = np.column_stack((yaw_predictions, speed_predictions))
        self.validation_records = self.records(inputs, labels, predictions)
        self.reset(self.seed + 1)
        hidden_range_features = np.asarray([self.observe(float(bearing), 1.7)
                                            for bearing in inputs[:, 0]], dtype=np.float32)
        hidden_range_speed = np.asarray(self.speed_readout.predict(hidden_range_features), dtype=float)
        return {"samples": count,
                "yaw_direction_accuracy": float(np.mean(np.sign(yaw_predictions) == yaw_labels)),
                "speed_mae_m_s": float(np.mean(np.abs(speed_predictions - labels[:, 1]))),
                "speed_no_range_mae_m_s": float(np.mean(np.abs(hidden_range_speed - labels[:, 1]))),
                "speed_constant_mae_m_s": float(np.mean(np.abs(labels[:, 1] - labels[:, 1].mean()))),
                "speed_r2": float(1 - np.sum((speed_predictions - labels[:, 1]) ** 2) /
                                  np.sum((labels[:, 1] - labels[:, 1].mean()) ** 2))}

    def action(self, bearing: float, range_m: float) -> tuple[float, float]:
        if self.yaw_readout is None or self.speed_readout is None:
            raise RuntimeError("Call fit() before action()")
        features = self.observe(bearing, range_m)
        raw_yaw = float(self.yaw_readout.predict(features[self.steering_positions]))
        yaw = float(np.clip(raw_yaw, -1.0, 1.0)) * min(abs(bearing) / 0.7, 1.0)
        speed = float(self.speed_readout.predict(features))
        return yaw, float(np.clip(speed, 0.0, 1.5))
