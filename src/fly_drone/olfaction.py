"""Deterministic virtual food-odor plume for a controlled sensory pilot."""

from __future__ import annotations

from math import cos, exp, sin

import numpy as np


class FoodOdorPlume:
    """Sample two virtual antennae; the controller never receives source pose."""

    def __init__(self, *, wind_xy: tuple[float, float] = (-1.0, 0.0),
                 antenna_half_span_m: float = 0.12, phase_s: float = 0.0):
        wind = np.asarray(wind_xy, dtype=float)
        norm = float(np.linalg.norm(wind))
        if not np.isfinite(wind).all() or norm <= 0:
            raise ValueError("wind must be finite and nonzero")
        if antenna_half_span_m <= 0:
            raise ValueError("antenna_half_span_m must be positive")
        self.wind = wind / norm
        self.crosswind = np.array([-self.wind[1], self.wind[0]])
        self.antenna_half_span_m = antenna_half_span_m
        self.phase_s = phase_s

    def concentration(self, point_xy: np.ndarray, source_xy: np.ndarray,
                      time_s: float) -> float:
        offset = np.asarray(point_xy, dtype=float) - np.asarray(source_xy, dtype=float)
        downwind = float(offset @ self.wind)
        if downwind < 0:
            return 0.0
        cross = float(offset @ self.crosswind)
        width = 0.22 + 0.13 * downwind
        plume_time = time_s + self.phase_s
        center = 0.10 * sin(0.6 * plume_time + 0.8 * downwind)
        envelope = exp(-downwind / 4.0 - 0.5 * ((cross - center) / width) ** 2)
        intermittency = 0.70 + 0.30 * (0.5 + 0.5 * sin(
            4.0 * plume_time - 1.3 * downwind + 3.0 * cross))
        return float(np.clip(envelope * intermittency, 0.0, 1.0))

    def sample(self, position_xy: np.ndarray, yaw: float,
               source_xy: np.ndarray, time_s: float) -> dict[str, float]:
        position = np.asarray(position_xy, dtype=float)
        left_axis = np.array([-sin(yaw), cos(yaw)])
        left = position + self.antenna_half_span_m * left_axis
        right = position - self.antenna_half_span_m * left_axis
        return {"L": self.concentration(left, source_xy, time_s),
                "R": self.concentration(right, source_xy, time_s)}
