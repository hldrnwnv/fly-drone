"""Explicit, testable engineering encoders into annotated MaleCNS cell pools."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .perception import VisualFeatures


@dataclass(frozen=True)
class EncodedSensoryInput:
    inject: list[tuple[np.ndarray, float]]
    channels: dict[str, float]


class FlySensoryEncoder:
    """Map virtual gyro and learned visual estimates to selected cell groups.

    SApp08 thirds encode roll, pitch, yaw with sign split across sides. T4/T5
    pools encode four image-motion directions; LPLC2 gets depth-weighted
    expansion; LC15 gets a near/far relative-motion proxy. These mappings and
    gains are hypotheses, not measured tuning.
    """

    def __init__(self, brain):
        self.gyro_cells = {}
        for side in "LR":
            thirds = np.array_split(brain.cells(["SApp08"], side=side), 3)
            for axis, cells in zip(("roll", "pitch", "yaw"), thirds):
                if not len(cells):
                    raise RuntimeError(f"Missing SApp08 {axis} cells on side {side}")
                self.gyro_cells[axis, side] = cells
        self.motion_cells = {(name, side): brain.cells([name], side=side)[:64]
                             for name in ("T4a", "T4b", "T4c", "T4d",
                                          "T5a", "T5b", "T5c", "T5d")
                             for side in "LR"}
        self.loom_cells = {side: brain.cells(["LPLC2"], side=side)[:64]
                           for side in "LR"}
        self.distance_cells = {side: brain.cells(["LC15"], side=side)[:64]
                               for side in "LR"}
        if any(not len(c) for c in (*self.motion_cells.values(),
                                    *self.loom_cells.values(),
                                    *self.distance_cells.values())):
            raise RuntimeError("Required T4/T5, LPLC2, or LC15 cell pool missing")

    def gyro(self, body_rates: np.ndarray) -> EncodedSensoryInput:
        rates = np.asarray(body_rates, dtype=float)
        if rates.shape != (3,) or not np.isfinite(rates).all():
            raise ValueError("body_rates must be finite roll/pitch/yaw rad/s")
        channels: dict[str, float] = {}
        inject = []
        for axis, rate in zip(("roll", "pitch", "yaw"), rates):
            # Sign convention is explicit and tunable; no haltere mechanics.
            side = "R" if rate > 0 else "L"
            amount = float(np.clip(abs(rate), 0, 2.0))
            channels[f"{axis}_rad_s"] = float(rate)
            channels[f"{axis}_{side}_drive"] = amount
            if amount:
                inject.append((self.gyro_cells[axis, side], amount))
        return EncodedSensoryInput(inject, channels)

    def vision(self, features: VisualFeatures) -> EncodedSensoryInput:
        flow = features.flow_xy_px
        if flow is None:
            return EncodedSensoryInput([], {"flow_available": 0.0,
                                            "depth_relative_mean": float(np.mean(
                                                features.inverse_depth))})
        if flow.ndim != 3 or flow.shape[2] != 2 or not np.isfinite(flow).all():
            raise ValueError("flow must be finite HxWx2 pixels per frame")
        depth = np.asarray(features.inverse_depth, dtype=float)
        if depth.ndim != 2 or not np.isfinite(depth).all():
            raise ValueError("inverse_depth must be a finite image")
        low, high = np.percentile(depth, [5, 95])
        normalized = np.clip((depth - low) / max(high - low, 1e-6), 0, 1)
        h, w = flow.shape[:2]
        ys = np.linspace(0, depth.shape[0] - 1, h).astype(int)
        xs = np.linspace(0, depth.shape[1] - 1, w).astype(int)
        proximity = normalized[np.ix_(ys, xs)]
        yy, xx = np.mgrid[:h, :w]
        expansion = ((xx - w / 2) / max(w / 2, 1) * flow[..., 0] +
                     (yy - h / 2) / max(h / 2, 1) * flow[..., 1])
        channels: dict[str, float] = {"flow_available": 1.0,
                                      "depth_relative_mean": float(depth.mean())}
        inject = []
        for side, area in (("L", slice(0, w // 2)), ("R", slice(w // 2, w))):
            fx = float(np.median(flow[:, area, 0]))
            fy = float(np.median(flow[:, area, 1]))
            directional = {"a": max(fx, 0), "b": max(-fx, 0),
                           "c": max(-fy, 0), "d": max(fy, 0)}
            for suffix, pixels in directional.items():
                amount = float(np.clip(pixels / 10.0, 0, 1.0))
                channels[f"flow_{side}_{suffix}_drive"] = amount
                if amount:
                    for prefix in ("T4", "T5"):
                        inject.append((self.motion_cells[prefix + suffix, side], amount))
            looming = float(np.percentile(np.maximum(expansion[:, area], 0) *
                                           proximity[:, area], 90))
            amount = float(np.clip(looming / 10.0, 0, 1.0))
            channels[f"loom_{side}_drive"] = amount
            if amount:
                inject.append((self.loom_cells[side], amount))
            speed = np.linalg.norm(flow[:, area], axis=-1)
            near = proximity[:, area] >= 0.75
            far = proximity[:, area] <= 0.25
            near_speed = float(np.median(speed[near])) if np.any(near) else 0.0
            far_speed = float(np.median(speed[far])) if np.any(far) else 0.0
            parallax_drive = float(np.clip((near_speed - far_speed) / 10.0, 0, 1))
            channels[f"parallax_{side}_near_px"] = near_speed
            channels[f"parallax_{side}_far_px"] = far_speed
            channels[f"parallax_{side}_drive"] = parallax_drive
            if parallax_drive:
                inject.append((self.distance_cells[side], parallax_drive))
        return EncodedSensoryInput(inject, channels)
