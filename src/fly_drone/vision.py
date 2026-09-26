"""Gate bearing measured from the same MuJoCo FPV camera used for videos."""

from __future__ import annotations

import hashlib
from math import atan, radians, tan

import mujoco
import numpy as np

from .fpv import FPVQuad


class GateVision:
    width = 320
    height = 180
    vertical_fov_degrees = 95

    def __init__(self, side: str, *, scenario: str = "nominal", seed: int = 0,
                 include_frame: bool = False, detect: bool = True):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        if scenario not in ("nominal", "dropout_30"):
            raise ValueError("scenario must be nominal or dropout_30")
        self.side = side
        self.scenario = scenario
        self.rng = np.random.default_rng(seed)
        self.include_frame = include_frame
        self.detect = detect
        self.last_angle = 0.0
        self.renderer = None

    def __enter__(self):
        return self

    def __exit__(self, *_exc):
        if self.renderer is not None:
            self.renderer.close()

    def observe(self, quad: FPVQuad, _target: tuple[float, float, float], _index: int) -> dict:
        if self.renderer is None:
            self.renderer = mujoco.Renderer(quad.model, width=self.width, height=self.height)
        self.renderer.update_scene(quad.data, camera="fpv")
        image = self.renderer.render()
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        frame = image.copy() if self.include_frame else None
        if self.scenario == "dropout_30" and self.rng.random() < 0.3:
            return {"bearing": self.last_angle, "visible": False, "reason": "dropout",
                    "frame_sha256": digest, **({"frame": None} if self.include_frame else {})}
        if not self.detect:
            return {"bearing": self.last_angle, "visible": True, "reason": "raw_frame",
                    "frame_sha256": digest, **({"frame": frame} if self.include_frame else {})}
        pixels = image.astype(np.float32)
        red, green, blue = pixels[:, :, 0], pixels[:, :, 1], pixels[:, :, 2]
        if self.side == "left":
            mask = (green > 100) & (green > 1.5 * red) & (green > 1.1 * blue)
        else:
            mask = (red > 140) & (red > 1.35 * green) & (green > 35)
        rows, columns = np.nonzero(mask)
        if len(columns) < 25:
            return {"bearing": self.last_angle, "visible": False, "reason": "no_gate",
                    "frame_sha256": digest, **({"frame": frame} if self.include_frame else {})}
        x0, x1 = int(columns.min()), int(columns.max())
        y0, y1 = int(rows.min()), int(rows.max())
        if y1 - y0 >= 0.83 * self.height:
            # At the gate, one post may fill the image and give a false centre.
            return {"bearing": self.last_angle, "visible": False, "reason": "gate_too_close",
                    "frame_sha256": digest, **({"frame": frame} if self.include_frame else {})}
        centre_x = (x0 + x1) / 2
        tangent = (centre_x / (self.width / 2) - 1) * tan(radians(self.vertical_fov_degrees) / 2)
        self.last_angle = -atan(tangent * self.width / self.height)
        return {"bearing": self.last_angle, "visible": True, "reason": "detected",
                "bbox": [x0, y0, x1, y1], "pixels": len(columns), "frame_sha256": digest,
                **({"frame": frame} if self.include_frame else {})}
