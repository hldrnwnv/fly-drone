"""Project FPV luminance onto MaleCNS photoreceptors with known azimuths."""

from __future__ import annotations

import numpy as np


class RetinaProjector:
    """One-dimensional eye approximation; MaleCNS has no elevation labels here."""

    def __init__(self, azimuth: np.ndarray):
        self.azimuth = np.asarray(azimuth, dtype=np.float32)
        if self.azimuth.ndim != 1 or not len(self.azimuth):
            raise ValueError("azimuth must contain photoreceptor positions")
        if not np.isfinite(self.azimuth).all():
            raise ValueError("azimuth must be finite")

    def encode(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be HxWx3 uint8 RGB")
        height, width = frame.shape[:2]
        if height < 2 or width < 2:
            raise ValueError("frame is too small")
        middle = frame[height // 4:3 * height // 4].astype(np.float32)
        luminance = (0.299 * middle[..., 0] + 0.587 * middle[..., 1] +
                     0.114 * middle[..., 2]).mean(axis=0) / 255.0
        x = (np.clip(self.azimuth, -1.0, 1.0) + 1.0) * (width - 1) / 2.0
        return np.interp(x, np.arange(width), luminance).astype(np.float32)
