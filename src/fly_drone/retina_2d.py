"""Experimental projection onto source-located MaleCNS optic columns.

Coordinates are normalized *within each eye*. They are not calibrated camera
angles or photoreceptor optical axes. Unlocated receptors receive background.
"""

from __future__ import annotations

import numpy as np


def normalized_eye_coordinates(columns: np.ndarray, sides: np.ndarray) -> np.ndarray:
    """Convert source hex1/hex2 columns to a normalized per-eye image grid.

    The source's q=hex1 and p=hex2 axes point up-right and up-left. Thus
    horizontal is q-p and vertical is q+p. Flips relative to a physical FPV
    camera have not been calibrated and must be recorded by an experiment.
    """
    columns = np.asarray(columns)
    sides = np.asarray(sides)
    if columns.ndim != 2 or columns.shape[1] != 2 or sides.shape != (len(columns),):
        raise ValueError("columns must be Nx2 and sides must have N entries")
    xy = np.full((len(columns), 2), np.nan, dtype=np.float32)
    valid = np.isfinite(columns).all(axis=1)
    for side in ("L", "R"):
        selection = valid & (sides == side)
        if not selection.any():
            continue
        q, p = columns[selection].T
        raw = np.column_stack((q - p, -(q + p)))
        low, high = raw.min(axis=0), raw.max(axis=0)
        if np.any(high <= low):
            raise ValueError(f"eye {side} has a degenerate column map")
        xy[selection] = ((raw - low) / (high - low)).astype(np.float32)
    return xy


class RetinaProjector2D:
    """Sample a synthetic RGB stimulus at located column positions."""

    def __init__(self, xy: np.ndarray, sides: np.ndarray, *, background: float = 30 / 255):
        self.xy = np.asarray(xy, dtype=np.float32)
        self.sides = np.asarray(sides)
        if self.xy.ndim != 2 or self.xy.shape[1] != 2 or self.sides.shape != (len(self.xy),):
            raise ValueError("xy must be Nx2 and sides must have N entries")
        if not np.all(np.isfinite(self.xy) | np.isnan(self.xy)):
            raise ValueError("xy must be finite or NaN")
        valid = np.isfinite(self.xy).all(axis=1)
        if np.any(np.isfinite(self.xy).any(axis=1) != valid):
            raise ValueError("both coordinates must be present together")
        if np.any((self.xy[valid] < 0) | (self.xy[valid] > 1)):
            raise ValueError("xy must lie within [0, 1]")
        if not set(self.sides).issubset({"L", "R"}):
            raise ValueError("sides must be L or R")
        self.background = np.float32(background)
        self.valid = valid

    def encode(self, frame: np.ndarray, *, eye: str | None = None) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3 or frame.dtype != np.uint8:
            raise ValueError("frame must be HxWx3 uint8 RGB")
        height, width = frame.shape[:2]
        if height < 2 or width < 2:
            raise ValueError("frame is too small")
        if eye is not None and eye not in ("L", "R"):
            raise ValueError("eye must be L or R")
        drive = np.full(len(self.xy), self.background, dtype=np.float32)
        chosen = self.valid if eye is None else self.valid & (self.sides == eye)
        if not chosen.any():
            return drive
        rgb = frame.astype(np.float32)
        lum = (0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]) / 255.0
        x = self.xy[chosen, 0] * (width - 1)
        y = self.xy[chosen, 1] * (height - 1)
        x0 = np.floor(x).astype(int)
        y0 = np.floor(y).astype(int)
        x1 = np.minimum(x0 + 1, width - 1)
        y1 = np.minimum(y0 + 1, height - 1)
        dx, dy = x - x0, y - y0
        drive[chosen] = ((1 - dx) * (1 - dy) * lum[y0, x0] +
                         dx * (1 - dy) * lum[y0, x1] +
                         (1 - dx) * dy * lum[y1, x0] +
                         dx * dy * lum[y1, x1]).astype(np.float32)
        return drive
