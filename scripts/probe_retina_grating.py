"""Phase-matched moving-grating control for issue #1's bar asymmetry.

Both directions start with the same frame and contain the same 50 frames.
This tests order sensitivity without left/right starting-position differences.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from probe_retina_motion_2d import HEIGHT, N_FRAMES, WIDTH, run


def grating(phase: float, contrast: int) -> np.ndarray:
    x = np.arange(WIDTH, dtype=np.float32) / WIDTH
    luminance = np.rint(125 + contrast * np.sin(2 * np.pi * (4 * x - phase)))
    frame = np.full((HEIGHT, WIDTH, 3), 125, dtype=np.uint8)
    frame[HEIGHT // 4:3 * HEIGHT // 4] = luminance.astype(np.uint8)[None, :, None]
    return frame


def movies() -> dict[str, np.ndarray]:
    output = {}
    for label, contrast in (("high", 95), ("low", 40)):
        output[f"stationary_{label}"] = np.repeat(grating(0, contrast)[None], N_FRAMES, axis=0)
        for direction, sign in (("left_to_right", 1), ("right_to_left", -1)):
            output[f"{direction}_{label}"] = np.stack(
                [grating(sign * tick / N_FRAMES, contrast) for tick in range(N_FRAMES)])
        a = output[f"left_to_right_{label}"]
        b = output[f"right_to_left_{label}"]
        assert np.array_equal(a[0], b[0])
        assert sorted(image.tobytes() for image in a) == sorted(image.tobytes() for image in b)
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("runs/retina-2d/grating"))
    parser.add_argument("--map", type=Path, default=Path("runs/retina-2d/map.npz"))
    parser.add_argument("--data", type=Path, default=Path("data/flybrain"))
    parser.add_argument("--first-seed", type=int, default=19)
    parser.add_argument("--seeds", type=int, default=12)
    args = parser.parse_args()
    run(args.output, args.map, args.data, args.first_seed, args.seeds, movies())
