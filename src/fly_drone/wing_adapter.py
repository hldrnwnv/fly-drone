"""Wing-averaged intent to quad wrench to four rotor thrusts.

The fly-to-quad torque coefficients are hypotheses. The quad mixer and units
are physical: total force is N; roll, pitch, and yaw moments are N m.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class WingIntent:
    power_ratio: float
    amplitude_left: float
    amplitude_right: float
    pitch_left: float
    pitch_right: float


@dataclass(frozen=True)
class RotorCommand:
    thrusts_n: list[float]
    desired_wrench: list[float]
    achieved_wrench: list[float]


class WingRotorAdapter:
    """Use wing kinematic differences as body moments, then mix four rotors."""

    def __init__(self, *, mass_kg: float, mixer: np.ndarray,
                 collective_fraction: float = 0.05,
                 roll_gain_nm: float = 0.04,
                 yaw_gain_nm: float = 0.03):
        self.mass_kg = mass_kg
        self.mixer = np.asarray(mixer, dtype=float)
        if self.mixer.shape != (4, 4) or not np.isfinite(self.mixer).all():
            raise ValueError("mixer must be a finite 4x4 matrix")
        if mass_kg <= 0:
            raise ValueError("mass_kg must be positive")
        self.collective_fraction = collective_fraction
        self.roll_gain_nm = roll_gain_nm
        self.yaw_gain_nm = yaw_gain_nm

    def mix(self, intent: WingIntent) -> RotorCommand:
        # Indirect flight muscle spikes modulate a continuously beating virtual
        # wing. They are not interpreted as one spike = one wing stroke.
        collective = self.mass_kg * 9.81 * (1 + self.collective_fraction *
                                             np.tanh(intent.power_ratio - 1))
        # Hypothesis: wing pitch asymmetry gives roll; stroke-amplitude
        # asymmetry gives yaw. No pitch command is inferred from these five
        # outputs. The pilot may later add a distinct pitch motor channel.
        roll = self.roll_gain_nm * np.tanh(intent.pitch_left - intent.pitch_right)
        yaw = self.yaw_gain_nm * np.tanh(intent.amplitude_left - intent.amplitude_right)
        desired = np.array([collective, roll, 0.0, yaw], dtype=float)
        thrusts = np.clip(np.linalg.solve(self.mixer, desired), 0.0, 8.0)
        achieved = self.mixer @ thrusts
        return RotorCommand(thrusts.tolist(), desired.tolist(), achieved.tolist())
