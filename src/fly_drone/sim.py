"""A deliberately small, deterministic planar drone and waypoint task."""

from dataclasses import dataclass
from math import atan2, cos, hypot, pi, sin


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


@dataclass
class Drone:
    x: float = 0.0
    y: float = 0.0
    yaw: float = 0.0
    speed: float = 1.0
    max_yaw_rate: float = 1.8

    def bearing_to(self, target: tuple[float, float]) -> float:
        return wrap_angle(atan2(target[1] - self.y, target[0] - self.x) - self.yaw)

    def distance_to(self, target: tuple[float, float]) -> float:
        return hypot(target[0] - self.x, target[1] - self.y)

    def step(self, yaw_command: float, dt: float) -> None:
        command = max(-1.0, min(1.0, yaw_command))
        self.yaw = wrap_angle(self.yaw + command * self.max_yaw_rate * dt)
        self.x += self.speed * cos(self.yaw) * dt
        self.y += self.speed * sin(self.yaw) * dt


def fly_path(controller, target: tuple[float, float], *, steps: int = 110,
             dt: float = 0.2, goal_radius: float = 0.5) -> dict:
    drone = Drone()
    trace = []
    for index in range(steps):
        bearing = drone.bearing_to(target)
        distance = drone.distance_to(target)
        if distance <= goal_radius:
            break
        command = float(controller(bearing))
        drone.step(command, dt)
        trace.append({"t": round((index + 1) * dt, 3), "x": drone.x,
                      "y": drone.y, "yaw": drone.yaw, "bearing": bearing,
                      "command": command, "distance": drone.distance_to(target)})
    return {"target": list(target), "reached": drone.distance_to(target) <= goal_radius,
            "final_distance": drone.distance_to(target), "steps": len(trace), "trace": trace}
