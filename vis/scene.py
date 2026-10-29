from dataclasses import dataclass


@dataclass(frozen=True)
class Physics:
    position:         tuple[float, float, float]
    forward:          tuple[float, float, float] = (1.0, 0.0, 0.0)
    up:               tuple[float, float, float] = (0.0, 0.0, 1.0)
    velocity:         tuple[float, float, float] = (0.0, 0.0, 0.0)
    angular_velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Car:
    id:        int
    team:      int
    physics:   Physics
    boost:     float = 100.0
    on_ground: bool = True
    demoed:    bool = False
