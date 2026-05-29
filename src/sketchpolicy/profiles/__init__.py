"""Robot workspace profiles.

A profile is a *kinematic envelope only* — it carries no URDF, no link meshes
and no inverse kinematics. It describes the volume an end-effector may occupy
and a coarse base keep-out cylinder, which is all the feasibility filter needs
to reject obviously-impossible variants. The defaults describe a generic
single-arm tabletop setup and are intentionally conservative.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RobotProfile:
    """A coarse kinematic envelope for feasibility checking.

    All lengths are in metres, in the robot base frame (base at ``base``).
    """

    name: str
    base: tuple[float, float, float] = (0.0, 0.0, 0.0)
    reach_min: float = 0.15  # inner radius of the reachable spherical shell
    reach_max: float = 0.85  # outer radius of the reachable spherical shell
    floor_z: float = 0.0  # the table / floor plane; EE must stay at or above
    workspace_box: tuple[float, float, float, float, float, float] = (
        -0.7,
        0.7,
        -0.7,
        0.7,
        0.0,
        1.0,
    )  # (x_min, x_max, y_min, y_max, z_min, z_max)
    base_keepout_radius: float = 0.12  # cylinder around the base z-axis
    base_keepout_height: float = 0.30
    gripper_open_width_m: float = 0.08  # nominal parallel-jaw aperture at gripper==1

    def base_array(self) -> np.ndarray:
        return np.asarray(self.base, dtype=np.float64)


#: Registry of built-in profiles.
_PROFILES: dict[str, RobotProfile] = {
    "generic_tabletop": RobotProfile(name="generic_tabletop"),
    # A smaller envelope, e.g. a compact desktop arm.
    "compact_desktop": RobotProfile(
        name="compact_desktop",
        reach_min=0.08,
        reach_max=0.45,
        workspace_box=(-0.4, 0.4, -0.4, 0.4, 0.0, 0.6),
        base_keepout_radius=0.06,
        base_keepout_height=0.15,
        gripper_open_width_m=0.05,
    ),
}


def get_profile(name: str) -> RobotProfile:
    """Return a built-in profile by name."""
    try:
        return _PROFILES[name]
    except KeyError as exc:
        raise KeyError(
            f"unknown profile {name!r}; available: {sorted(_PROFILES)}"
        ) from exc


def available_profiles() -> list[str]:
    return sorted(_PROFILES)
