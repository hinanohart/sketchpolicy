"""The reject-and-resample feasibility filter.

This is a *kinematic envelope* test, not a full collision check: we have no
robot mesh in the no-IK end-effector representation. The checks are therefore
named explicitly and the report says exactly which one failed:

* ``out_of_reach``  - a frame falls outside the reachable spherical shell;
* ``below_floor``   - a frame dips below the table/floor plane;
* ``outside_box``   - a frame leaves the workspace bounding box;
* ``base_keepout``  - a frame enters the base keep-out cylinder.

A variant is feasible iff none of these fire.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from sketchpolicy.eeplan import EEPlan
from sketchpolicy.profiles import RobotProfile

_TOL = 1e-6


@dataclass(frozen=True)
class FeasibilityReport:
    """Outcome of a feasibility check for one plan."""

    feasible: bool
    reasons: dict[str, bool] = field(default_factory=dict)

    def first_reason(self) -> str | None:
        for name, failed in self.reasons.items():
            if failed:
                return name
        return None


def check(plan: EEPlan, profile: RobotProfile) -> FeasibilityReport:
    """Return the feasibility report for ``plan`` under ``profile``."""
    pos = plan.positions
    base = profile.base_array()
    rel = pos - base
    dist = np.linalg.norm(rel, axis=1)

    out_of_reach = bool(
        np.any((dist < profile.reach_min - _TOL) | (dist > profile.reach_max + _TOL))
    )
    below_floor = bool(np.any(pos[:, 2] < profile.floor_z - _TOL))

    bx = profile.workspace_box
    outside_box = bool(
        np.any(
            (pos[:, 0] < bx[0] - _TOL)
            | (pos[:, 0] > bx[1] + _TOL)
            | (pos[:, 1] < bx[2] - _TOL)
            | (pos[:, 1] > bx[3] + _TOL)
            | (pos[:, 2] < bx[4] - _TOL)
            | (pos[:, 2] > bx[5] + _TOL)
        )
    )

    radial = np.linalg.norm(rel[:, :2], axis=1)
    base_keepout = bool(
        np.any(
            (radial < profile.base_keepout_radius - _TOL)
            & (rel[:, 2] < profile.base_keepout_height)
        )
    )

    reasons = {
        "out_of_reach": out_of_reach,
        "below_floor": below_floor,
        "outside_box": outside_box,
        "base_keepout": base_keepout,
    }
    return FeasibilityReport(feasible=not any(reasons.values()), reasons=reasons)
