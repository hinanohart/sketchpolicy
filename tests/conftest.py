"""Shared fixtures: deterministic synthetic EE trajectories (no robot, no GPU)."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.eeplan import EEPlan


def _identity_quats(n: int) -> np.ndarray:
    q = np.zeros((n, 4))
    q[:, 0] = 1.0  # wxyz identity
    return q


@pytest.fixture
def source_plan() -> EEPlan:
    """A comfortably-reachable arc on the table (well inside the workspace)."""
    t = np.linspace(0.0, 1.0, 30)
    pos = np.stack(
        [0.40 + 0.10 * np.cos(t * np.pi), 0.10 * np.sin(t * np.pi), 0.20 + 0.05 * t],
        axis=1,
    )
    grip = np.clip(0.5 + 0.4 * np.sin(t * 2 * np.pi), 0.0, 1.0)
    return EEPlan(
        positions=pos,
        quaternions=_identity_quats(len(t)),
        gripper=grip,
        timestamps=t,
        fps=30.0,
        task="pick cube",
    )


@pytest.fixture
def boundary_plan() -> EEPlan:
    """A tight trajectory cluster on the front diagonal at radius ~0.78, just
    inside generic_tabletop's reach_max=0.85 (and well within the x/y box).
    Outward transforms push it past the reach limit and are rejected; inward
    ones stay feasible, so reject-and-resample is genuinely exercised."""
    t = np.linspace(0.0, 1.0, 24)
    r = 0.78
    cx = r * np.cos(np.pi / 4)  # ~0.5515, < box limit 0.7
    cy = r * np.sin(np.pi / 4)
    pos = np.stack(
        [cx + 0.01 * np.cos(t * np.pi), cy + 0.01 * np.sin(t * np.pi), 0.12 + 0.02 * t],
        axis=1,
    )
    grip = np.full(len(t), 0.5)
    return EEPlan(
        positions=pos,
        quaternions=_identity_quats(len(t)),
        gripper=grip,
        timestamps=t,
        fps=30.0,
        task="reach far",
    )
