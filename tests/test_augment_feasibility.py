"""Tests for the reject-and-resample feasibility filter."""

from __future__ import annotations

import numpy as np

from sketchpolicy.augment.feasibility import check
from sketchpolicy.eeplan import EEPlan
from sketchpolicy.profiles import get_profile


def _flat_plan(positions: np.ndarray) -> EEPlan:
    n = positions.shape[0]
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    return EEPlan(
        positions=positions,
        quaternions=q,
        gripper=np.full(n, 0.5),
        timestamps=np.arange(n) / 10.0,
        fps=10.0,
    )


def test_source_inside_workspace_is_feasible(source_plan: EEPlan) -> None:
    report = check(source_plan, get_profile("generic_tabletop"))
    assert report.feasible
    assert report.first_reason() is None


def test_out_of_reach_detected() -> None:
    prof = get_profile("generic_tabletop")
    far = _flat_plan(np.array([[2.0, 0.0, 0.3], [2.1, 0.0, 0.3]]))
    report = check(far, prof)
    assert not report.feasible
    assert report.reasons["out_of_reach"]


def test_below_floor_detected() -> None:
    prof = get_profile("generic_tabletop")
    below = _flat_plan(np.array([[0.4, 0.0, -0.1], [0.4, 0.0, 0.2]]))
    report = check(below, prof)
    assert not report.feasible
    assert report.reasons["below_floor"]


def test_base_keepout_detected() -> None:
    prof = get_profile("generic_tabletop")
    # directly above the base, low z -> inside the keep-out cylinder, but still
    # within reach so only base_keepout should fire
    near_base = _flat_plan(np.array([[0.0, 0.0, 0.2], [0.02, 0.0, 0.2]]))
    report = check(near_base, prof)
    assert not report.feasible
    assert report.reasons["base_keepout"]
