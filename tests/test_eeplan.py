"""Tests for the EEPlan type boundary."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.eeplan import ACTION_DIM, EEPlan


def test_action_array_roundtrip(source_plan: EEPlan) -> None:
    arr = source_plan.to_action_array()
    assert arr.shape == (len(source_plan), ACTION_DIM)
    rebuilt = EEPlan.from_action_array(
        arr, fps=source_plan.fps, timestamps=source_plan.timestamps
    )
    assert np.allclose(rebuilt.positions, source_plan.positions)
    assert np.allclose(rebuilt.quaternions, source_plan.quaternions)
    assert np.allclose(rebuilt.gripper, source_plan.gripper)


def test_iteration_yields_steps(source_plan: EEPlan) -> None:
    steps = list(source_plan)
    assert len(steps) == len(source_plan)
    assert steps[0].position.shape == (3,)
    assert steps[0].quaternion.shape == (4,)


def test_rejects_non_unit_quaternion() -> None:
    n = 4
    q = np.tile(np.array([2.0, 0.0, 0.0, 0.0]), (n, 1))  # not unit norm
    with pytest.raises(ValueError, match="unit-norm"):
        EEPlan(
            positions=np.zeros((n, 3)),
            quaternions=q,
            gripper=np.zeros(n),
            timestamps=np.arange(n) / 10.0,
            fps=10.0,
        )


def test_rejects_gripper_out_of_range() -> None:
    n = 3
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    with pytest.raises(ValueError, match="gripper"):
        EEPlan(
            positions=np.zeros((n, 3)),
            quaternions=q,
            gripper=np.array([0.0, 1.5, 0.0]),
            timestamps=np.arange(n) / 10.0,
            fps=10.0,
        )


def test_rejects_decreasing_timestamps() -> None:
    n = 3
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    with pytest.raises(ValueError, match="non-decreasing"):
        EEPlan(
            positions=np.zeros((n, 3)),
            quaternions=q,
            gripper=np.zeros(n),
            timestamps=np.array([0.0, 0.2, 0.1]),
            fps=10.0,
        )


def test_rejects_bad_fps() -> None:
    n = 2
    q = np.zeros((n, 4))
    q[:, 0] = 1.0
    with pytest.raises(ValueError, match="fps"):
        EEPlan(
            positions=np.zeros((n, 3)),
            quaternions=q,
            gripper=np.zeros(n),
            timestamps=np.arange(n) / 10.0,
            fps=0.0,
        )
