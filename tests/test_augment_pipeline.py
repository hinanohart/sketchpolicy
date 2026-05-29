"""Tests for the augmentation spine: count, determinism, reject-and-resample."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.augment import multiply
from sketchpolicy.augment.pipeline import FeasibilityExhausted
from sketchpolicy.eeplan import EEPlan
from sketchpolicy.profiles import get_profile


def _plans_equal(a: EEPlan, b: EEPlan) -> bool:
    return (
        np.array_equal(a.positions, b.positions)
        and np.array_equal(a.quaternions, b.quaternions)
        and np.array_equal(a.gripper, b.gripper)
        and np.array_equal(a.timestamps, b.timestamps)
    )


def test_multiply_returns_requested_count(source_plan: EEPlan) -> None:
    res = multiply(source_plan, n=12, seed=0)
    assert len(res.variants) == 12
    assert res.n_requested == 12


def test_multiply_is_bit_exact_for_same_seed(source_plan: EEPlan) -> None:
    a = multiply(source_plan, n=8, seed=7)
    b = multiply(source_plan, n=8, seed=7)
    assert all(_plans_equal(x, y) for x, y in zip(a.variants, b.variants))


def test_multiply_differs_across_seeds(source_plan: EEPlan) -> None:
    a = multiply(source_plan, n=4, seed=0)
    b = multiply(source_plan, n=4, seed=1)
    assert not _plans_equal(a.variants[0], b.variants[0])


def test_all_variants_are_feasible(source_plan: EEPlan) -> None:
    from sketchpolicy.augment import check

    prof = get_profile("generic_tabletop")
    res = multiply(source_plan, n=10, seed=3)
    assert all(check(v, prof).feasible for v in res.variants)


def test_reject_and_resample_exercised(boundary_plan: EEPlan) -> None:
    # The boundary trajectory hugs the reach limit, so some transforms leave the
    # workspace and must be rejected, yet the requested count is still met.
    res = multiply(boundary_plan, n=5, seed=0)
    assert len(res.variants) == 5
    assert res.n_rejected >= 1
    assert res.n_drawn > 5
    assert 0.0 < res.acceptance_rate <= 1.0


def test_infeasible_request_raises(source_plan: EEPlan) -> None:
    # A trajectory far outside any reachable shell can never be made feasible.
    far = EEPlan(
        positions=source_plan.positions + np.array([5.0, 0.0, 0.0]),
        quaternions=source_plan.quaternions,
        gripper=source_plan.gripper,
        timestamps=source_plan.timestamps,
        fps=source_plan.fps,
    )
    with pytest.raises(FeasibilityExhausted):
        multiply(far, n=3, seed=0)


def test_non_positive_n_raises(source_plan: EEPlan) -> None:
    with pytest.raises(ValueError, match="n must be positive"):
        multiply(source_plan, n=0, seed=0)
