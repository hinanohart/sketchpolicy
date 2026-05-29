"""Tests for the SE(3) + time-warp transform group (golden + property)."""

from __future__ import annotations

import numpy as np
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from sketchpolicy.augment.transforms import (
    N_PARAMS,
    ParamRanges,
    TransformParams,
    apply,
)
from sketchpolicy.eeplan import EEPlan


def test_identity_transform_is_a_noop(source_plan: EEPlan) -> None:
    out = apply(source_plan, TransformParams(0.0, 0.0, 0.0, 0.0, 1.0))
    assert np.allclose(out.positions, source_plan.positions, atol=1e-12)
    assert np.allclose(out.quaternions, source_plan.quaternions, atol=1e-12)
    assert np.allclose(out.gripper, source_plan.gripper, atol=1e-12)


def test_transform_preserves_frame_count(source_plan: EEPlan) -> None:
    out = apply(source_plan, TransformParams(0.3, 0.1, 0.05, -0.03, 1.2))
    assert len(out) == len(source_plan)


def test_transform_records_params(source_plan: EEPlan) -> None:
    out = apply(source_plan, TransformParams(0.1, 0.0, 0.0, 0.0, 1.0))
    assert "augment_params" in out.source
    assert out.source["augment_params"]["azimuth"] == 0.1


def test_time_warp_preserves_endpoints(source_plan: EEPlan) -> None:
    out = apply(source_plan, TransformParams(0.0, 0.0, 0.0, 0.0, 1.25))
    assert np.allclose(out.positions[0], source_plan.positions[0], atol=1e-6)
    assert np.allclose(out.positions[-1], source_plan.positions[-1], atol=1e-6)
    # the middle of the trajectory should move under a non-trivial warp
    mid = len(source_plan) // 2
    assert not np.allclose(out.positions[mid], source_plan.positions[mid])


def test_rigid_transform_preserves_pairwise_distances(source_plan: EEPlan) -> None:
    # A pure rigid SE(3) (no time-warp) is an isometry on the point set.
    out = apply(source_plan, TransformParams(0.4, 0.15, 0.06, -0.04, 1.0))
    d_before = np.linalg.norm(
        source_plan.positions[1:] - source_plan.positions[:-1], axis=1
    )
    d_after = np.linalg.norm(out.positions[1:] - out.positions[:-1], axis=1)
    assert np.allclose(d_before, d_after, atol=1e-9)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    az=st.floats(-np.pi / 6, np.pi / 6),
    el=st.floats(-np.pi / 12, np.pi / 12),
    dx=st.floats(-0.08, 0.08),
    dy=st.floats(-0.08, 0.08),
    gamma=st.floats(0.8, 1.25),
)
def test_property_quaternions_stay_unit(source_plan, az, el, dx, dy, gamma) -> None:
    out = apply(source_plan, TransformParams(az, el, dx, dy, gamma))
    norms = np.linalg.norm(out.quaternions, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-6)
    # canonical sign convention: w >= 0
    assert np.all(out.quaternions[:, 0] >= -1e-9)


def test_param_ranges_scale_unit_cube() -> None:
    ranges = ParamRanges()
    lows, highs = ranges.lows_highs()
    u = np.zeros((1, N_PARAMS))
    assert np.allclose(ranges.scale_unit_cube(u)[0], lows)
    u = np.ones((1, N_PARAMS))
    assert np.allclose(ranges.scale_unit_cube(u)[0], highs)
