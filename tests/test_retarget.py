"""Tests for the experimental retarget: gripper, orientation, planar position."""

from __future__ import annotations

import numpy as np

from sketchpolicy.ingest import synthetic_hand_track
from sketchpolicy.ingest.types import HandTrack
from sketchpolicy.profiles import get_profile
from sketchpolicy.retarget import RetargetConfig, retarget_track
from sketchpolicy.retarget.gripper import aperture_to_gripper


def test_gripper_normalised_to_unit_interval() -> None:
    track = synthetic_hand_track(40)
    g = aperture_to_gripper(track)
    assert g.shape == (40,)
    assert g.min() >= 0.0 and g.max() <= 1.0
    # the synthetic hand opens and closes, so it should span a wide range
    assert g.max() - g.min() > 0.5


def test_gripper_constant_aperture_is_mid() -> None:
    track = synthetic_hand_track(20)
    flat = HandTrack(
        world_landmarks=np.repeat(track.world_landmarks[:1], 20, axis=0),
        image_landmarks=track.image_landmarks,
        timestamps=track.timestamps,
        fps=track.fps,
    )
    assert np.allclose(aperture_to_gripper(flat), 0.5)


def test_retarget_produces_valid_plan() -> None:
    track = synthetic_hand_track(40)
    plan = retarget_track(track)
    assert len(plan) == 40
    assert np.allclose(np.linalg.norm(plan.quaternions, axis=1), 1.0, atol=1e-6)
    assert plan.gripper.min() >= 0.0 and plan.gripper.max() <= 1.0


def test_retarget_default_is_feasible() -> None:
    plan = retarget_track(synthetic_hand_track(40))
    assert check_feasible(plan)


def check_feasible(plan) -> bool:
    from sketchpolicy.augment import check

    return check(plan, get_profile("generic_tabletop")).feasible


def test_planar_default_z_is_constant() -> None:
    plan = retarget_track(
        synthetic_hand_track(40), RetargetConfig(z_mode="constant", z_value=0.17)
    )
    assert np.allclose(plan.positions[:, 2], 0.17)


def test_depth_heuristic_varies_z() -> None:
    plan = retarget_track(
        synthetic_hand_track(40),
        RetargetConfig(z_mode="depth_heuristic", z_range=(0.10, 0.25), smooth=False),
    )
    z = plan.positions[:, 2]
    assert z.min() >= 0.10 - 1e-6 and z.max() <= 0.25 + 1e-6
    assert z.max() - z.min() > 0.05  # the heuristic actually moves z


def test_dropped_frames_are_interpolated() -> None:
    track = synthetic_hand_track(30)
    detected = np.ones(30, dtype=bool)
    detected[5:10] = False
    gappy = HandTrack(
        world_landmarks=track.world_landmarks,
        image_landmarks=track.image_landmarks,
        timestamps=track.timestamps,
        fps=track.fps,
        detected=detected,
    )
    plan = retarget_track(gappy)
    assert len(plan) == 30
    assert np.all(np.isfinite(plan.positions))


def test_orientation_is_proper_rotation() -> None:
    from scipy.spatial.transform import Rotation

    from sketchpolicy.retarget.pose import hand_frame_quaternions

    quats = hand_frame_quaternions(synthetic_hand_track(20))
    for q in quats:
        # wxyz -> xyzw; a proper rotation has det == +1 (Rotation guarantees it)
        rot = Rotation.from_quat(q[[1, 2, 3, 0]])
        assert np.isclose(np.linalg.det(rot.as_matrix()), 1.0, atol=1e-6)
