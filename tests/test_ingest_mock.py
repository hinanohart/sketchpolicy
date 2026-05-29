"""Tests for the synthetic hand-track generator and the HandTrack type."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.ingest import synthetic_hand_track
from sketchpolicy.ingest.types import N_LANDMARKS, HandTrack


def test_synthetic_track_shapes() -> None:
    track = synthetic_hand_track(n_frames=40, fps=30.0)
    assert len(track) == 40
    assert track.world_landmarks.shape == (40, N_LANDMARKS, 3)
    assert track.image_landmarks.shape == (40, N_LANDMARKS, 3)
    assert track.fps == 30.0
    assert track.detected_mask.all()


def test_synthetic_track_is_deterministic() -> None:
    a = synthetic_hand_track(30)
    b = synthetic_hand_track(30)
    assert np.array_equal(a.world_landmarks, b.world_landmarks)
    assert np.array_equal(a.image_landmarks, b.image_landmarks)


def test_handtrack_rejects_bad_shape() -> None:
    with pytest.raises(ValueError, match="world_landmarks"):
        HandTrack(
            world_landmarks=np.zeros((5, 3, 3)),  # wrong landmark count
            image_landmarks=np.zeros((5, N_LANDMARKS, 3)),
            timestamps=np.arange(5) / 10.0,
            fps=10.0,
        )


def test_too_few_frames_raises() -> None:
    with pytest.raises(ValueError, match="n_frames"):
        synthetic_hand_track(1)
