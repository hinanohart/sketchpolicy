"""A deterministic synthetic hand-track generator.

This lets the entire ingest -> retarget -> emit pipeline be tested without a
real video, a camera, or the optional MediaPipe dependency, and keeps CI fast
and GPU/GL-free. The synthetic hand opens and closes while translating across
the image and slowly rotating, so the downstream gripper, planar-position and
wrist-orientation retargeting are all exercised.
"""

from __future__ import annotations

import numpy as np

from sketchpolicy.ingest.types import N_LANDMARKS, HandTrack


def _canonical_hand() -> np.ndarray:
    """A static, roughly anatomical 21-landmark hand in a hand-centred frame (m).

    Fingers extend along +y, the palm spans x, and the thumb sits to one side.
    Returned shape is ``(21, 3)``.
    """
    lm = np.zeros((N_LANDMARKS, 3), dtype=np.float64)
    # wrist at origin-ish; fingers fan out along +y in metres (~0.08-0.10 m).
    lm[0] = [0.0, 0.0, 0.0]  # WRIST
    # thumb chain (to -x side)
    lm[1] = [-0.02, 0.02, 0.0]
    lm[2] = [-0.035, 0.04, 0.0]
    lm[3] = [-0.045, 0.06, 0.0]
    lm[4] = [-0.05, 0.075, 0.0]  # THUMB_TIP
    # index
    lm[5] = [-0.01, 0.08, 0.0]  # INDEX_MCP
    lm[6] = [-0.012, 0.10, 0.0]
    lm[7] = [-0.013, 0.115, 0.0]
    lm[8] = [-0.014, 0.13, 0.0]  # INDEX_TIP
    # middle
    lm[9] = [0.005, 0.085, 0.0]  # MIDDLE_MCP
    lm[10] = [0.006, 0.108, 0.0]
    lm[11] = [0.007, 0.124, 0.0]
    lm[12] = [0.008, 0.14, 0.0]
    # ring
    lm[13] = [0.02, 0.08, 0.0]
    lm[14] = [0.022, 0.10, 0.0]
    lm[15] = [0.023, 0.115, 0.0]
    lm[16] = [0.024, 0.13, 0.0]
    # pinky
    lm[17] = [0.035, 0.075, 0.0]  # PINKY_MCP
    lm[18] = [0.037, 0.092, 0.0]
    lm[19] = [0.038, 0.104, 0.0]
    lm[20] = [0.039, 0.115, 0.0]
    return lm


def synthetic_hand_track(
    n_frames: int = 40,
    fps: float = 30.0,
    *,
    handedness: str = "Right",
) -> HandTrack:
    """Generate a deterministic :class:`HandTrack` of a closing, moving hand."""
    if n_frames < 2:
        raise ValueError("n_frames must be >= 2")
    base = _canonical_hand()
    phase = np.linspace(0.0, 1.0, n_frames)

    world = np.repeat(base[None, :, :], n_frames, axis=0)
    # Open/close: scale the thumb's x-offset toward the index over time so the
    # thumb_tip<->index_tip aperture shrinks then grows.
    aperture_scale = 0.5 + 0.5 * np.cos(phase * 2 * np.pi)  # 1 -> 0 -> 1
    for i in range(n_frames):
        # move thumb chain toward index by (1 - aperture_scale)
        shift = (1.0 - aperture_scale[i]) * 0.045
        world[i, 1:5, 0] += shift  # thumb x toward index

    # image-space wrist trajectory: a diagonal sweep across the frame centre.
    image = np.repeat(base[None, :, :], n_frames, axis=0).copy()
    # normalise the canonical hand into a small image patch, then translate it.
    image[:, :, 0] = 0.5 + 0.6 * base[None, :, 0] + 0.18 * (phase[:, None] - 0.5)
    image[:, :, 1] = 0.5 + 0.6 * base[None, :, 1] - 0.18 * (phase[:, None] - 0.5)
    # a small relative-depth sweep so the experimental depth heuristic has signal
    image[:, :, 2] = 0.10 * (phase[:, None] - 0.5)

    timestamps = np.arange(n_frames, dtype=np.float64) / fps
    return HandTrack(
        world_landmarks=world,
        image_landmarks=image,
        timestamps=timestamps,
        fps=fps,
        handedness=handedness,
    )
