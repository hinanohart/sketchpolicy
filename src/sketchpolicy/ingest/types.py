"""The hand-track type produced by ingest and consumed by retarget.

MediaPipe Hands gives two complementary read-outs per frame, and the honest
retarget pipeline needs both:

* ``world_landmarks`` — 21 points in **metres**, in a hand-centred frame. These
  describe the hand's *shape and orientation* (used for gripper aperture and
  wrist orientation). They do **not** encode where the hand is in the scene.
* ``image_landmarks`` — 21 points in **normalised image coordinates** (x, y in
  ``[0, 1]``, z a relative depth). The wrist's image motion is what we map to a
  *planar* end-effector trajectory; absolute metric scale is not recoverable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Standard MediaPipe Hands 21-landmark indices.
WRIST = 0
THUMB_TIP = 4
INDEX_MCP = 5
INDEX_TIP = 8
MIDDLE_MCP = 9
PINKY_MCP = 17
N_LANDMARKS = 21


@dataclass(frozen=True)
class HandTrack:
    """A per-frame hand landmark track from a monocular video."""

    world_landmarks: np.ndarray  # (T, 21, 3) metres, hand-centred
    image_landmarks: np.ndarray  # (T, 21, 3) normalised image coords
    timestamps: np.ndarray  # (T,) seconds
    fps: float
    handedness: str = "Right"
    detected: np.ndarray | None = None  # (T,) bool; None => all detected

    def __post_init__(self) -> None:
        t = self.world_landmarks.shape[0]
        if self.world_landmarks.shape != (t, N_LANDMARKS, 3):
            raise ValueError(
                f"world_landmarks must be (T, {N_LANDMARKS}, 3), got "
                f"{self.world_landmarks.shape}"
            )
        if self.image_landmarks.shape != (t, N_LANDMARKS, 3):
            raise ValueError(
                f"image_landmarks must be (T, {N_LANDMARKS}, 3), got "
                f"{self.image_landmarks.shape}"
            )
        if self.timestamps.shape != (t,):
            raise ValueError(f"timestamps must be (T,), got {self.timestamps.shape}")
        if t == 0:
            raise ValueError("HandTrack must have at least one frame")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        if self.detected is not None and self.detected.shape != (t,):
            raise ValueError(f"detected must be (T,), got {self.detected.shape}")

    def __len__(self) -> int:
        return int(self.world_landmarks.shape[0])

    @property
    def detected_mask(self) -> np.ndarray:
        if self.detected is None:
            return np.ones(len(self), dtype=bool)
        return self.detected
