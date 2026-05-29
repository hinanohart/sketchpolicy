"""Thumb-index aperture -> normalised parallel-jaw gripper width.

Aperture is a *shape* property, so it is read from the metric, hand-centred
``world_landmarks``. The mapping is a per-track percentile calibration: the 5th
percentile aperture maps to a closed gripper (0) and the 95th to open (1). Using
percentiles (rather than min/max) makes the calibration robust to a few
mis-detected frames.
"""

from __future__ import annotations

import numpy as np

from sketchpolicy.ingest.types import INDEX_TIP, THUMB_TIP, HandTrack


def aperture_series(track: HandTrack) -> np.ndarray:
    """Per-frame thumb_tip<->index_tip distance (metres), shape ``(T,)``."""
    thumb = track.world_landmarks[:, THUMB_TIP, :]
    index = track.world_landmarks[:, INDEX_TIP, :]
    return np.linalg.norm(thumb - index, axis=1)


def aperture_to_gripper(
    track: HandTrack, *, p_low: float = 5.0, p_high: float = 95.0
) -> np.ndarray:
    """Map the aperture series to gripper width in ``[0, 1]`` (1 == open).

    Args:
        track: the hand track.
        p_low: percentile mapped to a closed gripper (0).
        p_high: percentile mapped to an open gripper (1).
    """
    aperture = aperture_series(track)
    lo = float(np.percentile(aperture, p_low))
    hi = float(np.percentile(aperture, p_high))
    if hi - lo < 1e-6:
        # Degenerate: aperture is essentially constant -> mid-open, well-defined.
        return np.full_like(aperture, 0.5)
    return np.clip((aperture - lo) / (hi - lo), 0.0, 1.0)
