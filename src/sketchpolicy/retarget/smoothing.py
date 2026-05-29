"""Dropped-frame interpolation and Savitzky-Golay smoothing helpers."""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from sketchpolicy.ingest.types import HandTrack


def fill_dropped(track: HandTrack) -> HandTrack:
    """Linearly interpolate landmarks on frames where no hand was detected.

    Raises:
        ValueError: if no frame has a detected hand.
    """
    mask = track.detected_mask
    if mask.all():
        return track
    if not mask.any():
        raise ValueError("no detected hand in any frame; cannot retarget")

    idx = np.arange(len(track))
    det_idx = idx[mask]

    def _interp(arr: np.ndarray) -> np.ndarray:
        out = arr.copy()
        flat = out.reshape(len(track), -1)
        for c in range(flat.shape[1]):
            flat[:, c] = np.interp(idx, det_idx, flat[mask, c])
        return flat.reshape(arr.shape)

    return HandTrack(
        world_landmarks=_interp(track.world_landmarks),
        image_landmarks=_interp(track.image_landmarks),
        timestamps=track.timestamps,
        fps=track.fps,
        handedness=track.handedness,
        detected=np.ones(len(track), dtype=bool),
    )


def _odd_window(length: int, requested: int) -> int:
    w = min(requested, length)
    if w % 2 == 0:
        w -= 1
    return max(w, 3) if length >= 3 else length


def savgol_smooth(arr: np.ndarray, window: int = 9, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay smooth along axis 0, adapting the window to short inputs.

    Returns the input unchanged if it is too short to smooth.
    """
    length = arr.shape[0]
    if length < 5:
        return arr
    w = _odd_window(length, window)
    if w <= poly:
        return arr
    return np.asarray(savgol_filter(arr, window_length=w, polyorder=poly, axis=0))
