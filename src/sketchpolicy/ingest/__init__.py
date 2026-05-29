"""Experimental hand-video ingest (the "sketch" adapter).

The :class:`HandTrack` type and the deterministic :func:`synthetic_hand_track`
mock are always importable (torch/mediapipe-free). The real
:func:`ingest_video` lazily imports MediaPipe/OpenCV from the ``[ingest]`` extra.
"""

from __future__ import annotations

from sketchpolicy.ingest.mock import synthetic_hand_track
from sketchpolicy.ingest.types import HandTrack

__all__ = ["HandTrack", "synthetic_hand_track", "ingest_video", "IngestUnavailable"]


def __getattr__(name: str):  # lazy: avoid importing mediapipe at package import
    if name in ("ingest_video", "IngestUnavailable"):
        from sketchpolicy.ingest import mediapipe_hands

        return getattr(mediapipe_hands, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
