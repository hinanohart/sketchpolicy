"""Experimental monocular hand-video ingest via MediaPipe Hands (CPU, Apache-2.0).

This is the *experimental* "sketch" adapter. It runs MediaPipe's HandLandmarker
in VIDEO mode over a monocular clip and returns a :class:`HandTrack`. Everything
is lazy-imported so the core install never needs MediaPipe; install the optional
extra with ``pip install 'sketchpolicy[ingest]'``.

Honest limits: MediaPipe world landmarks are metric but hand-centred, and image
landmarks have no absolute scale, so the resulting trajectory is **relative**.
This module deliberately uses MediaPipe Hands only — never MANO/HaMeR/WiLoR.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from sketchpolicy.ingest.types import N_LANDMARKS, HandTrack

#: The official Apache-2.0 MediaPipe Hands model. We deliberately do NOT fetch it
#: from library code (no silent network I/O); the user provisions it once.
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
    "hand_landmarker/float16/1/hand_landmarker.task"
)


class IngestUnavailable(RuntimeError):
    """Raised when the optional [ingest] dependencies or model are missing."""


def _default_model_path() -> Path:
    cache = Path(
        os.environ.get("SKETCHPOLICY_CACHE", Path.home() / ".cache" / "sketchpolicy")
    )
    return cache / "hand_landmarker.task"


def ensure_model(model_path: str | Path | None = None) -> Path:
    """Return a local path to the hand_landmarker.task model.

    The model is not auto-downloaded (library code performs no network I/O). If
    it is absent, a clear error explains how to provision it once.

    Raises:
        IngestUnavailable: if the model file cannot be found.
    """
    path = Path(model_path) if model_path else _default_model_path()
    if not path.exists():
        raise IngestUnavailable(
            f"MediaPipe Hands model not found at {path}. Download it once "
            f"(Apache-2.0, ~7 MB) and place it there, e.g.:\n"
            f"  curl -L -o {path} {MODEL_URL}\n"
            f"or pass model_path=..., or set SKETCHPOLICY_CACHE to its directory."
        )
    return path


def ingest_video(
    video_path: str | Path,
    *,
    model_path: str | Path | None = None,
    max_frames: int | None = None,
) -> HandTrack:
    """Run MediaPipe Hands over ``video_path`` and return a :class:`HandTrack`.

    Frames with no detected hand carry the previous frame's landmarks forward and
    are flagged ``detected=False`` so the retarget stage can smooth over them.

    Raises:
        IngestUnavailable: if mediapipe / opencv are not installed.
        FileNotFoundError: if the video does not exist.
    """
    try:
        import cv2
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except ImportError as exc:  # pragma: no cover - exercised via extra
        raise IngestUnavailable(
            "mediapipe and opencv are required for video ingest; install with "
            "`pip install 'sketchpolicy[ingest]'`"
        ) from exc

    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(video_path)

    model = ensure_model(model_path)
    options = vision.HandLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model)),
        running_mode=vision.RunningMode.VIDEO,
        num_hands=1,
    )
    landmarker = vision.HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(str(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    world_frames: list[np.ndarray] = []
    image_frames: list[np.ndarray] = []
    detected: list[bool] = []
    last_world = np.zeros((N_LANDMARKS, 3))
    last_image = np.zeros((N_LANDMARKS, 3))

    frame_idx = 0
    try:
        while True:
            ok, frame_bgr = cap.read()
            if not ok or (max_frames is not None and frame_idx >= max_frames):
                break
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
            ts_ms = int(round(frame_idx * 1000.0 / fps))
            result = landmarker.detect_for_video(mp_image, ts_ms)

            if result.hand_world_landmarks and result.hand_landmarks:
                w = np.array([[p.x, p.y, p.z] for p in result.hand_world_landmarks[0]])
                im = np.array([[p.x, p.y, p.z] for p in result.hand_landmarks[0]])
                last_world, last_image = w, im
                detected.append(True)
            else:
                detected.append(False)
            world_frames.append(last_world.copy())
            image_frames.append(last_image.copy())
            frame_idx += 1
    finally:
        cap.release()
        landmarker.close()

    if frame_idx == 0:
        raise IngestUnavailable(f"no frames decoded from {video_path}")

    timestamps = np.arange(frame_idx, dtype=np.float64) / fps
    return HandTrack(
        world_landmarks=np.stack(world_frames),
        image_landmarks=np.stack(image_frames),
        timestamps=timestamps,
        fps=float(fps),
        detected=np.array(detected, dtype=bool),
    )
