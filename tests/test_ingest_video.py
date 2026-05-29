"""Real MediaPipe video-ingest path (optional [ingest] extra + system GL).

This exercises sketchpolicy.ingest.mediapipe_hands end-to-end against a tiny
generated clip. It self-skips when MediaPipe/OpenCV are absent, when the GL
runtime needed by MediaPipe Tasks cannot be loaded, or when the model file is
not provisioned — so the core (extra-free) CI matrix simply skips it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2", reason="requires the [ingest] extra")
pytest.importorskip("mediapipe", reason="requires the [ingest] extra")

from sketchpolicy.ingest import ingest_video  # noqa: E402
from sketchpolicy.ingest.mediapipe_hands import IngestUnavailable  # noqa: E402
from sketchpolicy.ingest.types import N_LANDMARKS  # noqa: E402

pytestmark = pytest.mark.ingest


def _model_path() -> str | None:
    candidates = [
        os.environ.get("SKETCHPOLICY_HAND_MODEL"),
        str(Path.home() / ".cache" / "sketchpolicy" / "hand_landmarker.task"),
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    return None


def _write_clip(path: Path, n: int = 8) -> None:
    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (160, 120)
    )
    rng = np.random.default_rng(0)
    for _ in range(n):
        writer.write(rng.integers(0, 255, (120, 160, 3), dtype=np.uint8))
    writer.release()


def test_ingest_video_returns_handtrack(tmp_path) -> None:
    model = _model_path()
    if model is None:
        pytest.skip("hand_landmarker.task model not provisioned")
    clip = tmp_path / "clip.mp4"
    _write_clip(clip, n=8)
    if not clip.exists() or clip.stat().st_size == 0:
        pytest.skip("could not encode a test clip (no mp4 codec)")
    try:
        track = ingest_video(clip, model_path=model)
    except IngestUnavailable as exc:
        pytest.skip(f"ingest runtime unavailable: {exc}")
    except OSError as exc:  # e.g. missing system GL libraries
        pytest.skip(f"GL runtime unavailable: {exc}")
    assert track.world_landmarks.shape[1:] == (N_LANDMARKS, 3)
    assert track.image_landmarks.shape[1:] == (N_LANDMARKS, 3)
    assert len(track) >= 1
