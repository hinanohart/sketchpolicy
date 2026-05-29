"""README honesty tests: no leftover placeholders, no overclaim, numbers traceable.

The headline throughput number in the README must equal the value in the
committed ``bench_results/v0.1.0a1.json`` (ship-and-yank lesson: README numbers
are measured, never invented). CI does not regenerate the bench file, so the two
committed artifacts stay in lock-step.
"""

from __future__ import annotations

import json
from pathlib import Path

from sketchpolicy.contract import scan_for_banned_phrases

_ROOT = Path(__file__).resolve().parents[1]
_README = (_ROOT / "README.md").read_text(encoding="utf-8")
_BENCH = json.loads(
    (_ROOT / "bench_results" / "v0.1.0a1.json").read_text(encoding="utf-8")
)


def test_no_unfilled_measure_placeholders() -> None:
    assert "MEASURED@S6" not in _README


def test_readme_has_no_banned_phrases() -> None:
    assert scan_for_banned_phrases(_README) == []


def test_headline_fps_matches_bench_json() -> None:
    fps = _BENCH["measurements"]["mediapipe_cpu_fps"]["fps"]
    assert fps is not None, "bench json should record a measured fps"
    assert f"{fps} fps" in _README, f"README must cite the measured {fps} fps"


def test_readme_states_non_claims() -> None:
    low = _README.lower()
    assert "success rate" in low  # explicitly disclaims policy-performance gains
    assert "sim-to-real" in low  # explicit sim-to-real disclaimer
    assert "experimental" in low  # sketch path honestly flagged


def test_bench_makes_no_accuracy_claim() -> None:
    note = _BENCH["claim_note"].lower()
    assert "no policy" in note
    assert "accuracy" in note
