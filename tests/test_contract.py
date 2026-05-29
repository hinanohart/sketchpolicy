"""Honest-marketing contract tests (exact-string, plus a negative fixture).

These use ``==`` exact comparison rather than a shell grep on purpose: the
banned-phrase list lives only in :mod:`sketchpolicy.contract`, and the negative
fixture proves the scan actually fires so the positive assertions cannot pass
vacuously.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sketchpolicy.contract import (
    BOOTSTRAP_FLAG,
    DISCLAIMER,
    NON_CLAIMS,
    BootstrapAckError,
    require_bootstrap_ack,
    scan_for_banned_phrases,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]


def test_bootstrap_gate_blocks_without_flag() -> None:
    with pytest.raises(BootstrapAckError, match="bootstrap"):
        require_bootstrap_ack(False)


def test_bootstrap_gate_allows_with_flag() -> None:
    require_bootstrap_ack(True)  # must not raise


def test_bootstrap_flag_spelling() -> None:
    assert BOOTSTRAP_FLAG == "--i-understand-this-is-a-bootstrap-scaffold"


def test_disclaimer_has_no_banned_phrases() -> None:
    assert scan_for_banned_phrases(DISCLAIMER) == []


def test_non_claims_present() -> None:
    assert len(NON_CLAIMS) >= 5
    joined = " ".join(NON_CLAIMS).lower()
    assert "policy" in joined
    assert "scale" in joined


def test_readme_has_no_banned_phrases() -> None:
    readme = (_REPO_ROOT / "README.md").read_text(encoding="utf-8")
    assert scan_for_banned_phrases(readme) == []


def test_negative_fixture_scan_fires() -> None:
    # Proves the scan is not vacuous: a deliberately over-claiming string must
    # be flagged, exactly.
    bad = "This is a state-of-the-art, production-ready tool that is fully automatic."
    assert scan_for_banned_phrases(bad) == [
        "state-of-the-art",
        "production-ready",
        "fully automatic",
    ]
