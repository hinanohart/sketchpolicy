"""Tests for the doctor environment report."""

from __future__ import annotations

from sketchpolicy.doctor import format_report, run_doctor


def test_doctor_reports_core_present() -> None:
    report = run_doctor()
    for pkg in ("numpy", "scipy", "pyarrow"):
        assert report.components[pkg]["present"], f"{pkg} should be importable"
    assert report.capability("core") is True


def test_doctor_lists_optional_extras() -> None:
    report = run_doctor()
    # lerobot-dataset is intentionally not a core dep; it may or may not be
    # installed, but it must be *reported* either way.
    assert "lerobot-dataset" in report.components
    assert report.components["lerobot-dataset"]["extra"] == "lerobot"


def test_format_report_is_readable() -> None:
    text = format_report(run_doctor())
    assert "sketchpolicy doctor" in text
    assert "numpy" in text
    assert "Profiles:" in text
