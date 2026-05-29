"""Replay smoke test (requires the optional [replay] extra: pybullet)."""

from __future__ import annotations

import numpy as np
import pytest

from sketchpolicy.eeplan import EEPlan

pytest.importorskip("pybullet", reason="requires the [replay] extra")

from sketchpolicy.replay import ReplayReport, replay_plan  # noqa: E402

pytestmark = pytest.mark.replay


def _reachable_plan() -> EEPlan:
    t = np.linspace(0.0, 1.0, 12)
    pos = np.stack(
        [0.40 + 0.08 * np.cos(t * np.pi), 0.08 * np.sin(t * np.pi), 0.50 + 0.05 * t],
        axis=1,
    )
    q = np.tile(np.array([0.0, 1.0, 0.0, 0.0]), (len(t), 1))  # pointing down
    return EEPlan(
        positions=pos,
        quaternions=q,
        gripper=np.full(len(t), 0.5),
        timestamps=t,
        fps=30.0,
    )


def test_replay_runs_and_reports_finite_residuals() -> None:
    report = replay_plan(_reachable_plan())
    assert isinstance(report, ReplayReport)
    assert report.n_frames == 12
    assert np.isfinite(report.max_position_residual_m)
    assert report.max_position_residual_m >= 0.0
    assert 0.0 <= report.reachable_fraction <= 1.0
