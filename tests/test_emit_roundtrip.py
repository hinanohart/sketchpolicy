"""Tests for the torch-free LeRobotDataset v3.0 writer/reader round-trip."""

from __future__ import annotations

import json

import numpy as np
import pytest

from sketchpolicy.augment import multiply
from sketchpolicy.emit import read_plans, validate_dataset, write_dataset
from sketchpolicy.emit.schema import CODEBASE_VERSION, REQUIRED_INFO_KEYS
from sketchpolicy.emit.writer import EmitError
from sketchpolicy.eeplan import EEPlan


def test_write_creates_v30_layout(source_plan: EEPlan, tmp_path) -> None:
    out = write_dataset([source_plan], tmp_path / "ds")
    assert (out / "meta" / "info.json").exists()
    assert (out / "meta" / "stats.json").exists()
    assert (out / "meta" / "tasks.parquet").exists()
    assert (out / "meta" / "episodes" / "chunk-000" / "file-000.parquet").exists()
    assert (out / "data" / "chunk-000" / "file-000.parquet").exists()


def test_info_json_has_required_keys(source_plan: EEPlan, tmp_path) -> None:
    out = write_dataset([source_plan], tmp_path / "ds")
    info = json.loads((out / "meta" / "info.json").read_text(encoding="utf-8"))
    assert REQUIRED_INFO_KEYS.issubset(info.keys())
    assert info["codebase_version"] == CODEBASE_VERSION
    assert info["total_episodes"] == 1
    assert info["total_frames"] == len(source_plan)
    assert info["video_path"] is None


def test_roundtrip_recovers_plans(source_plan: EEPlan, tmp_path) -> None:
    res = multiply(source_plan, n=5, seed=0)
    plans = [source_plan, *res.variants]
    out = write_dataset(plans, tmp_path / "ds6")
    validate_dataset(out)
    back = read_plans(out)
    assert len(back) == len(plans)
    for original, recovered in zip(plans, back):
        assert np.allclose(
            original.to_action_array(), recovered.to_action_array(), atol=1e-6
        )


def test_overwrite_guard(source_plan: EEPlan, tmp_path) -> None:
    write_dataset([source_plan], tmp_path / "ds")
    with pytest.raises(EmitError, match="already exists"):
        write_dataset([source_plan], tmp_path / "ds")
    # overwrite=True succeeds
    write_dataset([source_plan], tmp_path / "ds", overwrite=True)


def test_empty_raises(tmp_path) -> None:
    with pytest.raises(EmitError, match="empty"):
        write_dataset([], tmp_path / "ds")


def test_inconsistent_fps_raises(source_plan: EEPlan, tmp_path) -> None:
    other = EEPlan(
        positions=source_plan.positions,
        quaternions=source_plan.quaternions,
        gripper=source_plan.gripper,
        timestamps=source_plan.timestamps * 2.0,  # half the rate
        fps=15.0,
        task=source_plan.task,
    )
    with pytest.raises(EmitError, match="fps"):
        write_dataset([source_plan, other], tmp_path / "ds")
