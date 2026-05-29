"""Cross-check: the real `lerobot-dataset` reader loads sketchpolicy output.

This is the gold-standard backing for the "schema-valid LeRobotDataset v3.0"
claim: it does not trust our hand-rolled writer in isolation, it loads the
emitted directory with the actual `ledataset` reader and the high-level
`LeRobotDataset`. It self-skips unless the optional `[lerobot]` extra is
installed (that extra pulls a heavy torch + CUDA stack), so the core CI matrix
never runs it.
"""

from __future__ import annotations

import numpy as np
import pytest

ledataset = pytest.importorskip(
    "ledataset", reason="requires the [lerobot] extra (heavy: torch + CUDA)"
)

from sketchpolicy.eeplan import EEPlan  # noqa: E402
from sketchpolicy.emit import write_dataset  # noqa: E402
from sketchpolicy.emit.schema import CODEBASE_VERSION  # noqa: E402

pytestmark = pytest.mark.lerobot


def _dataset(path) -> int:
    t = np.linspace(0.0, 1.0, 20)
    pos = np.stack(
        [0.40 + 0.10 * np.cos(t * np.pi), 0.10 * np.sin(t * np.pi), 0.20 + 0.05 * t],
        axis=1,
    )
    q = np.tile(np.array([1.0, 0.0, 0.0, 0.0]), (20, 1))
    g = np.clip(0.5 + 0.4 * np.sin(t * 6), 0.0, 1.0)
    src = EEPlan(
        positions=pos,
        quaternions=q,
        gripper=g,
        timestamps=t,
        fps=30.0,
        task="pick cube",
    )
    write_dataset([src, src], path)  # 2 episodes
    return 2


def test_codebase_version_matches_installed() -> None:
    from ledataset.datasets.lerobot_dataset import CODEBASE_VERSION as real

    assert CODEBASE_VERSION == real


def test_lower_level_loaders_accept_output(tmp_path) -> None:
    from ledataset.datasets.utils import load_episodes, load_info, load_nested_dataset

    root = tmp_path / "ds"
    n = _dataset(root)
    info = load_info(root)
    assert info["codebase_version"] == CODEBASE_VERSION
    assert info["total_episodes"] == n
    assert len(load_episodes(root)) == n
    assert len(load_nested_dataset(root / "data")) == n * 20


def test_high_level_lerobotdataset_loads_and_indexes(tmp_path) -> None:
    from ledataset.datasets.lerobot_dataset import LeRobotDataset

    root = tmp_path / "ds"
    n = _dataset(root)
    ds = LeRobotDataset("sketchpolicy/xcheck", root=root)
    assert len(ds) == n * 20
    item = ds[0]
    assert tuple(item["action"].shape) == (8,)
    assert "observation.state" in item
