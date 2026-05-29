"""A torch-free reader/validator for the sketchpolicy-written v3.0 datasets.

This is what backs the *round-trip* claim: a dataset written by
:func:`sketchpolicy.emit.writer.write_dataset` can be read back with pyarrow
alone and validated against the declared schema, and the per-episode
:class:`EEPlan` objects recovered. It does not depend on ``lerobot-dataset``;
the optional ``[lerobot]`` extra provides a separate cross-check with the real
reader.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from sketchpolicy.eeplan import ACTION_DIM, EEPlan
from sketchpolicy.emit import schema


class SchemaError(RuntimeError):
    """Raised when a dataset directory violates the v3.0 schema we emit."""


def load_info(root: str | Path) -> dict:
    return json.loads((Path(root) / schema.INFO_PATH).read_text(encoding="utf-8"))


def validate_dataset(root: str | Path) -> dict:
    """Validate the on-disk v3.0 structure. Returns the info dict on success.

    Raises:
        SchemaError: if any required file, key or column is missing/wrong.
    """
    root = Path(root)
    if not (root / schema.INFO_PATH).exists():
        raise SchemaError(f"missing {schema.INFO_PATH}")
    info = load_info(root)

    missing = schema.REQUIRED_INFO_KEYS - info.keys()
    if missing:
        raise SchemaError(f"info.json missing keys: {sorted(missing)}")
    if info["codebase_version"] != schema.CODEBASE_VERSION:
        raise SchemaError(
            f"codebase_version {info['codebase_version']!r} != {schema.CODEBASE_VERSION!r}"
        )

    for rel in (schema.STATS_PATH, schema.TASKS_PATH):
        if not (root / rel).exists():
            raise SchemaError(f"missing {rel}")

    data_path = root / schema.DATA_PATH.format(chunk_index=0, file_index=0)
    if not data_path.exists():
        raise SchemaError(f"missing data file {data_path}")
    ep_path = root / schema.EPISODES_PATH.format(chunk_index=0, file_index=0)
    if not ep_path.exists():
        raise SchemaError(f"missing episodes file {ep_path}")

    data_cols = set(pq.read_schema(data_path).names)
    expected_cols = set(info["features"].keys())
    if not expected_cols.issubset(data_cols):
        raise SchemaError(
            f"data parquet columns {sorted(data_cols)} miss declared features "
            f"{sorted(expected_cols - data_cols)}"
        )

    table = pq.read_table(data_path)
    if table.num_rows != info["total_frames"]:
        raise SchemaError(
            f"total_frames {info['total_frames']} != data rows {table.num_rows}"
        )

    n_eps = pq.read_table(ep_path).num_rows
    if n_eps != info["total_episodes"]:
        raise SchemaError(
            f"total_episodes {info['total_episodes']} != episodes rows {n_eps}"
        )
    return info


def read_plans(root: str | Path) -> list[EEPlan]:
    """Reconstruct the per-episode :class:`EEPlan` list from a v3.0 dataset."""
    root = Path(root)
    info = validate_dataset(root)
    fps = float(info["fps"])

    data = pq.read_table(
        root / schema.DATA_PATH.format(chunk_index=0, file_index=0)
    ).to_pydict()
    episodes = pq.read_table(
        root / schema.EPISODES_PATH.format(chunk_index=0, file_index=0)
    ).to_pydict()

    action = np.asarray(data["action"], dtype=np.float64)  # (N, 8)
    timestamp = np.asarray(data["timestamp"], dtype=np.float64)
    if action.ndim != 2 or action.shape[1] != ACTION_DIM:
        raise SchemaError(f"action column is not (N, {ACTION_DIM}): {action.shape}")

    plans: list[EEPlan] = []
    n = len(episodes["episode_index"])
    for i in range(n):
        start = episodes["dataset_from_index"][i]
        end = episodes["dataset_to_index"][i]
        task = (
            episodes["tasks"][i][0]
            if episodes["tasks"][i]
            else "sketchpolicy trajectory"
        )
        plans.append(
            EEPlan.from_action_array(
                action[start:end],
                fps=fps,
                task=task,
                timestamps=timestamp[start:end],
            )
        )
    return plans
