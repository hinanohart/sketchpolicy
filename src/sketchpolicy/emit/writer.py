"""A torch-free writer for the LeRobotDataset v3.0 format (pure pyarrow + json).

Given a list of :class:`~sketchpolicy.eeplan.EEPlan` episodes, :func:`write_dataset`
materialises a complete v3.0 directory: ``meta/info.json``, ``meta/stats.json``,
``meta/tasks.parquet``, ``meta/episodes/chunk-000/file-000.parquet`` and
``data/chunk-000/file-000.parquet``.

Convention for this kinematic scaffold: there is no dynamics model, so both
``action`` and ``observation.state`` encode the absolute end-effector pose
``[x, y, z, qw, qx, qy, qz, gripper]`` at each frame. v0.1 writes no videos.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from sketchpolicy.eeplan import ACTION_DIM, EEPlan
from sketchpolicy.emit import schema


class EmitError(RuntimeError):
    """Raised when a dataset cannot be written (e.g. inconsistent fps)."""


def _int_fps(plans: list[EEPlan]) -> int:
    fps_values = {round(p.fps) for p in plans}
    if len(fps_values) != 1:
        raise EmitError(f"all episodes must share one integer fps; got {fps_values}")
    fps = fps_values.pop()
    if not all(abs(p.fps - fps) < 1e-6 for p in plans):
        raise EmitError(f"fps must be integer-valued; got {[p.fps for p in plans]}")
    return int(fps)


def _stats_for(values: np.ndarray) -> dict:
    """Per-dimension statistics for a ``(N, d)`` array (d==1 for scalars)."""
    return {
        "min": values.min(axis=0).astype(np.float64).tolist(),
        "max": values.max(axis=0).astype(np.float64).tolist(),
        "mean": values.mean(axis=0).astype(np.float64).tolist(),
        "std": values.std(axis=0).astype(np.float64).tolist(),
        "count": [int(values.shape[0])],
    }


def write_dataset(
    plans: list[EEPlan],
    out_dir: str | Path,
    *,
    robot_type: str | None = "sketchpolicy_ee",
    overwrite: bool = False,
) -> Path:
    """Write ``plans`` as a LeRobotDataset v3.0 directory at ``out_dir``.

    Args:
        plans: episodes to write (each a valid :class:`EEPlan`).
        out_dir: destination directory.
        robot_type: value recorded in ``info.json``.
        overwrite: if True, an existing destination is removed first.

    Returns:
        The dataset root path.

    Raises:
        EmitError: on inconsistent fps or if the destination exists and
            ``overwrite`` is False.
    """
    if not plans:
        raise EmitError("cannot write an empty dataset (no episodes)")
    out = Path(out_dir)
    if out.exists():
        if not overwrite:
            raise EmitError(f"{out} already exists; pass overwrite=True to replace it")
        shutil.rmtree(out)

    fps = _int_fps(plans)

    # -- assign task indices in first-appearance order ----------------------
    task_to_index: dict[str, int] = {}
    for p in plans:
        if p.task not in task_to_index:
            task_to_index[p.task] = len(task_to_index)

    # -- build the concatenated per-frame data table ------------------------
    actions: list[np.ndarray] = []
    states: list[np.ndarray] = []
    timestamps: list[float] = []
    frame_index: list[int] = []
    episode_index: list[int] = []
    global_index: list[int] = []
    task_index: list[int] = []

    episode_rows: list[dict] = []
    running = 0
    for ep_idx, plan in enumerate(plans):
        pose = plan.to_action_array().astype(np.float32)  # (T, 8)
        t = pose.shape[0]
        actions.append(pose)
        states.append(pose)
        timestamps.extend(plan.timestamps.astype(np.float32).tolist())
        frame_index.extend(range(t))
        episode_index.extend([ep_idx] * t)
        global_index.extend(range(running, running + t))
        task_index.extend([task_to_index[plan.task]] * t)

        episode_rows.append(
            {
                "episode_index": ep_idx,
                "tasks": [plan.task],
                "length": t,
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": running,
                "dataset_to_index": running + t,
                "meta/episodes/chunk_index": 0,
                "meta/episodes/file_index": 0,
            }
        )
        running += t

    action_arr = np.concatenate(actions, axis=0)
    state_arr = np.concatenate(states, axis=0)
    total_frames = action_arr.shape[0]

    list_f32 = pa.list_(pa.float32())
    data_table = pa.table(
        {
            "action": pa.array(action_arr.tolist(), type=list_f32),
            "observation.state": pa.array(state_arr.tolist(), type=list_f32),
            "timestamp": pa.array(timestamps, type=pa.float32()),
            "frame_index": pa.array(frame_index, type=pa.int64()),
            "episode_index": pa.array(episode_index, type=pa.int64()),
            "index": pa.array(global_index, type=pa.int64()),
            "task_index": pa.array(task_index, type=pa.int64()),
        }
    )

    # -- write directory tree ----------------------------------------------
    (out / "meta").mkdir(parents=True, exist_ok=True)
    data_path = out / schema.DATA_PATH.format(chunk_index=0, file_index=0)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(data_table, data_path)

    size_mb = data_path.stat().st_size / (1024**2)
    if size_mb > schema.DEFAULT_DATA_FILE_SIZE_IN_MB:
        raise EmitError(
            f"v0.1 writes a single data file but the result is {size_mb:.0f} MB "
            f"(> {schema.DEFAULT_DATA_FILE_SIZE_IN_MB} MB). Multi-file chunking is "
            "deferred; split the input or reduce the episode count."
        )

    # tasks.parquet: index named "task" (string) + column "task_index"
    tasks_sorted = sorted(task_to_index.items(), key=lambda kv: kv[1])
    tasks_table = pa.table(
        {
            "task": pa.array([t for t, _ in tasks_sorted], type=pa.string()),
            "task_index": pa.array([i for _, i in tasks_sorted], type=pa.int64()),
        }
    )
    pq.write_table(tasks_table, out / schema.TASKS_PATH)

    # episodes parquet
    ep_path = out / schema.EPISODES_PATH.format(chunk_index=0, file_index=0)
    ep_path.parent.mkdir(parents=True, exist_ok=True)
    episodes_table = pa.table(
        {
            "episode_index": pa.array(
                [r["episode_index"] for r in episode_rows], pa.int64()
            ),
            "tasks": pa.array(
                [r["tasks"] for r in episode_rows], pa.list_(pa.string())
            ),
            "length": pa.array([r["length"] for r in episode_rows], pa.int64()),
            "data/chunk_index": pa.array(
                [r["data/chunk_index"] for r in episode_rows], pa.int64()
            ),
            "data/file_index": pa.array(
                [r["data/file_index"] for r in episode_rows], pa.int64()
            ),
            "dataset_from_index": pa.array(
                [r["dataset_from_index"] for r in episode_rows], pa.int64()
            ),
            "dataset_to_index": pa.array(
                [r["dataset_to_index"] for r in episode_rows], pa.int64()
            ),
            "meta/episodes/chunk_index": pa.array(
                [r["meta/episodes/chunk_index"] for r in episode_rows], pa.int64()
            ),
            "meta/episodes/file_index": pa.array(
                [r["meta/episodes/file_index"] for r in episode_rows], pa.int64()
            ),
        }
    )
    pq.write_table(episodes_table, ep_path)

    # info.json
    features = schema.ee_features()
    info = schema.build_info(
        fps=fps,
        features=features,
        total_episodes=len(plans),
        total_frames=total_frames,
        total_tasks=len(task_to_index),
        robot_type=robot_type,
    )
    (out / schema.INFO_PATH).write_text(
        json.dumps(info, indent=4) + "\n", encoding="utf-8"
    )

    # stats.json
    stats = {
        "action": _stats_for(action_arr.astype(np.float64)),
        "observation.state": _stats_for(state_arr.astype(np.float64)),
        "timestamp": _stats_for(np.asarray(timestamps, np.float64).reshape(-1, 1)),
        "frame_index": _stats_for(np.asarray(frame_index, np.float64).reshape(-1, 1)),
        "episode_index": _stats_for(
            np.asarray(episode_index, np.float64).reshape(-1, 1)
        ),
        "index": _stats_for(np.asarray(global_index, np.float64).reshape(-1, 1)),
        "task_index": _stats_for(np.asarray(task_index, np.float64).reshape(-1, 1)),
    }
    (out / schema.STATS_PATH).write_text(
        json.dumps(stats, indent=4) + "\n", encoding="utf-8"
    )

    assert ACTION_DIM == action_arr.shape[1]  # invariant: 8-D action layout
    return out
