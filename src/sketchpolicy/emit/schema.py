"""The LeRobotDataset v3.0 on-disk schema, replicated for a torch-free writer.

The constants and templates here mirror the public v3.0 layout (verified against
the ``lerobot-dataset`` wheel source at build time). We deliberately reproduce
*the format*, not the writer: sketchpolicy claims its output is **schema-valid**
(round-trips against a pinned fixture and, optionally, the real reader), not
byte-identical to lerobot's own serialiser. Keeping this in pure pyarrow is what
lets the core install stay torch-free.
"""

from __future__ import annotations

from sketchpolicy.eeplan import ACTION_DIM

#: The v3.0 codebase version tag, grepped from the lerobot-dataset wheel source
#: at build time (``ledataset.datasets.lerobot_dataset.CODEBASE_VERSION``). When
#: the optional ``[lerobot]`` extra is installed, tests/test_lerobot_compat.py
#: asserts this equals the installed value and loads our output with the real
#: reader; that test self-skips otherwise.
CODEBASE_VERSION = "v3.0"

# -- directory layout (mirrors ledataset.datasets.utils) --------------------
INFO_PATH = "meta/info.json"
STATS_PATH = "meta/stats.json"
TASKS_PATH = "meta/tasks.parquet"
EPISODES_DIR = "meta/episodes"
DATA_DIR = "data"
VIDEO_DIR = "videos"
CHUNK_FILE_PATTERN = "chunk-{chunk_index:03d}/file-{file_index:03d}"
DATA_PATH = DATA_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"
VIDEO_PATH = VIDEO_DIR + "/{video_key}/" + CHUNK_FILE_PATTERN + ".mp4"
EPISODES_PATH = EPISODES_DIR + "/" + CHUNK_FILE_PATTERN + ".parquet"

DEFAULT_CHUNK_SIZE = 1000
DEFAULT_DATA_FILE_SIZE_IN_MB = 100
DEFAULT_VIDEO_FILE_SIZE_IN_MB = 200

#: Component names for the 8-D end-effector action / state vector.
ACTION_NAMES = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]

#: Per-frame "default" features that every v3.0 dataset carries (stored as
#: scalar columns, declared with shape (1,) by convention).
DEFAULT_FEATURES: dict[str, dict] = {
    "timestamp": {"dtype": "float32", "shape": [1], "names": None},
    "frame_index": {"dtype": "int64", "shape": [1], "names": None},
    "episode_index": {"dtype": "int64", "shape": [1], "names": None},
    "index": {"dtype": "int64", "shape": [1], "names": None},
    "task_index": {"dtype": "int64", "shape": [1], "names": None},
}


def ee_features() -> dict[str, dict]:
    """Return the features dict for an EE-action/state v3.0 dataset."""
    feats: dict[str, dict] = {
        "action": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": list(ACTION_NAMES),
        },
        "observation.state": {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": list(ACTION_NAMES),
        },
    }
    feats.update({k: dict(v) for k, v in DEFAULT_FEATURES.items()})
    return feats


def build_info(
    *,
    fps: int,
    features: dict,
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    robot_type: str | None = "sketchpolicy_ee",
) -> dict:
    """Build the ``info.json`` dict for a finalised v3.0 dataset (no videos)."""
    return {
        "codebase_version": CODEBASE_VERSION,
        "robot_type": robot_type,
        "total_episodes": total_episodes,
        "total_frames": total_frames,
        "total_tasks": total_tasks,
        "chunks_size": DEFAULT_CHUNK_SIZE,
        "data_files_size_in_mb": DEFAULT_DATA_FILE_SIZE_IN_MB,
        "video_files_size_in_mb": DEFAULT_VIDEO_FILE_SIZE_IN_MB,
        "fps": fps,
        "splits": {"train": f"0:{total_episodes}"},
        "data_path": DATA_PATH,
        "video_path": None,
        "features": features,
    }


#: The set of info.json keys a valid v3.0 dataset must carry.
REQUIRED_INFO_KEYS = frozenset(
    {
        "codebase_version",
        "robot_type",
        "total_episodes",
        "total_frames",
        "total_tasks",
        "chunks_size",
        "data_files_size_in_mb",
        "video_files_size_in_mb",
        "fps",
        "splits",
        "data_path",
        "video_path",
        "features",
    }
)
