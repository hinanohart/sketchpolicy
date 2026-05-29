"""The shared type boundary: an end-effector trajectory plan.

Every subsystem (augment, emit, replay, retarget, ingest) speaks in terms of
:class:`EEPlan` so that the modules stay decoupled. The canonical action layout
is the 8-D vector ``[x, y, z, qw, qx, qy, qz, gripper]`` (position in metres,
unit quaternion in **wxyz** / scalar-first order, gripper normalised to
``[0, 1]`` where 1 is fully open). There is deliberately **no robot IK** in this
representation: it is an end-effector-space plan, and mapping it onto a specific
embodiment is left to downstream consumers.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator

import numpy as np

#: Index of each action component in the 8-D action vector.
ACTION_DIM = 8
POS_SLICE = slice(0, 3)
QUAT_SLICE = slice(3, 7)  # wxyz (scalar-first)
GRIPPER_INDEX = 7

_QUAT_TOL = 1e-4


@dataclass(frozen=True)
class EEStep:
    """A single end-effector keyframe."""

    position: np.ndarray  # (3,) float64, metres
    quaternion: np.ndarray  # (4,) float64, unit, wxyz
    gripper: float  # [0, 1], 1 == fully open
    timestamp: float  # seconds


@dataclass(frozen=True)
class EEPlan:
    """A time-indexed end-effector trajectory.

    Stored as parallel numpy arrays for vectorised transforms. Construction is
    validated eagerly so that an invalid plan can never enter the pipeline.
    """

    positions: np.ndarray  # (T, 3) float64
    quaternions: np.ndarray  # (T, 4) float64, unit, wxyz
    gripper: np.ndarray  # (T,) float64 in [0, 1]
    timestamps: np.ndarray  # (T,) float64, strictly non-decreasing
    fps: float
    task: str = "sketchpolicy trajectory"
    source: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.validate()

    # -- validation ---------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` if the plan violates the type contract."""
        t = self.positions.shape[0]
        if self.positions.shape != (t, 3):
            raise ValueError(f"positions must be (T, 3), got {self.positions.shape}")
        if self.quaternions.shape != (t, 4):
            raise ValueError(
                f"quaternions must be (T, 4), got {self.quaternions.shape}"
            )
        if self.gripper.shape != (t,):
            raise ValueError(f"gripper must be (T,), got {self.gripper.shape}")
        if self.timestamps.shape != (t,):
            raise ValueError(f"timestamps must be (T,), got {self.timestamps.shape}")
        if t == 0:
            raise ValueError("EEPlan must have at least one frame")
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}")
        norms = np.linalg.norm(self.quaternions, axis=1)
        if not np.allclose(norms, 1.0, atol=_QUAT_TOL):
            raise ValueError("quaternions must be unit-norm")
        if np.any(self.gripper < -_QUAT_TOL) or np.any(self.gripper > 1.0 + _QUAT_TOL):
            raise ValueError("gripper values must lie in [0, 1]")
        if np.any(np.diff(self.timestamps) < -1e-9):
            raise ValueError("timestamps must be non-decreasing")

    # -- conveniences -------------------------------------------------------
    def __len__(self) -> int:
        return int(self.positions.shape[0])

    def __iter__(self) -> Iterator[EEStep]:
        for i in range(len(self)):
            yield EEStep(
                position=self.positions[i],
                quaternion=self.quaternions[i],
                gripper=float(self.gripper[i]),
                timestamp=float(self.timestamps[i]),
            )

    def to_action_array(self) -> np.ndarray:
        """Return the ``(T, 8)`` ``[x,y,z,qw,qx,qy,qz,gripper]`` action array."""
        out = np.empty((len(self), ACTION_DIM), dtype=np.float64)
        out[:, POS_SLICE] = self.positions
        out[:, QUAT_SLICE] = self.quaternions
        out[:, GRIPPER_INDEX] = self.gripper
        return out

    @classmethod
    def from_action_array(
        cls,
        actions: np.ndarray,
        fps: float,
        task: str = "sketchpolicy trajectory",
        timestamps: np.ndarray | None = None,
        source: dict | None = None,
    ) -> "EEPlan":
        """Build a plan from a ``(T, 8)`` action array.

        If ``timestamps`` is omitted, a uniform ``1/fps`` grid is used.
        """
        actions = np.asarray(actions, dtype=np.float64)
        if actions.ndim != 2 or actions.shape[1] != ACTION_DIM:
            raise ValueError(f"actions must be (T, {ACTION_DIM}), got {actions.shape}")
        t = actions.shape[0]
        if timestamps is None:
            timestamps = np.arange(t, dtype=np.float64) / fps
        return cls(
            positions=actions[:, POS_SLICE].copy(),
            quaternions=actions[:, QUAT_SLICE].copy(),
            gripper=actions[:, GRIPPER_INDEX].copy(),
            timestamps=np.asarray(timestamps, dtype=np.float64),
            fps=fps,
            task=task,
            source=dict(source or {}),
        )

    def with_arrays(
        self,
        *,
        positions: np.ndarray | None = None,
        quaternions: np.ndarray | None = None,
        gripper: np.ndarray | None = None,
        source: dict | None = None,
    ) -> "EEPlan":
        """Return a validated copy with selected arrays replaced."""
        return replace(
            self,
            positions=self.positions if positions is None else positions,
            quaternions=self.quaternions if quaternions is None else quaternions,
            gripper=self.gripper if gripper is None else gripper,
            source=self.source if source is None else source,
        )
