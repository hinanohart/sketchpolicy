"""The SE(3) + time-warp transform group applied to an :class:`EEPlan`.

A variant is parameterised by a small vector:

* ``azimuth``   - yaw about the world z-axis, around the trajectory centroid;
* ``elevation`` - pitch about the world y-axis, around the trajectory centroid;
* ``dx, dy``    - planar translation in the table plane;
* ``time_gamma``- a monotonic power re-timing of the trajectory phase, which
  changes the velocity profile while preserving the endpoints.

The rotation/translation part is a genuine rigid SE(3) body transform: positions
are rotated about the trajectory centroid (so the task stays near its workspace)
then translated, and orientations are composed in the world frame. The time-warp
resamples positions/gripper linearly and orientations via SLERP, so the output
keeps exactly the original frame count and remains a valid plan.

Determinism: every operation here is pure numpy/scipy with no global RNG, so a
given parameter vector always yields a bit-identical plan.
"""

from __future__ import annotations

from dataclasses import astuple, dataclass

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from sketchpolicy.eeplan import EEPlan

#: Order of the parameter vector consumed by :func:`apply`.
PARAM_NAMES: tuple[str, ...] = ("azimuth", "elevation", "dx", "dy", "time_gamma")
N_PARAMS = len(PARAM_NAMES)


@dataclass(frozen=True)
class TransformParams:
    """A single point in the augmentation parameter space."""

    azimuth: float
    elevation: float
    dx: float
    dy: float
    time_gamma: float

    def as_vector(self) -> np.ndarray:
        return np.asarray(astuple(self), dtype=np.float64)

    @classmethod
    def from_vector(cls, v: np.ndarray) -> "TransformParams":
        if v.shape != (N_PARAMS,):
            raise ValueError(f"param vector must be ({N_PARAMS},), got {v.shape}")
        return cls(*(float(x) for x in v))


@dataclass(frozen=True)
class ParamRanges:
    """Box bounds for the parameter space (closed-form, profile-independent)."""

    azimuth: tuple[float, float] = (-np.pi / 6, np.pi / 6)  # ±30°
    elevation: tuple[float, float] = (-np.pi / 12, np.pi / 12)  # ±15°
    dx: tuple[float, float] = (-0.08, 0.08)  # ±8 cm
    dy: tuple[float, float] = (-0.08, 0.08)
    time_gamma: tuple[float, float] = (0.8, 1.25)

    def lows_highs(self) -> tuple[np.ndarray, np.ndarray]:
        lows = np.array(
            [
                self.azimuth[0],
                self.elevation[0],
                self.dx[0],
                self.dy[0],
                self.time_gamma[0],
            ]
        )
        highs = np.array(
            [
                self.azimuth[1],
                self.elevation[1],
                self.dx[1],
                self.dy[1],
                self.time_gamma[1],
            ]
        )
        return lows, highs

    def scale_unit_cube(self, u: np.ndarray) -> np.ndarray:
        """Map points from ``[0, 1)^N`` (shape ``(m, N)``) into the box."""
        lows, highs = self.lows_highs()
        return lows + u * (highs - lows)


def _quat_wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    return q[..., [1, 2, 3, 0]]


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def _rigid_part(params: TransformParams) -> Rotation:
    """The world-frame rotation = elevation (about y) then azimuth (about z)."""
    return Rotation.from_euler("zy", [params.azimuth, params.elevation])


def _apply_rigid(plan: EEPlan, params: TransformParams) -> EEPlan:
    """Apply the SE(3) rotation (about the trajectory centroid) and translation."""
    rot = _rigid_part(params)
    pivot = plan.positions.mean(axis=0)
    translation = np.array([params.dx, params.dy, 0.0])

    new_pos = rot.apply(plan.positions - pivot) + pivot + translation

    q_xyzw = _quat_wxyz_to_xyzw(plan.quaternions)
    composed = (rot * Rotation.from_quat(q_xyzw)).as_quat()  # (T, 4) xyzw
    new_quat = _quat_xyzw_to_wxyz(composed)
    # Canonicalise sign (w >= 0) so determinism does not depend on scipy's
    # internal quaternion sign convention.
    flip = new_quat[:, 0] < 0
    new_quat[flip] = -new_quat[flip]
    return plan.with_arrays(positions=new_pos, quaternions=new_quat)


def _apply_time_warp(plan: EEPlan, gamma: float) -> EEPlan:
    """Re-time the trajectory with a monotonic power warp of its phase.

    The endpoints are preserved; intermediate frames are resampled (linear for
    position/gripper, SLERP for orientation) so the frame count is unchanged.
    """
    if abs(gamma - 1.0) < 1e-9:
        return plan

    t = len(plan)
    if t < 2:
        return plan

    # Warp in normalised frame-parameter space, not timestamp space. The phase
    # axis ``linspace(0, 1, t)`` is always strictly increasing, so SLERP never
    # sees duplicate key times even when the source dataset has repeated
    # timestamps (e.g. a paused demo) -- which EEPlan permits (non-decreasing).
    phase = np.linspace(0.0, 1.0, t)
    warped = phase**gamma  # query points, monotonic in [0, 1], endpoints fixed

    # Linear resample of position and gripper along the phase axis.
    new_pos = np.empty_like(plan.positions)
    for c in range(3):
        new_pos[:, c] = np.interp(warped, phase, plan.positions[:, c])
    new_gripper = np.interp(warped, phase, plan.gripper)

    # SLERP resample of orientation against the strictly-increasing phase axis.
    q_xyzw = _quat_wxyz_to_xyzw(plan.quaternions)
    slerp = Slerp(phase, Rotation.from_quat(q_xyzw))
    warped_clamped = np.clip(warped, 0.0, 1.0)  # guard float overshoot at ends
    new_quat = _quat_xyzw_to_wxyz(slerp(warped_clamped).as_quat())
    flip = new_quat[:, 0] < 0
    new_quat[flip] = -new_quat[flip]

    return plan.with_arrays(
        positions=new_pos, quaternions=new_quat, gripper=new_gripper
    )


def apply(plan: EEPlan, params: TransformParams) -> EEPlan:
    """Apply the full transform (rigid SE(3) then time-warp) to ``plan``.

    The result records the applied parameters in ``source['augment_params']``.
    """
    out = _apply_rigid(plan, params)
    out = _apply_time_warp(out, params.time_gamma)
    source = dict(plan.source)
    source["augment_params"] = dict(zip(PARAM_NAMES, params.as_vector().tolist()))
    return out.with_arrays(source=source)
