"""Headless kinematic replay sanity-check (optional ``[replay]`` extra).

This loads a generic 7-DOF arm in a headless pybullet (``DIRECT``) session and
tracks the end-effector trajectory by inverse kinematics, reporting the per-frame
position residual between the IK solution's achieved pose and the target pose.

It is a **coarse reachability sanity signal, not a guarantee** for any specific
robot: a large residual means "no generic arm could follow this here", a small
residual does not certify your embodiment can. pybullet and its bundled URDFs are
loaded at runtime from ``pybullet_data``; sketchpolicy does not redistribute them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sketchpolicy.eeplan import EEPlan


class ReplayUnavailable(RuntimeError):
    """Raised when the optional pybullet dependency is not installed."""


@dataclass(frozen=True)
class ReplayReport:
    """Outcome of a kinematic replay."""

    n_frames: int
    max_position_residual_m: float
    mean_position_residual_m: float
    reachable_fraction: float  # fraction of frames with residual < threshold
    threshold_m: float


def _quat_wxyz_to_xyzw(q: np.ndarray) -> list[float]:
    return [float(q[1]), float(q[2]), float(q[3]), float(q[0])]


def replay_plan(
    plan: EEPlan,
    *,
    residual_threshold_m: float = 0.05,
    ik_iters: int = 50,
) -> ReplayReport:
    """Replay ``plan`` headlessly and report IK position residuals.

    Args:
        plan: the EE trajectory to track.
        residual_threshold_m: residual below which a frame counts as reachable.
        ik_iters: IK refinement iterations per frame.

    Raises:
        ReplayUnavailable: if pybullet is not installed (``pip install
            'sketchpolicy[replay]'``).
    """
    try:
        import pybullet as p
        import pybullet_data
    except ImportError as exc:  # pragma: no cover - exercised via extra
        raise ReplayUnavailable(
            "pybullet is required for replay; install with "
            "`pip install 'sketchpolicy[replay]'`"
        ) from exc

    cid = p.connect(p.DIRECT)
    try:
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        robot = p.loadURDF("kuka_iiwa/model.urdf", useFixedBase=True)
        ee_link = p.getNumJoints(robot) - 1

        residuals = np.empty(len(plan), dtype=np.float64)
        for i, step in enumerate(plan):
            target_pos = [float(x) for x in step.position]
            target_orn = _quat_wxyz_to_xyzw(step.quaternion)
            joints = p.calculateInverseKinematics(
                robot,
                ee_link,
                target_pos,
                target_orn,
                maxNumIterations=ik_iters,
            )
            for j, q in enumerate(joints):
                p.resetJointState(robot, j, q)
            achieved = np.asarray(p.getLinkState(robot, ee_link)[4], dtype=np.float64)
            residuals[i] = np.linalg.norm(achieved - np.asarray(target_pos))

        reachable = float(np.mean(residuals < residual_threshold_m))
        return ReplayReport(
            n_frames=len(plan),
            max_position_residual_m=float(residuals.max()),
            mean_position_residual_m=float(residuals.mean()),
            reachable_fraction=reachable,
            threshold_m=residual_threshold_m,
        )
    finally:
        p.disconnect(cid)
