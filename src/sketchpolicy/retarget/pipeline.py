"""Retarget a :class:`HandTrack` into an :class:`EEPlan` (experimental adapter).

This wires together aperture->gripper, wrist->orientation and planar position,
plus optional dropped-frame interpolation and Savitzky-Golay smoothing. It is the
*experimental* sketch path: the output is a relative, planar, parallel-jaw
trajectory, not a metric or robot-executable one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sketchpolicy.eeplan import EEPlan
from sketchpolicy.ingest.types import HandTrack
from sketchpolicy.retarget.gripper import aperture_to_gripper
from sketchpolicy.retarget.pose import hand_frame_quaternions, planar_positions
from sketchpolicy.retarget.smoothing import fill_dropped, savgol_smooth


@dataclass(frozen=True)
class RetargetConfig:
    """Tunables for the experimental sketch retarget."""

    x_range: tuple[float, float] = (0.35, 0.60)
    y_range: tuple[float, float] = (-0.15, 0.15)
    z_value: float = 0.15
    z_mode: str = "constant"  # "constant" (honest default) | "depth_heuristic"
    z_range: tuple[float, float] = (0.10, 0.25)
    p_low: float = 5.0
    p_high: float = 95.0
    smooth: bool = True
    smooth_window: int = 9
    smooth_poly: int = 2
    task: str = "sketched trajectory (experimental)"


def retarget_track(track: HandTrack, config: RetargetConfig | None = None) -> EEPlan:
    """Convert a hand track to an EE plan. See :class:`RetargetConfig`."""
    cfg = config or RetargetConfig()
    track = fill_dropped(track)

    gripper = aperture_to_gripper(track, p_low=cfg.p_low, p_high=cfg.p_high)
    quaternions = hand_frame_quaternions(track)
    positions = planar_positions(
        track,
        x_range=cfg.x_range,
        y_range=cfg.y_range,
        z_value=cfg.z_value,
        z_mode=cfg.z_mode,
        z_range=cfg.z_range,
    )

    if cfg.smooth:
        positions = savgol_smooth(positions, cfg.smooth_window, cfg.smooth_poly)
        gripper = np.clip(
            savgol_smooth(gripper, cfg.smooth_window, cfg.smooth_poly), 0.0, 1.0
        )

    return EEPlan(
        positions=positions,
        quaternions=quaternions,
        gripper=gripper,
        timestamps=track.timestamps,
        fps=track.fps,
        task=cfg.task,
        source={"ingest": "mediapipe_sketch_experimental", "z_mode": cfg.z_mode},
    )
