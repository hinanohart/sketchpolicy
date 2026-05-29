"""Wrist orientation (Gram-Schmidt) and planar position retargeting.

Orientation comes from the metric, hand-centred ``world_landmarks``: a hand frame
is built by Gram-Schmidt from the "forward" axis (middle-MCP minus wrist) and the
"across" axis (index-MCP minus pinky-MCP), then converted to a unit quaternion.

Position is *planar by default*: monocular landmarks carry no absolute metric
scale, so the wrist's normalised image motion is mapped into a configurable
workspace box on the table and z is held constant. A relative depth heuristic is
available but is explicitly experimental.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.transform import Rotation

from sketchpolicy.ingest.types import (
    INDEX_MCP,
    MIDDLE_MCP,
    PINKY_MCP,
    WRIST,
    HandTrack,
)


def _quat_xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    return q[..., [3, 0, 1, 2]]


def hand_frame_quaternions(track: HandTrack) -> np.ndarray:
    """Per-frame orientation as unit quaternions ``(T, 4)`` in wxyz order."""
    w = track.world_landmarks
    forward = w[:, MIDDLE_MCP, :] - w[:, WRIST, :]
    across = w[:, INDEX_MCP, :] - w[:, PINKY_MCP, :]

    quats = np.empty((len(track), 4))
    for i in range(len(track)):
        y = forward[i]
        ny = np.linalg.norm(y)
        if ny < 1e-9:
            quats[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        y = y / ny
        x = across[i] - np.dot(across[i], y) * y  # Gram-Schmidt orthogonalise
        nx = np.linalg.norm(x)
        if nx < 1e-9:
            quats[i] = [1.0, 0.0, 0.0, 0.0]
            continue
        x = x / nx
        z = np.cross(x, y)  # right-handed: det == +1 by construction
        rot = np.column_stack([x, y, z])
        q_xyzw = Rotation.from_matrix(rot).as_quat()
        q = _quat_xyzw_to_wxyz(q_xyzw)
        if q[0] < 0:  # canonical sign (w >= 0)
            q = -q
        quats[i] = q
    return quats


def planar_positions(
    track: HandTrack,
    *,
    x_range: tuple[float, float] = (0.35, 0.60),
    y_range: tuple[float, float] = (-0.15, 0.15),
    z_value: float = 0.15,
    z_mode: str = "constant",
    z_range: tuple[float, float] = (0.10, 0.25),
) -> np.ndarray:
    """Map the wrist's normalised image motion to a workspace trajectory ``(T, 3)``.

    ``z_mode='constant'`` (default, honest) holds z fixed. ``z_mode='depth_heuristic'``
    is **experimental**: it maps the wrist's relative image-z into ``z_range`` and
    must not be read as metric depth.
    """
    wrist_img = track.image_landmarks[:, WRIST, :]  # (T, 3), normalised
    # image x in [0,1] -> x_range; image y in [0,1] (top-down) -> y_range.
    x = x_range[0] + np.clip(wrist_img[:, 0], 0.0, 1.0) * (x_range[1] - x_range[0])
    y = y_range[0] + np.clip(wrist_img[:, 1], 0.0, 1.0) * (y_range[1] - y_range[0])

    if z_mode == "constant":
        z = np.full(len(track), z_value)
    elif z_mode == "depth_heuristic":
        zr = wrist_img[:, 2]
        span = zr.max() - zr.min()
        norm = (zr - zr.min()) / span if span > 1e-9 else np.zeros_like(zr)
        z = z_range[0] + norm * (z_range[1] - z_range[0])
    else:
        raise ValueError(f"unknown z_mode {z_mode!r}")
    return np.column_stack([x, y, z])
