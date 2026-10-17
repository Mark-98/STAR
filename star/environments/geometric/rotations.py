"""Deterministic DMC-GB2 geometric camera view generation.

Axes follow the explicit convention requested by DMC-GB2:

* yaw rotates about Z,
* pitch rotates about Y,
* roll rotates about X,
* rotations compose as ``Rz(yaw) @ Ry(pitch) @ Rx(roll)``.

``original`` returns one identity view. Every other mode returns
``num_views`` views and shuffles their episode order once with ``seed``.
"""

from dataclasses import dataclass
from typing import List

import numpy as np


VALID_CAMERA_MODES = {
    "original",
    "rotate_easy",
    "rotate_hard",
    "shift_easy",
    "shift_hard",
}


@dataclass(frozen=True)
class CameraView:
    """Camera angles and the corresponding local rotation matrix."""

    yaw_deg: float
    pitch_deg: float
    roll_deg: float
    rotation: np.ndarray


def camera_rotation_matrix(
    yaw_deg: float = 0.0,
    pitch_deg: float = 0.0,
    roll_deg: float = 0.0,
) -> np.ndarray:
    """Return ``Rz(yaw) @ Ry(pitch) @ Rx(roll)`` as a float64 matrix."""

    yaw, pitch, roll = np.deg2rad(
        np.asarray([yaw_deg, pitch_deg, roll_deg], dtype=np.float64)
    )

    cos_yaw, sin_yaw = np.cos(yaw), np.sin(yaw)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)

    rotate_yaw = np.asarray(
        [
            [cos_yaw, -sin_yaw, 0.0],
            [sin_yaw, cos_yaw, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rotate_pitch = np.asarray(
        [
            [cos_pitch, 0.0, sin_pitch],
            [0.0, 1.0, 0.0],
            [-sin_pitch, 0.0, cos_pitch],
        ],
        dtype=np.float64,
    )
    rotate_roll = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, cos_roll, -sin_roll],
            [0.0, sin_roll, cos_roll],
        ],
        dtype=np.float64,
    )
    return rotate_yaw @ rotate_pitch @ rotate_roll


def _make_view(yaw_deg: float, pitch_deg: float, roll_deg: float) -> CameraView:
    return CameraView(
        yaw_deg=float(yaw_deg),
        pitch_deg=float(pitch_deg),
        roll_deg=float(roll_deg),
        rotation=camera_rotation_matrix(yaw_deg, pitch_deg, roll_deg),
    )


def _perimeter_samples(
    num_views: int,
    pitch_bound: float,
    roll_bound: float,
) -> np.ndarray:
    if num_views % 4 != 0:
        raise ValueError("shift modes require num_views divisible by 4")

    corners = np.asarray(
        [
            [-pitch_bound, -roll_bound],
            [pitch_bound, -roll_bound],
            [pitch_bound, roll_bound],
            [-pitch_bound, roll_bound],
        ],
        dtype=np.float64,
    )
    points_per_edge = num_views // 4
    edges = []
    for index in range(4):
        edges.append(
            np.linspace(
                corners[index],
                corners[(index + 1) % 4],
                points_per_edge,
                endpoint=False,
                dtype=np.float64,
            )
        )
    return np.concatenate(edges, axis=0)


def build_camera_views(
    mode: str,
    seed: int,
    num_views: int = 100,
    rotate_easy_yaw_bound: float = 90.0,
    rotate_hard_yaw_bound: float = 180.0,
    shift_easy_pitch_bound: float = 6.6666667,
    shift_easy_roll_bound: float = 6.6666667,
    shift_hard_pitch_bound: float = 10.0,
    shift_hard_roll_bound: float = 10.0,
) -> List[CameraView]:
    """Build and seed-shuffle geometric views for one camera mode.

    Rotate modes sample yaw on a half-open interval. Shift modes sample equal
    numbers of points from the four edges of the pitch-roll rectangle; they do
    not sample its interior. The returned order is the episode order.
    """

    if mode not in VALID_CAMERA_MODES:
        valid = ", ".join(sorted(VALID_CAMERA_MODES))
        raise ValueError(f"unsupported camera rotation mode {mode!r}; expected one of: {valid}")
    if num_views <= 0:
        raise ValueError("num_views must be positive")

    if mode == "original":
        views = [_make_view(0.0, 0.0, 0.0)]
    elif mode in {"rotate_easy", "rotate_hard"}:
        bound = (
            float(rotate_easy_yaw_bound)
            if mode == "rotate_easy"
            else float(rotate_hard_yaw_bound)
        )
        yaw_values = np.linspace(
            -bound,
            bound,
            num=num_views,
            endpoint=False,
            dtype=np.float64,
        )
        views = [_make_view(yaw, 0.0, 0.0) for yaw in yaw_values]
    else:
        if mode == "shift_easy":
            pitch_bound = float(shift_easy_pitch_bound)
            roll_bound = float(shift_easy_roll_bound)
        else:
            pitch_bound = float(shift_hard_pitch_bound)
            roll_bound = float(shift_hard_roll_bound)
        pitch_roll_values = _perimeter_samples(
            num_views=num_views,
            pitch_bound=pitch_bound,
            roll_bound=roll_bound,
        )
        views = [
            _make_view(0.0, pitch, roll)
            for pitch, roll in pitch_roll_values
        ]

    rng = np.random.RandomState(seed)
    rng.shuffle(views)
    return views
