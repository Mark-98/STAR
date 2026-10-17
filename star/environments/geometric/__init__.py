"""DMC-GB2 style geometric camera generalization for DMCVGB.

The extension is opt-in. Importing it does not alter the existing RL-ViGen
environment factory or any registered Gym environment.
"""

from .rotations import (
    VALID_CAMERA_MODES,
    CameraView,
    build_camera_views,
    camera_rotation_matrix,
)

__all__ = [
    "VALID_CAMERA_MODES",
    "CameraView",
    "build_camera_views",
    "camera_rotation_matrix",
]
