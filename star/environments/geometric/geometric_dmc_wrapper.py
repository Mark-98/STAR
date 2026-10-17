"""DMCWrapper subclass that applies one geometric camera view per episode."""

from typing import Optional

import numpy as np

from ._paths import ensure_dmcgb_paths
from .rotations import CameraView, build_camera_views, camera_rotation_matrix


ensure_dmcgb_paths()

from dmc2gym.wrappers import DMCWrapper  # noqa: E402


class GeometricDMCWrapper(DMCWrapper):
    """Apply deterministic episode-level camera orientation changes.

    The current MuJoCo ``cam_xmat`` is read immediately before every render.
    The episode rotation is applied only for that render and the matrix is
    restored in a ``finally`` block. This is important for tracking cameras
    such as ``targetbody`` and ``trackcom``, whose base matrix can change at
    every timestep.
    """

    def __init__(
        self,
        *args,
        camera_rotation_mode="original",
        camera_rotation_seed=1,
        camera_num_views=100,
        rotate_easy_yaw_bound=90.0,
        rotate_hard_yaw_bound=180.0,
        shift_easy_pitch_bound=6.6666667,
        shift_easy_roll_bound=6.6666667,
        shift_hard_pitch_bound=10.0,
        shift_hard_roll_bound=10.0,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._camera_views = build_camera_views(
            mode=camera_rotation_mode,
            seed=camera_rotation_seed,
            num_views=camera_num_views,
            rotate_easy_yaw_bound=rotate_easy_yaw_bound,
            rotate_hard_yaw_bound=rotate_hard_yaw_bound,
            shift_easy_pitch_bound=shift_easy_pitch_bound,
            shift_easy_roll_bound=shift_easy_roll_bound,
            shift_hard_pitch_bound=shift_hard_pitch_bound,
            shift_hard_roll_bound=shift_hard_roll_bound,
        )
        self._camera_rotation_mode = camera_rotation_mode
        self._camera_view_index = 0
        self._episode_camera_view = CameraView(
            yaw_deg=0.0,
            pitch_deg=0.0,
            roll_deg=0.0,
            rotation=camera_rotation_matrix(),
        )

    @property
    def camera_views(self):
        """Return the seed-shuffled episode view sequence."""

        return tuple(self._camera_views)

    @property
    def camera_view_index(self):
        """Return the index that will be selected at the next reset."""

        return self._camera_view_index

    @property
    def episode_camera_view(self):
        """Return the view held fixed for the current episode."""

        return self._episode_camera_view

    def _select_next_camera_view(self):
        self._episode_camera_view = self._camera_views[self._camera_view_index]
        self._camera_view_index = (
            self._camera_view_index + 1
        ) % len(self._camera_views)
        return self._episode_camera_view

    def reset(self):
        # Select first so DMCWrapper.reset()'s initial pixel observation uses
        # the same view as every later render in this episode.
        self._select_next_camera_view()
        return super().reset()

    @staticmethod
    def _read_camera_matrix(cam_xmat, camera_id):
        storage = np.asarray(cam_xmat[camera_id])
        if storage.size != 9 or storage.shape not in {(9,), (3, 3)}:
            raise ValueError(
                "cam_xmat entries must have shape (9,) or (3, 3); "
                f"got {storage.shape} for camera_id={camera_id}"
            )
        return storage.copy(), storage.reshape(3, 3).copy()

    @staticmethod
    def _write_camera_matrix(cam_xmat, camera_id, matrix, storage_shape):
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"camera matrix must have shape (3, 3), got {matrix.shape}")
        cam_xmat[camera_id] = matrix.reshape(storage_shape)

    def render(
        self,
        mode="rgb_array",
        height=None,
        width=None,
        camera_id: Optional[int] = None,
    ):
        selected_camera_id = self._camera_id if camera_id is None else camera_id
        data = self._env.physics.data
        cam_xmat = data.cam_xmat
        original_storage, current_xmat = self._read_camera_matrix(
            cam_xmat, selected_camera_id
        )
        rotated_xmat = current_xmat @ self._episode_camera_view.rotation

        # The vendored DMCWrapper uses ``camera_id or self._camera_id``. Keep
        # camera_id=0 valid even when the wrapper's configured default differs.
        original_default_camera_id = self._camera_id
        if selected_camera_id == 0:
            self._camera_id = 0

        try:
            self._write_camera_matrix(
                cam_xmat,
                selected_camera_id,
                rotated_xmat,
                original_storage.shape,
            )
            return super().render(
                mode=mode,
                height=height,
                width=width,
                camera_id=selected_camera_id,
            )
        finally:
            self._write_camera_matrix(
                cam_xmat,
                selected_camera_id,
                original_storage.reshape(3, 3),
                original_storage.shape,
            )
            self._camera_id = original_default_camera_id
