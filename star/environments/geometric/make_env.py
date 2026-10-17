"""Opt-in DMCVGB factory using :class:`GeometricDMCWrapper` directly."""

import math
import os

from ._paths import ensure_dmcgb_paths
from .geometric_dmc_wrapper import GeometricDMCWrapper


ensure_dmcgb_paths()

import gym  # noqa: E402
from dmcvgb import utils  # noqa: E402
from dmcvgb.wrappers import ColorWrapper, FrameStack, VideoWrapper  # noqa: E402


def make_env(
    domain_name,
    task_name,
    seed,
    action_repeat=2,
    frame_stack=3,
    type="original",
    difficulty="easy",
    obj_color="original",
    cam_pos="original",
    episode_length=1000,
    image_size=84,
    cam_id=0,
    is_distracting_cs=False,
    intensity=0,
    light_position="original",
    light_color="original",
    moving_light="original",
    table_texture="original",
    camera_rotation_mode="original",
    camera_rotation_seed=None,
    camera_num_views=100,
    rotate_easy_yaw_bound=90.0,
    rotate_hard_yaw_bound=180.0,
    shift_easy_pitch_bound=6.6666667,
    shift_easy_roll_bound=6.6666667,
    shift_hard_pitch_bound=10.0,
    shift_hard_roll_bound=10.0,
):
    """Create the standard DMCVGB stack with geometric camera rotation.

    This mirrors ``dmcvgb.make_env`` without calling the Gym registry. The
    only base-environment difference is direct construction of
    :class:`GeometricDMCWrapper` in place of ``DMCWrapper``.
    """

    paths = []
    if is_distracting_cs:
        loaded_paths = [
            os.path.join(directory, "DAVIS/JPEGImages/480p")
            for directory in utils.load_config("datasets")
        ]
        paths = [path for path in loaded_paths if os.path.exists(path)]

    task_kwargs = {
        "random": seed,
        # Preserve the effectively unbounded dm_control limit used by the
        # existing dmc2gym factory; Gym's TimeLimit enforces episode_length.
        "time_limit": 1e6,
    }
    base_env = GeometricDMCWrapper(
        domain_name=domain_name,
        task_name=task_name,
        task_kwargs=task_kwargs,
        visualize_reward=False,
        from_pixels=True,
        height=image_size,
        width=image_size,
        camera_id=cam_id,
        frame_skip=action_repeat,
        is_distracting_cs=is_distracting_cs,
        distracting_cs_intensity=intensity,
        background_dataset_paths=paths,
        channels_first=True,
        seed=seed,
        cam_pos=cam_pos,
        camera_rotation_mode=camera_rotation_mode,
        camera_rotation_seed=(
            seed if camera_rotation_seed is None else camera_rotation_seed
        ),
        camera_num_views=camera_num_views,
        rotate_easy_yaw_bound=rotate_easy_yaw_bound,
        rotate_hard_yaw_bound=rotate_hard_yaw_bound,
        shift_easy_pitch_bound=shift_easy_pitch_bound,
        shift_easy_roll_bound=shift_easy_roll_bound,
        shift_hard_pitch_bound=shift_hard_pitch_bound,
        shift_hard_roll_bound=shift_hard_roll_bound,
    )

    # gym.make() normally adds these two wrappers. Recreate them explicitly so
    # the existing VideoWrapper/ColorWrapper traversal remains unchanged.
    env = gym.wrappers.OrderEnforcing(base_env)
    env = gym.wrappers.TimeLimit(
        env,
        max_episode_steps=int(math.ceil(episode_length / action_repeat)),
    )

    background = {"type": type, "difficulty": difficulty}
    if not is_distracting_cs:
        env = VideoWrapper(env, background, seed, obj_color, cam_pos)
    env = FrameStack(env, frame_stack)
    if not is_distracting_cs:
        env = ColorWrapper(
            env,
            background,
            seed,
            obj_color,
            table_texture,
            light_position,
            light_color,
            moving_light,
            cam_pos,
        )
    return env
