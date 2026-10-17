# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import cv2
import imageio
import numpy as np


class VideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20, camera_id=0, camera_name='agentview'):
        if root_dir is not None:
            self.save_dir = root_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

        self.camera_id = camera_id
        self.camera_name = camera_name

    def set_camera(self, camera_id=None, camera_name=None):
        if camera_id is not None:
            self.camera_id = camera_id
        if camera_name is not None:
            self.camera_name = camera_name

    def init(self, env, enabled=True, camera_name=None):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if camera_name is not None:
            self.camera_name = camera_name
        self.record(env)

    def init_dmc(self, env, enabled=True, camera_id=None):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        if camera_id is not None:
            self.camera_id = camera_id
        self.record_dmc(env, video=True)

    def record(self, env, height=84, width=84):
        if self.enabled:
            if hasattr(env, 'physics'):
                frame = env.physics.render(
                    height=self.render_size,
                    width=self.render_size,
                    camera_id=self.camera_id
                )
            else:
                #  robosuite: use synced camera_name instead of hard-coded 'agentview'
                cam = self.camera_name if self.camera_name is not None else 'agentview'
                frame = env._gym_env.env.sim.render(
                    mode="offscreen",
                    width=width,
                    height=height,
                    camera_name=cam
                )
            return frame[::-1, :, :]

    def record_frame(self, obs, height=224, width=224, dsize=(252, 84)):
        if self.enabled:
            frame = cv2.resize(obs, dsize=dsize, interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def record_dmc(self, env, video=False, height=84, width=84):
        if self.enabled:
            if hasattr(env, 'physics'):
                # use synced camera_id instead of hard-coded 0
                frame = env.physics.render(
                    height=self.render_size,
                    width=self.render_size,
                    camera_id=self.camera_id
                )
            else:
                if video:
                    frame = env._env._env._env._gym_env.env.env.render(height=height, width=width)
                else:
                    frame = env.render()
            return frame


    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)


class TrainVideoRecorder:
    def __init__(self, root_dir, render_size=256, fps=20):
        if root_dir is not None:
            self.save_dir = root_dir / 'train_video'
            self.save_dir.mkdir(exist_ok=True)
        else:
            self.save_dir = None

        self.render_size = render_size
        self.fps = fps
        self.frames = []

    def init(self, obs, enabled=True):
        self.frames = []
        self.enabled = self.save_dir is not None and enabled
        self.record(obs)

    def record(self, obs):
        if self.enabled:
            frame = cv2.resize(obs[-3:].transpose(1, 2, 0),
                               dsize=(self.render_size, self.render_size),
                               interpolation=cv2.INTER_CUBIC)
            self.frames.append(frame)

    def save(self, file_name):
        if self.enabled:
            path = self.save_dir / file_name
            imageio.mimsave(str(path), self.frames, fps=self.fps)
