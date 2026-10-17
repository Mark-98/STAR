from star.rl.checkpoint import load_checkpoint
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ.setdefault('MUJOCO_GL', 'egl')
try:
    import quaternion
except:
    pass
from pathlib import Path
import sys
import time

import hydra
import numpy as np
import torch
from dm_env import specs
from tqdm import tqdm

import star.rl.wrappers.dmc as dmc
from star.rl import utils
from star.rl.logger import Logger
from star.rl.wandblogger import WandbLogger
from star.rl.replay_buffer import ReplayBufferStorage, make_replay_loader
from star.rl.video import TrainVideoRecorder, VideoRecorder
from omegaconf import OmegaConf
from star.rl.arguments import parse_args
from star.rl.algos.factory import make_agent
from star.rl.read_buffer.split_buffer import merge_npz

torch.backends.cudnn.benchmark = True


def robo_make(*args, **kwargs):
    from star.rl.wrappers.robo_wrapper import robo_make as make
    return make(*args, **kwargs)


def _format_duration(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f'{hours:02d}:{minutes:02d}:{seconds:02d}'

class Workspace:
    def __init__(self, cfg):
        if cfg.env == 'robosuite':
            # Import before seeding, as in the original entrypoint. DMC does not
            # need the optional Robosuite/mujoco-py installation.
            import star.rl.wrappers.robo_wrapper
        self.work_dir = Path.cwd()
        (self.work_dir / 'logs' / cfg.exp_name).mkdir(parents=True, exist_ok=False)
        self.model_dir = Path(utils.make_dir(os.path.join(self.work_dir, 'logs', cfg.exp_name,'model')))
        self.data_dir = Path(utils.make_dir(os.path.join(self.work_dir, 'logs', cfg.exp_name,'buffer')))
        self.video_dir = Path(utils.make_dir(os.path.join(self.work_dir, 'logs', cfg.exp_name, 'video')))
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device("cuda")
        self.setup(self.cfg.env)

        self.agent = make_agent(self.train_env.observation_spec().shape,
                                self.train_env.action_spec().shape[0],
                                self.cfg)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0
        self._mechanism = None
        if getattr(cfg, 'mechanism_config', None):
            raise NotImplementedError('Experimental diagnostic hooks are not included in this release.')


    def setup(self, env):

        self.wandblogger = WandbLogger(self.cfg)
        self.replay_dir = Path(os.path.join(self.work_dir , self.cfg.exp_name,"replay"))
        assert env in ['dmc', 'robosuite', 'habitat']
        self.env_name = env
        # create envs
        if env == 'dmc':
            import star.rl.wrappers.loco_wrapper as dmc
            self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed)
            self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                    self.cfg.action_repeat, self.cfg.seed)
            self.eval_color_easy_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, type='color', difficulty='easy', object_color='easy')
            self.eval_color_hard_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, type='color', difficulty='hard', object_color='hard')
            self.eval_video_easy_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, type='video', difficulty='easy')
            self.eval_video_hard_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed, type='video', difficulty='hard')
        elif env == 'robosuite':
            self.train_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                       frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='train')
            self.eval_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                      frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval')
            self.eval_easy_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                      frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-easy')
            self.eval_medium_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                      frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-medium')
            self.eval_hard_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                      frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-hard')

        else:
            raise ValueError(f"env {env} not supported.")

        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, 'reward'),
                        specs.Array((1,), np.float32, 'discount'))

        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.data_dir)

        self.replay_loader = make_replay_loader(
            self.data_dir , self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount, self.cfg.buffer_save, False, reward_scale=self.cfg.reward_scale)
        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.video_dir if self.cfg.save_video else None,
            camera_id=0,
            camera_name=self.cfg.robo_camera_name
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.video_dir if self.cfg.save_train_video else None)

    @property
    def global_step(self):
        return self._global_step


    @property
    def global_episode(self):
        return self._global_episode


    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat


    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter


    def eval(self):
        step, episode, total_reward = 0, 0, 0
        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]
        # robosuite   eval_env . dmc  self.eval_env .
        eval_env = None
        close_after = None

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                #

                scene_id = episode % 10  # scene_id 0..9
                if self.env_name == 'robosuite':
                    eval_env = robo_make(
                        name=self.cfg.task_name,
                        action_repeat=self.cfg.action_repeat,
                        frame_stack=self.cfg.frame_stack,
                        seed=self.cfg.seed,
                        scene_id=scene_id,
                        mode='eval'
                    )
                    close_after = getattr(eval_env, "close", None)
                if self.env_name == 'robosuite':
                    time_step = eval_env.reset()
                else:  # dmc
                    time_step = self.eval_env.reset()

                record = (episode == 0)
                if self.env_name == 'dmc':
                    self.video_recorder.init_dmc(self.eval_env, enabled=record)
                else:
                    self.video_recorder.init(eval_env, enabled=record)
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.multi_act:
                            action = self.agent.multi_act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True,
                                            decoding=False,
                                            )
                        else:
                            action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True,
                                                decoding=False)
                    if self.env_name == 'robosuite':
                        time_step = eval_env.step(action)
                    else:  # dmc
                        time_step = self.eval_env.step(action)
                    if record:
                        if self.env_name == 'dmc':
                            original = self.video_recorder.record_dmc(self.eval_env, video=True)
                            self.video_recorder.record_frame(original, dsize=(96, 96))

                        elif self.env_name == 'robosuite':
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                if record:
                    self.video_recorder.save(f'{self.global_frame}.mp4')
            avg_reward = total_reward / episode
            avg_length = step * self.cfg.action_repeat / episode

            self.wandblogger.scalar_summary("eval/episode_reward", avg_reward, self.global_frame)


        finally:
            # robosuite
            if close_after is not None and callable(close_after):
                try:
                    close_after()
                except Exception:
                    pass
            #
            eval_env = None


    def eval_video_easy(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_video_easy")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        do_video = (self.cfg.algo == 'dad')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        while eval_until_episode(episode):
            time_step = self.eval_video_easy_env.reset()

            # dad
            record = do_video and (episode == 0)

            if self.env_name == 'dmc':
                if record:
                    self.video_recorder.init_dmc(self.eval_video_easy_env, enabled=True)
            else:
                print("Wrong env")
                break

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if do_video:
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False
                        )
                    else:
                        # not dad: action  (/ X)
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False
                        )

                time_step = self.eval_video_easy_env.step(action)

                # dad
                if self.env_name == 'dmc' and record:
                    original = self.video_recorder.record_dmc(
                        self.eval_video_easy_env, video=True, height=84, width=84
                    )
                    self.video_recorder.record_frame(original, dsize=(96, 96))

                total_reward += time_step.reward
                step += 1

            episode += 1

            # dad
            if record:
                self.video_recorder.save(f'{self.global_frame}_easy.mp4')

        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        self.wandblogger.scalar_summary("eval_video_easy/episode_reward", avg_reward, self.global_frame)


    def eval_video_hard(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_video_hard")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        do_video = (self.cfg.algo == 'dad')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        while eval_until_episode(episode):
            time_step = self.eval_video_hard_env.reset()

            # dad
            record = do_video and (episode == 0)

            if self.env_name == 'dmc':
                if record:
                    self.video_recorder.init_dmc(self.eval_video_hard_env, enabled=True)
            else:
                print("Wrong env")
                break

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if do_video:
                        # dad: action + (recon, z_hat)
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False
                        )
                    else:
                        # not dad: action  (/ X)
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False
                        )

                time_step = self.eval_video_hard_env.step(action)

                # dad
                if self.env_name == 'dmc' and record:
                    original = self.video_recorder.record_dmc(
                        self.eval_video_hard_env, video=True, height=84, width=84
                    )
                    self.video_recorder.record_frame(original, dsize=(96, 96))

                total_reward += time_step.reward
                step += 1

            episode += 1

            # dad
            if record:
                self.video_recorder.save(f'{self.global_frame}_hard.mp4')

        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        self.wandblogger.scalar_summary("eval_video_hard/episode_reward", avg_reward, self.global_frame)

    def eval_color_easy(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_color_easy")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_color_easy_env.reset()
            record = (episode == 0)

            if self.env_name == 'dmc':
                self.video_recorder.init_dmc(self.eval_color_easy_env, enabled=record)
            else:
                print("Wrong env")
                break

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.multi_act:
                        action = self.agent.multi_act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False,
                        )
                    else:
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False,
                        )

                time_step = self.eval_color_easy_env.step(action)

                if self.env_name == 'dmc' and record:
                    original = self.video_recorder.record_dmc(
                        self.eval_color_easy_env,
                        video=False,
                        height=84,
                        width=84,
                    )
                    self.video_recorder.record_frame(original, dsize=(96, 96))

                total_reward += time_step.reward
                step += 1

            episode += 1
            if record:
                self.video_recorder.save(f'{self.global_frame}_color_easy.mp4')

        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        self.wandblogger.scalar_summary("eval_color_easy/episode_reward", avg_reward, self.global_frame)
        self.wandblogger.scalar_summary("eval_color_easy/episode_length", avg_length, self.global_frame)

    def eval_color_hard(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_color_hard")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_color_hard_env.reset()
            record = (episode == 0)

            if self.env_name == 'dmc':
                self.video_recorder.init_dmc(self.eval_color_hard_env, enabled=record)
            else:
                print("Wrong env")
                break

            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    if self.cfg.multi_act:
                        action = self.agent.multi_act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False,
                        )
                    else:
                        action = self.agent.act(
                            time_step.observation,
                            self.global_step,
                            eval_mode=True,
                            decoding=False,
                        )

                time_step = self.eval_color_hard_env.step(action)

                if self.env_name == 'dmc' and record:
                    original = self.video_recorder.record_dmc(
                        self.eval_color_hard_env,
                        video=False,
                        height=84,
                        width=84,
                    )
                    self.video_recorder.record_frame(original, dsize=(96, 96))

                total_reward += time_step.reward
                step += 1

            episode += 1
            if record:
                self.video_recorder.save(f'{self.global_frame}_color_hard.mp4')

        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        self.wandblogger.scalar_summary("eval_color_hard/episode_reward", avg_reward, self.global_frame)
        self.wandblogger.scalar_summary("eval_color_hard/episode_length", avg_length, self.global_frame)



    def robo_eval_easy(self):
        print(f"[INFO] Step {self.global_step}: Running robo_eval_easy")
        step, episode, total_reward = 0, 0, 0
        episode = 0      # evaluated episodes (= scenes)
        # (    )

        do_video = (self.cfg.algo == 'dad')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                # 1) scene_id   env
                scene_id = scene_id % 10  # scene_id 0~9
                eval_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    scene_id=scene_id,
                    mode='eval-easy'
                )

                close_after = getattr(eval_env, "close", None)

                try:
                    time_step = eval_env.reset()

                    # ===    ===
                    # dad   (  scene )
                    record = do_video
                    if record:
                        self.video_recorder.init(eval_env, enabled=True)

                    # 1 episode rollout
                    while not time_step.last():
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            if do_video:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )
                            else:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )

                        time_step = eval_env.step(action)

                        if record:
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))

                        total_reward += float(time_step.reward)
                        step += 1

                    episode += 1
                    if record:
                        self.video_recorder.save(f'{self.global_frame}_scene{scene_id:02d}_eval_easy.mp4')

                finally:
                    if callable(close_after):
                        try:
                            close_after()
                        except Exception:
                            pass
                    del eval_env

            avg_reward = total_reward / max(1, episode)
            self.wandblogger.scalar_summary("eval_easy_env/episode_reward", avg_reward, self.global_frame)

        finally:
            # train_env  (  )
            if self.env_name == 'robosuite':
                if hasattr(self.train_env, "close"):
                    try:
                        self.train_env.close()
                    except Exception:
                        pass

                self.train_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    mode='train'
                )
                self._need_reset_after_eval = True


    def robo_eval_medium(self):
        print(f"[INFO] Step {self.global_step}: Running robo_eval_medium (1 ep per scene_id, 10 scenes)")
        step, total_reward = 0, 0.0
        num_scenes = 10
        episode = 0
        do_video = (self.cfg.algo == 'dad')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                scene_id = scene_id % 10  # scene_id 0~9
                eval_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    scene_id=scene_id,
                    mode='eval-medium'
                )

                close_after = getattr(eval_env, "close", None)

                try:
                    time_step = eval_env.reset()

                    record = do_video
                    if record:
                        self.video_recorder.init(eval_env, enabled=True)

                    while not time_step.last():
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            if do_video:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )
                            else:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )

                        time_step = eval_env.step(action)

                        if record:
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))

                        total_reward += float(time_step.reward)
                        step += 1

                    episode += 1
                    if record:
                        self.video_recorder.save(f'{self.global_frame}_scene{scene_id:02d}_eval_medium.mp4')

                finally:
                    if callable(close_after):
                        try:
                            close_after()
                        except Exception:
                            pass
                    del eval_env

            avg_reward = total_reward / max(1, episode)
            self.wandblogger.scalar_summary("eval_medium_env/episode_reward", avg_reward, self.global_frame)

        finally:
            if self.env_name == 'robosuite':
                if hasattr(self.train_env, "close"):
                    try:
                        self.train_env.close()
                    except Exception:
                        pass

                self.train_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    mode='train'
                )
                self._need_reset_after_eval = True


    def robo_eval_hard(self):
        print(f"[INFO] Step {self.global_step}: Running robo_eval_hard (1 ep per scene_id, 10 scenes)")
        step, total_reward = 0, 0.0
        episode = 0
        do_video = (self.cfg.algo == 'dad')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                scene_id = scene_id % 10  # scene_id 0~9
                eval_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    scene_id=scene_id,
                    mode='eval-hard'
                )

                close_after = getattr(eval_env, "close", None)

                try:
                    time_step = eval_env.reset()

                    record = do_video
                    if record:
                        self.video_recorder.init(eval_env, enabled=True)

                    while not time_step.last():
                        with torch.no_grad(), utils.eval_mode(self.agent):
                            if do_video:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )
                            else:
                                action = self.agent.act(
                                    time_step.observation,
                                    self.global_step,
                                    eval_mode=True,
                                    decoding=False
                                )

                        time_step = eval_env.step(action)

                        if record:
                            import cv2
                            original = time_step.observation  # numpy.ndarray (CHW or similar)
                            original = np.transpose(original, (1, 2, 0))  # CHW -> HWC
                            if original.shape[-1] > 3:
                                original = original[..., :3]
                            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)
                            self.video_recorder.record_frame(original, dsize=(96, 96))

                        total_reward += float(time_step.reward)
                        step += 1

                    episode += 1
                    if record:
                        self.video_recorder.save(f'{self.global_frame}_scene{scene_id:02d}_eval_hard.mp4')

                finally:
                    if callable(close_after):
                        try:
                            close_after()
                        except Exception:
                            pass
                    del eval_env

            avg_reward = total_reward / max(1, episode)

            self.wandblogger.scalar_summary("eval_hard_env/episode_reward", avg_reward, self.global_frame)

        finally:
            if self.env_name == 'robosuite':
                if hasattr(self.train_env, "close"):
                    try:
                        self.train_env.close()
                    except Exception:
                        pass

                self.train_env = robo_make(
                    name=self.cfg.task_name,
                    action_repeat=self.cfg.action_repeat,
                    frame_stack=self.cfg.frame_stack,
                    seed=self.cfg.seed,
                    mode='train'
                )
                self._need_reset_after_eval = True

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames*self.cfg.action_repeat,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames * self.cfg.action_repeat,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames*self.cfg.action_repeat,
                                      self.cfg.action_repeat)
        # 0 (native default) keeps the shifted suites on the source evaluation grid.
        shift_every = getattr(self.cfg, 'eval_shift_every_frames', 0) or 0
        shift_eval_every_step = (utils.Every(shift_every*self.cfg.action_repeat, self.cfg.action_repeat)
                                 if shift_every else eval_every_step)

        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        action = self.agent.act(np.zeros(self.agent.obs_shape, dtype=np.float32),0)
        print(f"act one time {action}")
        total_train_steps = self.cfg.num_train_frames
        progress_initial_step = min(self.global_step, total_train_steps)
        progress = tqdm(
            total=total_train_steps,
            initial=progress_initial_step,
            desc='train_teacher',
            dynamic_ncols=True,
            unit='step'
        )
        progress_start_time = time.time()
        progress.set_postfix(
            episode=self.global_episode,
            reward=f'{float(episode_reward):.2f}',
            eta='--:--:--'
        )

        try:
            while train_until_step(self.global_step):
                if getattr(self, "_need_reset_after_eval", False):
                    time_step = self.train_env.reset()
                    self.replay_storage.add(time_step)
                    self.train_video_recorder.init(time_step.observation)
                    self._need_reset_after_eval = False
                    episode_step = 0  # :

                if time_step.last():
                    self._global_episode += 1
                    self.train_video_recorder.save(f'{self.global_frame}.mp4')
                    # wait until all the metrics schema is populated
                    if metrics is not None:
                        # log stats
                        elapsed_time, total_time = self.timer.reset()
                        episode_frame = episode_step * self.cfg.action_repeat

                        if self.global_step % 100 == 0:
                            self.wandblogger.scalar_summary("train/fps", episode_frame / elapsed_time, self.global_frame)
                            self.wandblogger.scalar_summary('train/total_time', total_time, self.global_frame)
                            self.wandblogger.scalar_summary('train/episode_reward', episode_reward, self.global_frame)
                            self.wandblogger.scalar_summary('train/episode_length', episode_frame, self.global_frame)
                            self.wandblogger.scalar_summary('train/episode', self.global_episode, self.global_frame)
                            self.wandblogger.scalar_summary('train/buffer_size', len(self.replay_storage), self.global_frame)
                            self.wandblogger.scalar_summary('train/step', self.global_step, self.global_frame)

                    # reset env
                    time_step = self.train_env.reset()
                    self.replay_storage.add(time_step)
                    self.train_video_recorder.init(time_step.observation)
                    # try to save snapshot
                    if (self.cfg.save_snapshot and
                            not getattr(self._mechanism, 'manages_snapshots', False) and
                            self.global_step % self.cfg.save_freq == 0):
                        self.save_snapshot(suffix=str(self.global_frame))
                    episode_step = 0
                    episode_reward = 0

                # try to evaluate
                if (not self.cfg.skip_eval) and eval_every_step(self.global_step):
                    shifted = shift_eval_every_step(self.global_step)
                    if self.cfg.env == 'dmc':
                        self.eval()
                        if shifted:
                            self.eval_color_easy()
                            self.eval_color_hard()
                            self.eval_video_easy()
                            self.eval_video_hard()
                    elif self.cfg.env == 'robosuite':
                        self.train_env_eval()
                        self.eval()
                        if shifted:
                            self.robo_eval_easy()
                            self.robo_eval_medium()
                            self.robo_eval_hard()

                    if getattr(self, "_need_reset_after_eval", False):
                        time_step = self.train_env.reset()
                        self.replay_storage.add(time_step)
                        self.train_video_recorder.init(time_step.observation)
                        self._need_reset_after_eval = False
                        episode_step = 0

                # sample action
                # try to update the agent
                if not seed_until_step(self.global_step):
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False)

                    if self.global_step >= self.cfg.init_steps:
                        # Warmup controls data collection, not an initial update
                        # burst: take one learner update here on every step.
                        num_updates = 1
                        for _ in range(num_updates):
                            if self._mechanism is not None:
                                self._mechanism.before_update()
                            metrics = self.agent.update(self.replay_iter, self.global_step)
                            if self._mechanism is not None:
                                self._mechanism.after_update()
                                if hasattr(self._mechanism, 'record_update'):
                                    self._mechanism.record_update(metrics)
                        self.wandblogger.log_metrics(metrics, self.global_frame, ty='train')
                else:
                    if self.cfg.algo == "secant" or self.cfg.algo == "dad2_online":
                        with torch.no_grad():
                            action, _, _, _ = self.agent.teacher_agent.actor(
                                torch.as_tensor(time_step.observation, device='cuda').unsqueeze(0),
                                compute_log_pi=False,
                                encoded=False,
                            )
                        action = action.detach().cpu().numpy()[0]
                    else:
                        action = np.random.uniform(
                            self.train_env.action_spec().minimum,
                            self.train_env.action_spec().maximum,
                            self.train_env.action_spec().shape
                        ).astype(np.float32)
                # take env step
                time_step = self.train_env.step(action)
                episode_reward += time_step.reward
                self.replay_storage.add(time_step)
                self.train_video_recorder.record(time_step.observation)
                episode_step += 1
                self._global_step += 1
                if self._mechanism is not None:
                    self._mechanism.after_transition()
                progress.update(1)

                steps_since_progress_start = self.global_step - progress_initial_step
                if steps_since_progress_start > 0 and (
                    self.global_step % 100 == 0 or self.global_step == total_train_steps
                ):
                    elapsed = max(1e-6, time.time() - progress_start_time)
                    steps_per_second = steps_since_progress_start / elapsed
                    remaining_steps = max(0, total_train_steps - self.global_step)
                    eta_seconds = remaining_steps / max(1e-6, steps_per_second)
                    progress.set_postfix(
                        episode=self.global_episode,
                        reward=f'{float(episode_reward):.2f}',
                        eta=_format_duration(eta_seconds)
                    )
        finally:
            progress.close()
        if not self.cfg.skip_eval:
            if self.cfg.env == 'dmc':
                self.eval()
                self.eval_color_easy()
                self.eval_color_hard()
                self.eval_video_easy()
                self.eval_video_hard()
            elif self.cfg.env == 'robosuite':
                self.train_env_eval()
                self.eval()
                self.robo_eval_easy()
                self.robo_eval_medium()
                self.robo_eval_hard()
        if self.cfg.save_snapshot and not getattr(self._mechanism, 'manages_snapshots', False):
            self.save_snapshot(suffix=str(self.global_frame))
        if self.cfg.buffer_save and getattr(self.cfg, 'buffer_merge_on_finish', True):
            data_dir = Path(os.path.join(self.data_dir, "merged"))
            merge_npz(self.data_dir, data_dir)
        if self._mechanism is not None:
            self._mechanism.finish()


    def train_env_eval(self):
        step, episode, total_reward = 0, 0, 0
        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]
        # robosuite   eval_env . dmc  self.eval_env .
        eval_env = None
        close_after = None

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                #

                scene_id = 0 # scene_id 0..9
                if self.env_name == 'robosuite':
                    eval_env = robo_make(
                        name=self.cfg.task_name,
                        action_repeat=self.cfg.action_repeat,
                        frame_stack=self.cfg.frame_stack,
                        seed=self.cfg.seed,
                        scene_id=scene_id,
                        mode='eval'
                    )
                    close_after = getattr(eval_env, "close", None)
                if self.env_name == 'robosuite':
                    time_step = eval_env.reset()
                else:  # dmc
                    time_step = self.eval_env.reset()

                record = (episode == 0)
                if self.env_name == 'dmc':
                    self.video_recorder.init_dmc(self.eval_env, enabled=record)
                else:
                    self.video_recorder.init(eval_env, enabled=record)
                while not time_step.last():
                    with torch.no_grad(), utils.eval_mode(self.agent):
                        if self.cfg.multi_act:
                            action = self.agent.multi_act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True,
                                            decoding=False,
                                            )
                        else:
                            action = self.agent.act(time_step.observation,
                                                self.global_step,
                                                eval_mode=True,
                                                decoding=False)
                    if self.env_name == 'robosuite':
                        time_step = eval_env.step(action)
                    else:  # dmc
                        time_step = self.eval_env.step(action)
                    if record:
                        if self.env_name == 'dmc':
                            original = self.video_recorder.record_dmc(self.eval_env, video=True)
                            self.video_recorder.record_frame(original, dsize=(96, 96))

                        elif self.env_name == 'robosuite':
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))
                    total_reward += time_step.reward
                    step += 1

                episode += 1
                if record:
                    self.video_recorder.save(f'{self.global_frame}.mp4')
            avg_reward = total_reward / episode
            avg_length = step * self.cfg.action_repeat / episode

            self.wandblogger.scalar_summary("eval_train/episode_reward", avg_reward, self.global_frame)


        finally:
            # robosuite
            if close_after is not None and callable(close_after):
                try:
                    close_after()
                except Exception:
                    pass
            #
            eval_env = None


    def save_snapshot(self, suffix: str = None):
        if suffix is None:
            filename = self.model_dir / 'snapshot.pt'
        else:
            filename = self.model_dir / f'snapshot_{suffix}.pt'

        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with filename.open('xb') as f:
            torch.save(payload, f)


    def load_snapshot(self):
        snapshot = self.model_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = load_checkpoint(f)
        for k, v in payload.items():
            self.__dict__[k] = v


def main(cfg):
    from star.rl.train_teacher import Workspace as W
    workspace = W(cfg)
    try:
        workspace.train()
    finally:
        workspace.wandblogger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
