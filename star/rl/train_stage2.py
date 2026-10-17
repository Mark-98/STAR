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

import hydra
import numpy as np
import torch
from dm_env import specs
import time

from star.rl.arguments import parse_args

import star.rl.wrappers.dmc as dmc
from star.rl import utils
from star.rl.logger import Logger
from star.rl.wandblogger import WandbLogger
from star.rl.replay_buffer import ReplayBufferStorage, make_replay_loader
from star.rl.video import TrainVideoRecorder, VideoRecorder
from omegaconf import OmegaConf
from tqdm import tqdm
from star.rl.wrappers.robo_wrapper import robo_make

torch.backends.cudnn.benchmark = True
from star.rl.read_buffer.split_buffer import merge_npz
from star.rl.algos.factory import make_agent


def debug_log(message):
    if os.environ.get("DAD_DEBUG_STARTUP", "0").lower() in {"1", "true", "yes", "on"}:
        print(f"[DAD DEBUG] {message}", flush=True)


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()

        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device("cuda")
        run_dir = self.work_dir / 'logs' / cfg.exp_name
        run_dir.mkdir(parents=True, exist_ok=False)
        self.cfg.local_metrics_path = str(run_dir / 'metrics.jsonl')
        self.model_dir = run_dir / 'model'
        self.model_dir.mkdir()
        buffer_dir = Path(self.cfg.buffer_dir)
        from star.datasets.replay import stage_replay, validate_full_replay
        if getattr(cfg, 'require_full_replay', False):
            import json
            report = validate_full_replay(buffer_dir, cfg)
            (run_dir / 'replay_manifest.json').write_text(json.dumps(report, indent=2) + '\n')
        # The legacy reader can unlink NPZs. Preserve sampling/eviction semantics
        # on private copies; never give it the user's immutable source directory.
        self.data_dir = stage_replay(buffer_dir, run_dir / 'replay_work' / 'merged')
        self.video_dir = Path(utils.make_dir(os.path.join(self.work_dir, 'logs', cfg.exp_name, 'video')))

        debug_log(f"setup start env={self.cfg.env} task={self.cfg.task_name} data_dir={self.data_dir}")
        self.setup(self.cfg.env)

        debug_log("make_agent start")
        self.agent = make_agent(self.eval_env.observation_spec().shape,
                                self.eval_env.action_spec().shape[0],
                                self.cfg)
        debug_log("make_agent done")
        import json
        (run_dir / 'resolved_config.json').write_text(json.dumps(vars(self.cfg), indent=2, default=str) + '\n')
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0


    def setup(self, env):
        debug_log("logger init start")
        self.logger = Logger(self.model_dir)
        # merge_npz(self.cfg.buffer_dir, self.data_dir)

        debug_log("wandb init start")
        self.wandblogger = WandbLogger(self.cfg)
        debug_log("wandb init done")
        assert env in ['dmc', 'robosuite', 'habitat']
        self.env_name = env
        # create envs
        if env == 'dmc':
            import star.rl.wrappers.loco_wrapper as dmc
            debug_log("dmc env init start")
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
            debug_log("dmc env init done")
        elif env == 'robosuite':
            debug_log("robosuite eval_env init start")
            self.eval_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                    frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval')
            debug_log("robosuite eval_easy init start")
            self.eval_easy_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                    frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-easy')
            debug_log("robosuite eval_medium init start")
            self.eval_medium_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                    frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-medium')
            debug_log("robosuite eval_hard init start")
            self.eval_hard_env = robo_make(name=self.cfg.task_name, action_repeat=self.cfg.action_repeat,
                                    frame_stack=self.cfg.frame_stack, seed=self.cfg.seed, mode='eval-hard')
            debug_log("robosuite env init done")

        else:
            raise ValueError(f"env {env} not supported.")

        # create replay buffer
        data_specs = (self.eval_env.observation_spec(),
                      self.eval_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        debug_log("ReplayBufferStorage init start")
        self.replay_storage = ReplayBufferStorage(data_specs, self.data_dir)
        debug_log("ReplayBufferStorage init done")

        debug_log(
            f"make_replay_loader start data_dir={self.data_dir} "
            f"batch_size={self.cfg.batch_size} workers={self.cfg.replay_buffer_num_workers} "
            f"replay_dataset_index={self.cfg.replay_dataset_index} "
            f"replay_dataset_start={self.cfg.replay_dataset_start} "
            f"replay_dataset_end={self.cfg.replay_dataset_end} "
            f"replay_dataset_tail_count={self.cfg.replay_dataset_tail_count} "
            f"replay_dataset_start_ratio={self.cfg.replay_dataset_start_ratio} "
            f"replay_dataset_end_ratio={self.cfg.replay_dataset_end_ratio} "
            f"replay_dataset_tail_ratio={self.cfg.replay_dataset_tail_ratio}"
        )
        self.train_replay_loader = make_replay_loader(
            self.data_dir,
            self.cfg.replay_buffer_size,
            self.cfg.batch_size,
            self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot,
            self.cfg.nstep,
            self.cfg.discount,
            self.cfg.buffer_save,
            True if self.cfg.algo in ["dad2"] else False,
            replay_dataset_index=self.cfg.replay_dataset_index,
            replay_dataset_start=self.cfg.replay_dataset_start,
            replay_dataset_end=self.cfg.replay_dataset_end,
            replay_dataset_tail_count=self.cfg.replay_dataset_tail_count,
            replay_dataset_start_ratio=self.cfg.replay_dataset_start_ratio,
            replay_dataset_end_ratio=self.cfg.replay_dataset_end_ratio,
            replay_dataset_tail_ratio=self.cfg.replay_dataset_tail_ratio,
        )
        debug_log("make_replay_loader done")

        self._replay_iter = None

        self.video_recorder = VideoRecorder(
            self.video_dir if self.cfg.save_video else None)
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
        return self.global_step # offline -> no action_repeat


    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.train_replay_loader)
        return self._replay_iter

    @property
    def val_replay_iter(self):  # <-- NEW
        if self._val_replay_iter is None:
            self._val_replay_iter = iter(self.val_replay_loader)
        return self._val_replay_iter

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

            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', avg_reward)
                log('episode_length', avg_length)
                log('episode', self.global_episode)
                log('step', self.global_step)
                self.wandblogger.scalar_summary("eval_train/episode_reward", avg_reward, self.global_frame)
                self.wandblogger.scalar_summary("eval_train/episode_length", avg_length, self.global_frame)
                self.wandblogger.scalar_summary("eval_train/episode", self.global_episode, self.global_frame)

        finally:
            # robosuite
            if close_after is not None and callable(close_after):
                try:
                    close_after()
                except Exception:
                    pass
            #
            eval_env = None

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

            with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
                log('episode_reward', avg_reward)
                log('episode_length', avg_length)
                log('episode', self.global_episode)
                log('step', self.global_step)
                self.wandblogger.scalar_summary("eval/episode_reward", avg_reward, self.global_frame)
                self.wandblogger.scalar_summary("eval/episode_length", avg_length, self.global_frame)
                self.wandblogger.scalar_summary("eval/episode", self.global_episode, self.global_frame)

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
        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        for scene_id in range(self.cfg.num_eval_episodes):
            time_step = self.eval_video_easy_env.reset()
            record = (episode == 0)
            if self.env_name == 'dmc':
                self.video_recorder.init_dmc(self.eval_video_easy_env, enabled=(episode == 0))
            else:
                print("Wrong env")
                break
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
                time_step = self.eval_video_easy_env.step(action)
                if self.env_name == 'dmc':
                    if record:
                        original = self.video_recorder.record_dmc(self.eval_video_easy_env, video=True, height=84, width=84)
                        self.video_recorder.record_frame(original, dsize=(96, 96))
                else:
                    print("Wrong env")
                    break
                total_reward += time_step.reward
                step += 1

            episode += 1
            if record:
                self.video_recorder.save(f'{self.global_frame}_easy.mp4')
        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_reward)
            log('episode_length', avg_length)
            log('episode', self.global_episode)
            log('step', self.global_step)

            self.wandblogger.scalar_summary("eval_video_easy/episode_reward", avg_reward, self.global_frame)

    def eval_color_easy(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_color_easy")
        step, episode, total_reward = 0, 0, 0

        for _ in range(self.cfg.num_eval_episodes):
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

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_reward)
            log('episode_length', avg_length)
            log('episode', self.global_episode)
            log('step', self.global_step)

            self.wandblogger.scalar_summary("eval_color_easy/episode_reward", avg_reward, self.global_frame)
            self.wandblogger.scalar_summary("eval_color_easy/episode_length", avg_length, self.global_frame)


    def eval_video_hard(self):
        print(f"[INFO] Step {self.global_step}: Running dmc_eval_video_hard")
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        while eval_until_episode(episode):
            time_step = self.eval_video_hard_env.reset()
            record = (episode == 0)
            if self.env_name == 'dmc':
                self.video_recorder.init_dmc(self.eval_video_hard_env, enabled=(episode == 0))
            else:
                print("Wrong env")
                break
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
                time_step = self.eval_video_hard_env.step(action)
                if self.env_name == 'dmc':
                    if record:
                        original = self.video_recorder.record_dmc(self.eval_video_hard_env, video=True, height=84, width=84)
                        self.video_recorder.record_frame(original, dsize=(96, 96))
                else:
                    print("Wrong env")
                    break
                total_reward += time_step.reward
                step += 1

            episode += 1
            if record:
                self.video_recorder.save(f'{self.global_frame}_hard.mp4')
        avg_reward = total_reward / episode
        avg_length = step * self.cfg.action_repeat / episode

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_reward)
            log('episode_length', avg_length)
            log('episode', self.global_episode)
            log('step', self.global_step)

            self.wandblogger.scalar_summary("eval_video_hard/episode_reward", avg_reward, self.global_frame)

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

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', avg_reward)
            log('episode_length', avg_length)
            log('episode', self.global_episode)
            log('step', self.global_step)

            self.wandblogger.scalar_summary("eval_color_hard/episode_reward", avg_reward, self.global_frame)
            self.wandblogger.scalar_summary("eval_color_hard/episode_length", avg_length, self.global_frame)


    def robo_eval_easy(self):
        print(f"[INFO] Step {self.global_step}: Running robo_eval_easy")
        step, episode, total_reward = 0, 0, 0
        num_scenes = 10  # scene_id 0..9
        episode = 0      # evaluated episodes (= scenes)

        do_video = (self.cfg.algo == 'ours')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                # 1) scene_id   env
                scene_id = episode % 10  # scene_id 0..9
                if self.cfg.robo_camera_name == 'robot0_eye_in_hand':
                    eval_env = robo_make(
                        name=self.cfg.task_name,
                        action_repeat=self.cfg.action_repeat,
                        frame_stack=self.cfg.frame_stack,
                        seed=self.cfg.seed,
                        scene_id=scene_id,
                        mode='eval_easy_hand'
                    )
                else:
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
                    # ours   (  scene )
                    record = do_video
                    if record:
                        self.video_recorder.init(eval_env, enabled=True)

                    # 1 episode rollout
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

                        time_step = eval_env.step(action)

                        if record:
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))
                        total_reward += time_step.reward
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

        do_video = (self.cfg.algo == 'ours')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(self.cfg.num_eval_episodes):
                scene_id = episode % 10  # scene_id 0..9
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

                        time_step = eval_env.step(action)

                        if record:
                            original = self.video_recorder.record(eval_env)
                            self.video_recorder.record_frame(original, dsize=(96, 96))
                        total_reward += time_step.reward
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
        num_scenes = 30
        episode = 0

        do_video = (self.cfg.algo == 'ours')

        def process_frame(obs):
            obs = np.clip(obs, 0.0, 1.0)
            obs = (obs * 255.0).astype(np.uint8)
            return obs[..., -3:]

        try:
            for scene_id in range(num_scenes):
                scene_id = episode % 10  # scene_id 0..9
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

                        time_step = eval_env.step(action)

                        if record:
                            import cv2
                            original = time_step.observation  # numpy.ndarray
                            original = np.transpose(original, (1, 2, 0))  # CHW -> HWC
                            if original.shape[-1] > 3:
                                original = original[..., :3]
                            original = cv2.cvtColor(original, cv2.COLOR_RGB2BGR)

                            self.video_recorder.record_frame(original, dsize=(256, 96))

                        total_reward += time_step.reward
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
            self.train_env = robo_make(
                name=self.cfg.task_name,
                action_repeat=self.cfg.action_repeat,
                frame_stack=self.cfg.frame_stack,
                seed=self.cfg.seed,
                mode='train'
            )
            self._need_reset_after_eval = True


    def train(self):
        # Teacher    0 step eval
        # self.eval()
        # if self.cfg.env == 'dmc':
        #     self.eval_video_easy()
        #     self.eval_video_hard()
        # elif self.cfg.env == 'robosuite':
        #     self.robo_eval_easy()
        #     self.robo_eval_medium()
        #     self.robo_eval_hard()

        start_time = time.time()
        last_time = start_time
        last_step = 0

        start_step = self.global_step + 1
        # eval_env = robo_make(
        #                 name=self.cfg.task_name,
        #                 action_repeat=self.cfg.action_repeat,
        #                 frame_stack=self.cfg.frame_stack,
        #                 seed=self.cfg.seed,
        #                 mode='eval'
        #             )
        # timestep = eval_env.reset()
        # print(self.agent.act(timestep.observation, self.global_step, eval_mode=True, decoding=False))  # warmup for act
        # eval_env.close()
        debug_log(f"train loop start start_step={start_step} num_train_frames={self.cfg.num_train_frames}")
        for step in tqdm(range(start_step, self.cfg.num_train_frames + 1),
                         initial=max(0, start_step - 1),
                         total=self.cfg.num_train_frames):
            self._global_step = step
            if step == start_step:
                debug_log("first agent.update start")
            metrics = self.agent.update(self.replay_iter, self.global_step)
            if step == start_step:
                debug_log(f"first agent.update done metrics_keys={sorted(metrics.keys())}")
            if step == start_step or step % 1000 == 0 or step == self.cfg.num_train_frames:
                import json
                progress = dict(step=step, total=self.cfg.num_train_frames,
                                elapsed_seconds=time.time() - start_time,
                                metrics={k: float(v) for k, v in metrics.items()})
                progress_path = self.model_dir.parent / 'progress.json'
                temporary = progress_path.with_suffix('.tmp')
                temporary.write_text(json.dumps(progress, indent=2) + '\n')
                temporary.replace(progress_path)

            if step % 1000 == 0:
                now = time.time()
                dt = max(1e-6, now - last_time)
                dstep = step - last_step

                last_time = now
                last_step = step

                self.wandblogger.log_metrics(metrics, self.global_frame, ty='train')

            if step!=0 and step % self.cfg.eval_every_frames == 0:
                self.save_snapshot(suffix=str(step))

                self.eval()
                if self.cfg.env == 'dmc':
                    self.eval_color_easy()
                    self.eval_color_hard()
                    self.eval_video_easy()
                    self.eval_video_hard()
                elif self.cfg.env == 'robosuite':
                    self.train_env_eval()
                    self.robo_eval_easy()
                    self.robo_eval_medium()
                    self.robo_eval_hard()
            if step!=0 and step % self.cfg.save_freq == 0 and step % self.cfg.eval_every_frames != 0:
                self.save_snapshot(suffix=str(step))
            if step == self.cfg.num_train_frames:
                print(f"[INFO] Final step {step}: saving final snapshot")
                self.save_snapshot()

    def save_snapshot(self, suffix: str = None):
        if suffix is None:
            filename = self.model_dir  / 'snapshot.pt'
        else:
            filename = self.model_dir  / f'snapshot_{suffix}.pt'

        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with filename.open('xb') as f:
            torch.save(payload, f)


    def load_snapshot(self, snapshot_path: str = None):
        if snapshot_path is None:
            snapshot = self.model_dir / 'snapshot.pt'
        else:
            snapshot = Path(snapshot_path)
            if snapshot.is_dir():
                snapshot = snapshot / 'snapshot.pt'

        if not snapshot.exists():
            raise FileNotFoundError(f"Snapshot not found: {snapshot}")

        with snapshot.open('rb') as f:
            payload = load_checkpoint(f, map_location=self.device)
        for k, v in payload.items():
            self.__dict__[k] = v

        print(f"[INFO] Resumed training from {snapshot} at step {self.global_step}")

def main(cfg):
    if getattr(cfg, 'causal_config', None):
        raise NotImplementedError('Opt-in causal experiment suite is outside this release; see docs/audit.md.')
    from star.rl.train_stage2 import Workspace as W
    workspace = W(cfg)
    resume_path = None
    if getattr(cfg, 'model_load_dir', 'none') != 'none':
        resume_path = cfg.model_load_dir

    should_resume = getattr(cfg, 'resume', False) or resume_path is not None
    if should_resume:
        if resume_path is None:
            resume_path = str(workspace.model_dir / 'snapshot.pt')
        print(f"[INFO] Attempting to resume from {resume_path}")
        workspace.load_snapshot(resume_path)
    try:
        workspace.train()
    finally:
        workspace.wandblogger.close()


if __name__ == '__main__':
    args = parse_args()
    main(args)
