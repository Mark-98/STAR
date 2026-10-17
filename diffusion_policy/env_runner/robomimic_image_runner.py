import os
import wandb
import numpy as np
import torch
import collections
import pathlib
import tqdm
import h5py
import math
import dill
import wandb.sdk.data_types.video as wv
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.video_recording_wrapper import VideoRecordingWrapper, VideoRecorder
from diffusion_policy.model.common.rotation_transformer import RotationTransformer

from diffusion_policy.policy.base_image_policy import BaseImagePolicy
from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env_runner.base_image_runner import BaseImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
    RobomimicVisualModWrapper,
)
from diffusion_policy.env.robomimic.rebalanced_difficulty_layout import (
    DEFAULT_LAYOUT_NAME,
    SUPPORTED_LAYOUT_NAMES,
    build_rebalanced_scene_records,
)
import robomimic.utils.file_utils as FileUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils


def create_env(env_meta, shape_meta, enable_render=True):
    modality_mapping = collections.defaultdict(list)
    for key, attr in shape_meta['obs'].items():
        modality_mapping[attr.get('type', 'low_dim')].append(key)
    ObsUtils.initialize_obs_modality_mapping_from_dict(modality_mapping)

    env = EnvUtils.create_env_from_metadata(
        env_meta=env_meta,
        render=False,
        render_offscreen=enable_render,
        use_image_obs=enable_render,
    )
    return env


def create_env_with_texture_mod(
    env_meta,
    shape_meta,
    enable_render=True,
    render_obs_key='agentview_image',
    task_name=None,
    texture_map=None,
    material_map=None,
    skybox_config=None,
    light_config=None,
    texture_seed=None,
    visual_mode=None,
    scene_id=None,
    randomize_on_reset=False
):
    env = create_env(
        env_meta=env_meta,
        shape_meta=shape_meta,
        enable_render=enable_render,
    )
    return RobomimicVisualModWrapper(
        env=env,
        shape_meta=shape_meta,
        init_state=None,
        render_obs_key=render_obs_key,
        task_name=task_name,
        texture_map=texture_map,
        material_map=material_map,
        skybox_config=skybox_config,
        light_config=light_config,
        texture_seed=texture_seed,
        visual_mode=visual_mode,
        scene_id=scene_id,
        randomize_on_reset=randomize_on_reset,
    )


class RobomimicImageRunner(BaseImageRunner):
    """
    Robomimic envs already enforces number of steps.
    """

    def __init__(self,
            output_dir,
            dataset_path,
            shape_meta:dict,
            n_train=10,
            n_train_vis=3,
            train_start_idx=0,
            n_test=22,
            n_test_vis=6,
            test_start_seed=10000,
            n_eval_easy=0,
            n_eval_easy_vis=0,
            n_eval_medium=0,
            n_eval_medium_vis=0,
            n_eval_hard=0,
            n_eval_hard_vis=0,
            eval_difficulty_start_seed=200000,
            eval_difficulty_scene_start_id=0,
            difficulty_seed_mode='offset',
            difficulty_layout=DEFAULT_LAYOUT_NAME,
            difficulty_eval_configs='',
            max_steps=400,
            n_obs_steps=2,
            n_action_steps=8,
            render_obs_key='agentview_image',
            fps=10,
            crf=22,
            past_action=False,
            abs_action=False,
            tqdm_interval_sec=5.0,
            n_envs=None,
            dense_reward_eval=False
        ):
        super().__init__(output_dir)

        difficulty_specs = (
            ('eval-easy', 'eval_easy/', int(n_eval_easy), int(n_eval_easy_vis)),
            ('eval-medium', 'eval_medium/', int(n_eval_medium), int(n_eval_medium_vis)),
            ('eval-hard', 'eval_hard/', int(n_eval_hard), int(n_eval_hard_vis)),
        )
        use_visual_mod_env = any(count > 0 for _, _, count, _ in difficulty_specs)

        def validate_seed_mode(seed_mode):
            seed_mode = str(seed_mode)
            if seed_mode not in ('offset', 'matched', 'record'):
                raise ValueError(
                    "difficulty_seed_mode must be one of: 'offset', 'matched', "
                    "'record'. "
                    f'Got: {seed_mode}'
                )
            return seed_mode

        def validate_layout(layout_name):
            layout_name = str(layout_name or DEFAULT_LAYOUT_NAME)
            if layout_name not in SUPPORTED_LAYOUT_NAMES:
                raise ValueError(
                    f'difficulty_layout must be one of {sorted(SUPPORTED_LAYOUT_NAMES)}. '
                    f'Got: {layout_name}'
                )
            return layout_name

        difficulty_seed_mode = str(difficulty_seed_mode)
        difficulty_seed_mode = validate_seed_mode(difficulty_seed_mode)
        difficulty_layout = validate_layout(difficulty_layout)

        def parse_difficulty_eval_configs(config_value):
            if config_value is None or str(config_value).strip() == '':
                return [('', difficulty_seed_mode, difficulty_layout)]
            items = [
                item.strip()
                for item in str(config_value).replace(';', ',').replace(' ', ',').split(',')
                if item.strip()
            ]
            configs = []
            for item in items:
                parts = [part.strip() for part in item.split(':')]
                if len(parts) != 3 or any(part == '' for part in parts):
                    raise ValueError(
                        'difficulty_eval_configs entries must use '
                        'label:seed_mode:layout format, e.g. '
                        'fixed:matched:rebalanced_v1,legacy:offset:default. '
                        f'Got: {item}'
                    )
                label, seed_mode, layout_name = parts
                if not label.replace('_', '').replace('-', '').isalnum():
                    raise ValueError(
                        'difficulty_eval_configs labels may contain only letters, '
                        f'numbers, hyphens, and underscores. Got: {label}'
                    )
                configs.append((
                    label.replace('-', '_'),
                    validate_seed_mode(seed_mode),
                    validate_layout(layout_name),
                ))
            return configs

        difficulty_eval_config_list = parse_difficulty_eval_configs(
            difficulty_eval_configs
        )

        if n_envs is None:
            n_envs = (
                n_train
                + n_test
                + len(difficulty_eval_config_list)
                * sum(count for _, _, count, _ in difficulty_specs)
            )

        # assert n_obs_steps <= n_action_steps
        dataset_path = os.path.expanduser(dataset_path)
        robosuite_fps = 20
        steps_per_render = max(robosuite_fps // fps, 1)

        # read from dataset
        env_meta = FileUtils.get_env_metadata_from_dataset(
            dataset_path)
        # disable object state observation
        env_meta['env_kwargs']['use_object_obs'] = False

        difficulty_records_by_mode = dict()
        for _, _, layout_name in difficulty_eval_config_list:
            if layout_name == DEFAULT_LAYOUT_NAME:
                continue
            for visual_mode, _, count, _ in difficulty_specs:
                if count <= 0:
                    continue
                cache_key = (layout_name, visual_mode)
                if cache_key in difficulty_records_by_mode:
                    continue
                records = build_rebalanced_scene_records(
                    task_name=env_meta['env_name'],
                    visual_mode=visual_mode,
                    layout_name=layout_name,
                )
                if records is None:
                    raise ValueError(
                        f'difficulty_layout={layout_name} is not supported '
                        f'for task={env_meta["env_name"]}.'
                    )
                difficulty_records_by_mode[cache_key] = records

        rotation_transformer = None
        if abs_action:
            env_meta['env_kwargs']['controller_configs']['control_delta'] = False
            rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

        def wrap_env(robomimic_env):
            if use_visual_mod_env:
                return RobomimicVisualModWrapper(
                    env=robomimic_env,
                    shape_meta=shape_meta,
                    init_state=None,
                    render_obs_key=render_obs_key,
                    task_name=env_meta['env_name'],
                    randomize_on_reset=False,
                )
            return RobomimicImageWrapper(
                env=robomimic_env,
                shape_meta=shape_meta,
                init_state=None,
                render_obs_key=render_obs_key
            )

        def reset_original_visual_mod(policy_env):
            if isinstance(policy_env, RobomimicVisualModWrapper):
                policy_env.restore_visual_mod()
                policy_env.visual_mode = None
                policy_env.scene_id = None
                policy_env.texture_seed = None
                policy_env.texture_map = None
                policy_env.material_map = None
                policy_env.skybox_config = None
                policy_env.light_config = None
                policy_env.visual_config_override = None
                policy_env.randomize_on_reset = False

        def env_fn():
            robomimic_env = create_env(
                env_meta=env_meta,
                shape_meta=shape_meta
            )
            # Robosuite's hard reset causes excessive memory consumption.
            # Disabled to run more envs.
            # https://github.com/ARISE-Initiative/robosuite/blob/92abf5595eddb3a845cd1093703e5a3ccd01e77e/robosuite/environments/base.py#L247-L248
            robomimic_env.env.hard_reset = False
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    wrap_env(robomimic_env),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        # For each process the OpenGL context can only be initialized once
        # Since AsyncVectorEnv uses fork to create worker process,
        # a separate env_fn that does not create OpenGL context (enable_render=False)
        # is needed to initialize spaces.
        def dummy_env_fn():
            robomimic_env = create_env(
                    env_meta=env_meta,
                    shape_meta=shape_meta,
                    enable_render=False
                )
            return MultiStepWrapper(
                VideoRecordingWrapper(
                    wrap_env(robomimic_env),
                    video_recoder=VideoRecorder.create_h264(
                        fps=fps,
                        codec='h264',
                        input_pix_fmt='rgb24',
                        crf=crf,
                        thread_type='FRAME',
                        thread_count=1
                    ),
                    file_path=None,
                    steps_per_render=steps_per_render
                ),
                n_obs_steps=n_obs_steps,
                n_action_steps=n_action_steps,
                max_episode_steps=max_steps
            )

        env_fns = [env_fn] * n_envs
        env_seeds = list()
        env_prefixs = list()
        env_init_fn_dills = list()

        # train
        with h5py.File(dataset_path, 'r') as f:
            for i in range(n_train):
                train_idx = train_start_idx + i
                enable_render = i < n_train_vis
                init_state = f[f'data/demo_{train_idx}/states'][0]

                def init_fn(env, init_state=init_state,
                    enable_render=enable_render):
                    # setup rendering
                    # video_wrapper
                    assert isinstance(env.env, VideoRecordingWrapper)
                    env.env.video_recoder.stop()
                    env.env.file_path = None
                    if enable_render:
                        filename = pathlib.Path(output_dir).joinpath(
                            'media', wv.util.generate_id() + ".mp4")
                        filename.parent.mkdir(parents=False, exist_ok=True)
                        filename = str(filename)
                        env.env.file_path = filename

                    # switch to init_state reset
                    assert isinstance(env.env.env, RobomimicImageWrapper)
                    reset_original_visual_mod(env.env.env)
                    env.env.env.init_state = init_state
                    env.env.env._seed = None
                    # train env: keep sparse reward
                    env.env.env.set_reward_shaping(False)

                env_seeds.append(train_idx)
                env_prefixs.append('train/')
                env_init_fn_dills.append(dill.dumps(init_fn))

        # test
        for i in range(n_test):
            seed = test_start_seed + i
            enable_render = i < n_test_vis

            def init_fn(env, seed=seed,
                enable_render=enable_render,
                dense_reward_eval=dense_reward_eval):
                # setup rendering
                # video_wrapper
                assert isinstance(env.env, VideoRecordingWrapper)
                env.env.video_recoder.stop()
                env.env.file_path = None
                if enable_render:
                    filename = pathlib.Path(output_dir).joinpath(
                        'media', wv.util.generate_id() + ".mp4")
                    filename.parent.mkdir(parents=False, exist_ok=True)
                    filename = str(filename)
                    env.env.file_path = filename

                # switch to seed reset
                assert isinstance(env.env.env, RobomimicImageWrapper)
                reset_original_visual_mod(env.env.env)
                env.env.env.init_state = None
                env.seed(seed)
                # test env only: dense (shaped) reward when requested
                env.env.env.set_reward_shaping(dense_reward_eval)

            env_seeds.append(seed)
            env_prefixs.append('test/')
            env_init_fn_dills.append(dill.dumps(init_fn))

        # difficulty texture evaluation
        for difficulty_label, seed_mode, layout_name in difficulty_eval_config_list:
            difficulty_seed_offset = 0
            for visual_mode, prefix, count, n_vis in difficulty_specs:
                if difficulty_label:
                    difficulty_name = visual_mode.replace('eval-', '').replace('-', '_')
                    prefix = f'eval_{difficulty_label}_{difficulty_name}/'
                mode_records = difficulty_records_by_mode.get((layout_name, visual_mode))
                for i in range(count):
                    if seed_mode == 'matched':
                        seed = eval_difficulty_start_seed + i
                    elif seed_mode == 'record':
                        seed = eval_difficulty_start_seed + i
                    else:
                        seed = eval_difficulty_start_seed + difficulty_seed_offset + i
                    scene_id = eval_difficulty_scene_start_id + i
                    enable_render = i < n_vis
                    record_visual_mode = visual_mode
                    visual_config_override = None
                    if mode_records is not None:
                        record_idx = eval_difficulty_scene_start_id + i
                        if record_idx < 0 or record_idx >= len(mode_records):
                            raise ValueError(
                                f'difficulty scene_id={record_idx} is out of range for '
                                f'{env_meta["env_name"]} {visual_mode} {layout_name}. '
                                f'Available range: 0..{len(mode_records) - 1}.'
                            )
                        record = mode_records[record_idx]
                        record_visual_mode = record.get('source_visual_mode', visual_mode)
                        scene_id = int(record.get('source_scene_id', scene_id))
                        if seed_mode == 'record':
                            seed = int(record.get('eval_seed', seed))
                        visual_config_override = {
                            'texture_map': dict(record['texture_map']),
                            'material_map': None if record.get('material_map') is None else dict(record['material_map']),
                            'skybox_config': None if record.get('skybox_config') is None else dict(record['skybox_config']),
                            'light_config': None if record.get('light_config') is None else dict(record['light_config']),
                        }

                    def init_fn(
                        env,
                        seed=seed,
                        enable_render=enable_render,
                        visual_mode=record_visual_mode,
                        scene_id=scene_id,
                        visual_config_override=visual_config_override,
                        dense_reward_eval=dense_reward_eval,
                    ):
                        # setup rendering
                        # video_wrapper
                        assert isinstance(env.env, VideoRecordingWrapper)
                        env.env.video_recoder.stop()
                        env.env.file_path = None
                        if enable_render:
                            filename = pathlib.Path(output_dir).joinpath(
                                'media', wv.util.generate_id() + ".mp4")
                            filename.parent.mkdir(parents=False, exist_ok=True)
                            filename = str(filename)
                            env.env.file_path = filename

                        # switch to seeded reset with deterministic difficulty texture
                        assert isinstance(env.env.env, RobomimicVisualModWrapper)
                        policy_env = env.env.env
                        policy_env.init_state = None
                        policy_env.seed(seed)
                        policy_env.texture_map = None
                        policy_env.material_map = None
                        policy_env.skybox_config = None
                        policy_env.light_config = None
                        policy_env.texture_seed = None
                        policy_env.visual_mode = visual_mode
                        policy_env.scene_id = scene_id
                        policy_env.visual_config_override = visual_config_override
                        policy_env.randomize_on_reset = False
                        policy_env._has_applied_visual_mod = False
                        # eval env only: dense (shaped) reward when requested
                        policy_env.set_reward_shaping(dense_reward_eval)

                    env_seeds.append(seed)
                    env_prefixs.append(prefix)
                    env_init_fn_dills.append(dill.dumps(init_fn))
                difficulty_seed_offset += count

        env = AsyncVectorEnv(env_fns, dummy_env_fn=dummy_env_fn)
        # env = SyncVectorEnv(env_fns)


        self.env_meta = env_meta
        self.env = env
        self.env_fns = env_fns
        self.env_seeds = env_seeds
        self.env_prefixs = env_prefixs
        self.env_init_fn_dills = env_init_fn_dills
        self.fps = fps
        self.crf = crf
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.past_action = past_action
        self.max_steps = max_steps
        self.rotation_transformer = rotation_transformer
        self.abs_action = abs_action
        self.tqdm_interval_sec = tqdm_interval_sec
        self.dense_reward_eval = dense_reward_eval

    def run(self, policy: BaseImagePolicy):
        device = policy.device
        dtype = policy.dtype
        env = self.env

        # plan for rollout
        n_envs = len(self.env_fns)
        n_inits = len(self.env_init_fn_dills)
        n_chunks = math.ceil(n_inits / n_envs)

        # allocate data
        all_video_paths = [None] * n_inits
        all_rewards = [None] * n_inits
        all_successes = [None] * n_inits

        for chunk_idx in range(n_chunks):
            start = chunk_idx * n_envs
            end = min(n_inits, start + n_envs)
            this_global_slice = slice(start, end)
            this_n_active_envs = end - start
            this_local_slice = slice(0,this_n_active_envs)

            this_init_fns = self.env_init_fn_dills[this_global_slice]
            n_diff = n_envs - len(this_init_fns)
            if n_diff > 0:
                this_init_fns.extend([self.env_init_fn_dills[0]]*n_diff)
            assert len(this_init_fns) == n_envs

            # init envs
            env.call_each('run_dill_function',
                args_list=[(x,) for x in this_init_fns])

            # start rollout
            obs = env.reset()
            past_action = None
            policy.reset()

            env_name = self.env_meta['env_name']
            pbar = tqdm.tqdm(total=self.max_steps, desc=f"Eval {env_name}Image {chunk_idx+1}/{n_chunks}",
                leave=False, mininterval=self.tqdm_interval_sec)

            done = False
            while not done:
                # create obs dict
                np_obs_dict = dict(obs)
                if self.past_action and (past_action is not None):
                    # TODO: not tested
                    np_obs_dict['past_action'] = past_action[
                        :,-(self.n_obs_steps-1):].astype(np.float32)

                # device transfer
                obs_dict = dict_apply(np_obs_dict,
                    lambda x: torch.from_numpy(x).to(
                        device=device))

                # run policy
                with torch.no_grad():
                    action_dict = policy.predict_action(obs_dict)

                # device_transfer
                np_action_dict = dict_apply(action_dict,
                    lambda x: x.detach().to('cpu').numpy())

                action = np_action_dict['action']
                if not np.all(np.isfinite(action)):
                    print(action)
                    raise RuntimeError("Nan or Inf action")

                # step env
                env_action = action
                if self.abs_action:
                    env_action = self.undo_transform_action(action)

                obs, reward, done, info = env.step(env_action)
                done = np.all(done)
                past_action = action

                # update pbar
                pbar.update(action.shape[1])
            pbar.close()

            # collect data for this round
            all_video_paths[this_global_slice] = env.render()[this_local_slice]
            all_rewards[this_global_slice] = env.call('get_attr', 'reward')[this_local_slice]
            all_successes[this_global_slice] = env.call('get_attr', 'episode_success')[this_local_slice]
        # clear out video buffer
        _ = env.reset()

        # log
        max_rewards = collections.defaultdict(list)
        successes = collections.defaultdict(list)
        log_data = dict()
        # results reported in the paper are generated using the commented out line below
        # which will only report and average metrics from first n_envs initial condition and seeds
        # fortunately this won't invalidate our conclusion since
        # 1. This bug only affects the variance of metrics, not their mean
        # 2. All baseline methods are evaluated using the same code
        # to completely reproduce reported numbers, uncomment this line:
        # for i in range(len(self.env_fns)):
        # and comment out this line
        for i in range(n_inits):
            seed = self.env_seeds[i]
            prefix = self.env_prefixs[i]
            max_reward = np.max(all_rewards[i])
            max_rewards[prefix].append(max_reward)
            log_data[prefix+f'sim_max_reward_{seed}'] = max_reward
            success = float(bool(all_successes[i]))
            successes[prefix].append(success)
            log_data[prefix+f'sim_success_{seed}'] = success

            # visualize sim
            video_path = all_video_paths[i]
            if video_path is not None:
                sim_video = wandb.Video(video_path)
                log_data[prefix+f'sim_video_{seed}'] = sim_video

        # log aggregate metrics
        for prefix, value in max_rewards.items():
            name = prefix+'mean_score'
            value = np.mean(value)
            log_data[name] = value
        for prefix, value in successes.items():
            # task success condition (robosuite _check_success), independent of reward shaping
            log_data[prefix+'success_rate'] = np.mean(value)

        return log_data

    def undo_transform_action(self, action):
        raw_shape = action.shape
        if raw_shape[-1] == 20:
            # dual arm
            action = action.reshape(-1,2,10)

        d_rot = action.shape[-1] - 4
        pos = action[...,:3]
        rot = action[...,3:3+d_rot]
        gripper = action[...,[-1]]
        rot = self.rotation_transformer.inverse(rot)
        uaction = np.concatenate([
            pos, rot, gripper
        ], axis=-1)

        if raw_shape[-1] == 20:
            # dual arm
            uaction = uaction.reshape(*raw_shape[:-1], 14)

        return uaction
