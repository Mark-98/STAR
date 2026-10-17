"""
Usage example:
python eval_robomimic_texture_generalization.py \
    --checkpoint checkpoints/square/base_seed0.ckpt \
    --output_dir outputs/square_texture_eval \
    --overwrite \
    --dataset-path datasets/square/ph/image_abs.hdf5\

"""

import sys
# use line-buffering for both stdout and stderr
sys.stdout = open(sys.stdout.fileno(), mode='w', buffering=1)
sys.stderr = open(sys.stderr.fileno(), mode='w', buffering=1)

import csv
import copy
import hashlib
import json
import math
import os
import pathlib
import statistics
import time
from typing import Any, Dict, List, Optional

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

import click
import dill
import hydra
import numpy as np
import torch
import huggingface_hub

try:
    import imageio.v2 as imageio
except ModuleNotFoundError:
    import imageio

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.env.robomimic.rebalanced_difficulty_layout import (
    DEFAULT_LAYOUT_NAME,
    REBALANCED_LAYOUT_NAME,
    SUPPORTED_LAYOUT_NAMES,
    build_rebalanced_scene_records,
)
from diffusion_policy.env.robomimic.robomimic_image_wrapper import (
    RobomimicImageWrapper,
    sample_task_visual_config,
)
from diffusion_policy.env_runner.robomimic_image_runner import (
    create_env,
    create_env_with_texture_mod,
)
from diffusion_policy.gym_util.multistep_wrapper import MultiStepWrapper
from diffusion_policy.gym_util.async_vector_env import AsyncVectorEnv
from diffusion_policy.gym_util.sync_vector_env import SyncVectorEnv
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from diffusion_policy.workspace.base_workspace import BaseWorkspace
import robomimic.utils.file_utils as FileUtils


def ensure_huggingface_hub_compat():
    """
    Older diffusers releases (such as 0.11.1 in this repo) import
    `cached_download` from huggingface_hub, but newer huggingface_hub
    versions removed that symbol. We only need import compatibility here,
    so aliasing to hf_hub_download is enough to unblock loading.
    """
    if hasattr(huggingface_hub, 'cached_download'):
        return

    def _cached_download_compat(*args, **kwargs):
        return huggingface_hub.hf_hub_download(*args, **kwargs)

    huggingface_hub.cached_download = _cached_download_compat


def parse_int_list(text: str) -> List[int]:
    values: List[int] = []
    for chunk in text.split(','):
        chunk = chunk.strip()
        if not chunk:
            continue
        if '-' in chunk:
            start_str, end_str = chunk.split('-', 1)
            start = int(start_str)
            end = int(end_str)
            step = 1 if end >= start else -1
            values.extend(list(range(start, end + step, step)))
        else:
            values.append(int(chunk))
    return values


def resolve_path(path_str: str) -> str:
    path = pathlib.Path(os.path.expanduser(path_str))
    if path.is_absolute():
        return str(path)
    repo_candidate = REPO_ROOT / path
    if repo_candidate.exists():
        return str(repo_candidate)
    return str(path)


def load_policy_from_checkpoint(
    checkpoint: str,
    output_dir: str,
    device: torch.device,
    weight_source: str = 'auto',
):
    if weight_source not in ('auto', 'model', 'ema'):
        raise ValueError(f'Unknown checkpoint weight source: {weight_source}')
    ensure_huggingface_hub_compat()
    with open(checkpoint, 'rb') as stream:
        payload = torch.load(stream, pickle_module=dill, map_location='cpu')
    cfg = copy.deepcopy(payload['cfg'])
    # Evaluation restores a complete strict state dict, not training warm-starts.
    from omegaconf import open_dict
    with open_dict(cfg):
        if 'pretrained_checkpoint' in cfg:
            cfg.pretrained_checkpoint = None
        for key in ('pretrained_student_checkpoint_path', 'teacher_checkpoint_path'):
            if key in cfg.policy:
                cfg.policy[key] = None
    use_ema = bool(cfg.training.use_ema) if weight_source == 'auto' else weight_source == 'ema'
    if use_ema and (not cfg.training.use_ema or 'ema_model' not in payload['state_dicts']):
        raise ValueError('EMA weights are unavailable for this checkpoint.')
    cls = hydra.utils.get_class(cfg._target_)
    workspace = cls(cfg, output_dir=output_dir)
    workspace: BaseWorkspace
    workspace.load_payload(payload, exclude_keys=None, include_keys=[])

    policy = workspace.model
    if use_ema:
        policy = workspace.ema_model

    policy.to(device)
    policy.eval()
    return cfg, policy


def state_dict_sha256(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(f'{key}:{tensor.dtype}:{tuple(tensor.shape)}'.encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def replace_frozen_actor_with_base_ema(policy, base_checkpoint):
    if any(parameter.requires_grad for parameter in policy.model.parameters()):
        raise ValueError('Actor replacement requires a frozen diffusion model.')
    ensure_huggingface_hub_compat()
    with open(base_checkpoint, 'rb') as stream:
        payload = torch.load(stream, pickle_module=dill, map_location='cpu')
    states = payload['state_dicts']
    if 'model' not in states or 'ema_model' not in states:
        raise ValueError('Both raw and EMA Base weights are required.')
    raw_actor = {key[len('model.'):]: value for key, value in states['model'].items()
                 if key.startswith('model.')}
    ema_actor = {key[len('model.'):]: value for key, value in states['ema_model'].items()
                 if key.startswith('model.')}
    current = policy.model.state_dict()
    if not ema_actor or current.keys() != ema_actor.keys():
        raise ValueError('EMA actor state keys do not match Ours.')
    if any(current[key].shape != ema_actor[key].shape or current[key].dtype != ema_actor[key].dtype
           for key in current):
        raise ValueError('EMA actor tensor shapes or dtypes do not match Ours.')
    before_actor = state_dict_sha256(current)
    if before_actor != state_dict_sha256(raw_actor):
        raise ValueError('Ours actor does not match the paired Base raw actor; refusing replacement.')
    before_other = state_dict_sha256({key: value for key, value in policy.state_dict().items()
                                     if not key.startswith('model.')})
    policy.model.load_state_dict(ema_actor, strict=True)
    after_actor = state_dict_sha256(policy.model.state_dict())
    source_actor = state_dict_sha256(ema_actor)
    after_other = state_dict_sha256({key: value for key, value in policy.state_dict().items()
                                    if not key.startswith('model.')})
    if after_actor != source_actor or before_other != after_other:
        raise RuntimeError('Actor-only EMA replacement verification failed.')
    return {
        'source_checkpoint': str(pathlib.Path(base_checkpoint).resolve()),
        'source_state_dict': 'ema_model', 'component': 'model',
        'actor_tensor_count': len(ema_actor), 'previous_actor_sha256': before_actor,
        'source_actor_sha256': source_actor, 'loaded_actor_sha256': after_actor,
        'non_actor_sha256_before': before_other, 'non_actor_sha256_after': after_other,
        'non_actor_weights_unchanged': True, 'matches_base_raw_before': True,
        'actor_changed': before_actor != after_actor,
    }


def maybe_disable_visual_components(
    visual_config: Dict[str, Any],
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
) -> Dict[str, Any]:
    result = {
        'texture_map': dict(visual_config['texture_map']),
        'material_map': visual_config['material_map'],
        'skybox_config': visual_config['skybox_config'],
        'light_config': visual_config['light_config'],
    }
    if disable_material:
        result['material_map'] = None
    if disable_skybox:
        result['skybox_config'] = None
    if disable_light:
        result['light_config'] = None
    return result


def resolve_visual_config(
    env_meta: Dict[str, Any],
    apply_visual_mod: bool,
    texture_seed: Optional[int],
    visual_mode: Optional[str],
    scene_id: Optional[int],
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
) -> Optional[Dict[str, Any]]:
    if not apply_visual_mod:
        return None
    return maybe_disable_visual_components(
        sample_task_visual_config(
            env_meta['env_name'],
            seed=texture_seed,
            visual_mode=visual_mode,
            scene_id=scene_id,
        ),
        disable_material=disable_material,
        disable_skybox=disable_skybox,
        disable_light=disable_light,
    )


def create_wrapped_eval_env(
    cfg,
    env_meta: Dict[str, Any],
    env_seed: int,
    apply_visual_mod: bool = False,
    visual_config: Optional[Dict[str, Any]] = None,
    texture_seed: Optional[int] = None,
    visual_mode: Optional[str] = None,
    scene_id: Optional[int] = None,
    enable_render: bool = True,
    exact_visual_config: bool = False,
):
    shape_meta = cfg.task.env_runner.shape_meta
    render_obs_key = cfg.task.env_runner.render_obs_key

    if not apply_visual_mod:
        base_env = create_env(
            env_meta=env_meta,
            shape_meta=shape_meta,
            enable_render=enable_render,
        )
        base_env.env.hard_reset = False
        wrapped_env = RobomimicImageWrapper(
            env=base_env,
            shape_meta=shape_meta,
            init_state=None,
            render_obs_key=render_obs_key,
        )
    else:
        if visual_config is None:
            raise ValueError('visual_config is required when apply_visual_mod=True')
        wrapped_env = create_env_with_texture_mod(
            env_meta=env_meta,
            shape_meta=shape_meta,
            enable_render=enable_render,
            render_obs_key=render_obs_key,
            task_name=env_meta['env_name'],
            texture_map=visual_config['texture_map'],
            material_map=visual_config['material_map'],
            skybox_config=visual_config['skybox_config'],
            light_config=visual_config['light_config'],
            texture_seed=texture_seed,
            visual_mode=visual_mode,
            scene_id=scene_id,
            randomize_on_reset=False,
        )
        wrapped_env.env.env.hard_reset = False
        if exact_visual_config:
            # A saved scene's explicit nulls must not trigger new sampling.
            wrapped_env.visual_config_override = copy.deepcopy(visual_config)

    wrapped_env.seed(env_seed)

    env = MultiStepWrapper(
        wrapped_env,
        n_obs_steps=cfg.task.env_runner.n_obs_steps,
        n_action_steps=cfg.task.env_runner.n_action_steps,
        max_episode_steps=cfg.task.env_runner.max_steps,
    )
    return env


def build_eval_env(
    cfg,
    env_meta: Dict[str, Any],
    env_seed: int,
    apply_visual_mod: bool = False,
    texture_seed: Optional[int] = None,
    visual_mode: Optional[str] = None,
    scene_id: Optional[int] = None,
    disable_material: bool = True,
    disable_skybox: bool = True,
    disable_light: bool = True,
):
    visual_config = resolve_visual_config(
        env_meta=env_meta,
        apply_visual_mod=apply_visual_mod,
        texture_seed=texture_seed,
        visual_mode=visual_mode,
        scene_id=scene_id,
        disable_material=disable_material,
        disable_skybox=disable_skybox,
        disable_light=disable_light,
    )
    env = create_wrapped_eval_env(
        cfg=cfg,
        env_meta=env_meta,
        env_seed=env_seed,
        apply_visual_mod=apply_visual_mod,
        visual_config=visual_config,
        texture_seed=texture_seed,
        visual_mode=visual_mode,
        scene_id=scene_id,
        enable_render=True,
    )
    return env, visual_config


def seed_policy_rng(seed: int, device: torch.device):
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(seed)


def undo_transform_action(
    action: np.ndarray,
    rotation_transformer: Optional[RotationTransformer],
) -> np.ndarray:
    if rotation_transformer is None:
        return action

    raw_shape = action.shape
    if raw_shape[-1] == 20:
        action = action.reshape(-1, 2, 10)

    d_rot = action.shape[-1] - 4
    pos = action[..., :3]
    rot = action[..., 3:3 + d_rot]
    gripper = action[..., [-1]]
    rot = rotation_transformer.inverse(rot)
    uaction = np.concatenate([pos, rot, gripper], axis=-1)

    if raw_shape[-1] == 20:
        uaction = uaction.reshape(*raw_shape[:-1], 14)
    return uaction


def save_video_frames(
    frames: List[np.ndarray],
    output_path: str,
    fps: int = 10,
):
    if len(frames) == 0:
        return None
    output_file = pathlib.Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    ext = output_file.suffix.lower()
    if ext == '.gif':
        imageio.mimsave(output_file, frames, format='GIF', fps=fps)
    else:
        imageio.mimsave(output_file, frames, fps=fps)
    return str(output_file)


def extract_success_flag(info: Optional[Dict[str, Any]], env) -> bool:
    def _to_bool(value: Any) -> bool:
        if isinstance(value, dict):
            if 'task' in value:
                return _to_bool(value['task'])
            return any(_to_bool(v) for v in value.values())
        if isinstance(value, np.ndarray):
            if value.size == 0:
                return False
            return bool(np.max(value))
        if isinstance(value, (list, tuple)):
            if len(value) == 0:
                return False
            return any(_to_bool(v) for v in value)
        return bool(value)

    if isinstance(info, dict):
        for key in ('success', 'is_success', 'task'):
            if key not in info:
                continue
            value = info[key]
            return _to_bool(value)

    if env is not None and hasattr(env, 'is_success'):
        value = env.is_success()
        return _to_bool(value)

    return False


def initial_state_record(state):
    values = np.asarray(state['states'], dtype='<f8').reshape(-1)
    if not np.all(np.isfinite(values)):
        raise ValueError('Non-finite simulator initial state.')
    return {
        'initial_state': values.tolist(),
        'initial_state_hash': hashlib.sha256(values.tobytes()).hexdigest(),
    }


def rollout_episode(
    policy,
    env,
    max_steps: int,
    abs_action: bool,
    rotation_transformer: Optional[RotationTransformer],
    policy_seed: int,
    save_video_path: Optional[str] = None,
    video_fps: int = 10,
    capture_initial_frame: bool = True,
    record_initial_state: bool = False,
) -> Dict[str, Any]:
    device = policy.device
    obs = env.reset()
    initial = initial_state_record(env.get_state()) if record_initial_state else {}
    policy.reset()
    seed_policy_rng(policy_seed, device)

    total_reward = 0.0
    reward_history: List[float] = []
    success = False
    done = False
    horizon = 0
    max_policy_steps = math.ceil(max_steps / env.n_action_steps) + 1
    frames: List[np.ndarray] = []

    if save_video_path is not None and capture_initial_frame:
        frames.append(env.render())

    while (not done) and (not success) and (horizon < max_policy_steps):
        np_obs_dict = dict(obs)
        obs_dict = dict_apply(
            np_obs_dict,
            lambda x: torch.from_numpy(x[None].astype(np.float32, copy=False)).to(device=device),
        )

        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        np_action_dict = dict_apply(
            action_dict,
            lambda x: x.detach().to('cpu').numpy(),
        )
        action = np_action_dict['action'][0]
        if not np.all(np.isfinite(action)):
            raise RuntimeError('Nan or Inf action encountered during evaluation.')

        env_action = undo_transform_action(action, rotation_transformer) if abs_action else action
        obs, reward, done, info = env.step(env_action)
        total_reward += float(reward)
        reward_history.append(float(reward))
        success = extract_success_flag(info, env)
        horizon += 1
        if save_video_path is not None:
            frames.append(env.render())

    saved_video_path = None
    if save_video_path is not None:
        saved_video_path = save_video_frames(
            frames=frames,
            output_path=save_video_path,
            fps=video_fps,
        )

    return {
        **initial,
        'policy_batch_seed': int(policy_seed),
        'policy_batch_lane': 0,
        'return': total_reward,
        'max_reward': max(reward_history) if reward_history else 0.0,
        'success_rate': float(success),
        'horizon': horizon,
        'video_path': saved_video_path,
    }


def build_rollout_specs(
    env_meta: Dict[str, Any],
    n_base: int,
    base_start_seed: int,
    texture_eval_seed: int,
    texture_seed_values: List[int],
    scene_id_values: Optional[List[int]],
    scene_source_records: Optional[List[Dict[str, Any]]],
    visual_mode: str,
    scene_id: Optional[int],
    policy_seed: int,
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
    save_video: bool,
    videos_dir: pathlib.Path,
    difficulty_layout: str = DEFAULT_LAYOUT_NAME,
) -> Dict[str, List[Dict[str, Any]]]:
    original_specs: List[Dict[str, Any]] = []
    modified_specs: List[Dict[str, Any]] = []

    for rollout_idx in range(n_base):
        env_seed = int(base_start_seed + rollout_idx)
        original_specs.append({
            'split': 'original_env',
            'rollout_index': rollout_idx,
            'env_seed': env_seed,
            'texture_seed': None,
            'scene_id': None,
            'apply_visual_mod': False,
            'visual_mode': visual_mode,
            'visual_config': None,
            'policy_seed': int(policy_seed + rollout_idx),
            'video_path': (
                str(videos_dir / 'original_env' / f'episode_{rollout_idx:02d}_seed_{env_seed}.mp4')
                if save_video else None
            ),
        })

    use_scene_sweep = (scene_id_values is not None) or (scene_source_records is not None)
    rebalanced_records = None
    if (
        use_scene_sweep
        and scene_source_records is None
        and difficulty_layout != DEFAULT_LAYOUT_NAME
    ):
        rebalanced_records = build_rebalanced_scene_records(
            task_name=env_meta['env_name'],
            visual_mode=visual_mode,
            layout_name=difficulty_layout,
        )
    if use_scene_sweep:
        if scene_id_values is not None:
            modified_values = scene_id_values
        elif scene_source_records is not None:
            modified_values = list(range(len(scene_source_records)))
        else:
            modified_values = []
    else:
        modified_values = texture_seed_values
    modified_split = 'train_scene' if use_scene_sweep else 'texture'
    for rollout_idx, modified_value in enumerate(modified_values):
        current_scene_id = int(modified_value) if use_scene_sweep else scene_id
        current_texture_seed = None if use_scene_sweep else int(modified_value)
        source_split = None
        source_visual_mode = None
        source_scene_id = None
        source_rollout_index = None

        if scene_source_records is not None:
            if current_scene_id < 0 or current_scene_id >= len(scene_source_records):
                raise ValueError(
                    f'scene_id={current_scene_id} is out of range for the provided '
                    f'scene source file. Available range: 0..{len(scene_source_records) - 1}.'
                )
            record = scene_source_records[current_scene_id]
            visual_config = {
                'texture_map': dict(record['texture_map']),
                'material_map': None if record.get('material_map') is None else dict(record['material_map']),
                'skybox_config': None if record.get('skybox_config') is None else dict(record['skybox_config']),
                'light_config': None if record.get('light_config') is None else dict(record['light_config']),
            }
            source_split = record.get('source_split')
            source_visual_mode = record.get('source_visual_mode')
            source_scene_id = record.get('source_scene_id')
            source_rollout_index = record.get('source_rollout_index')
        elif rebalanced_records is not None:
            if current_scene_id < 0 or current_scene_id >= len(rebalanced_records):
                raise ValueError(
                    f'scene_id={current_scene_id} is out of range for '
                    f'{env_meta["env_name"]} {visual_mode} {difficulty_layout}. '
                    f'Available range: 0..{len(rebalanced_records) - 1}.'
                )
            record = rebalanced_records[current_scene_id]
            visual_config = {
                'texture_map': dict(record['texture_map']),
                'material_map': None if record.get('material_map') is None else dict(record['material_map']),
                'skybox_config': None if record.get('skybox_config') is None else dict(record['skybox_config']),
                'light_config': None if record.get('light_config') is None else dict(record['light_config']),
            }
            source_split = record.get('source_split')
            source_visual_mode = record.get('source_visual_mode')
            source_scene_id = record.get('source_scene_id')
            source_rollout_index = record.get('source_rollout_index')
        else:
            visual_config = resolve_visual_config(
                env_meta=env_meta,
                apply_visual_mod=True,
                texture_seed=current_texture_seed,
                visual_mode=visual_mode,
                scene_id=current_scene_id,
                disable_material=disable_material,
                disable_skybox=disable_skybox,
                disable_light=disable_light,
            )
        video_name = (
            f'scene_{int(modified_value):02d}.mp4'
            if use_scene_sweep else
            f'texture_{int(modified_value):02d}.mp4'
        )
        modified_specs.append({
            'split': modified_split,
            'rollout_index': rollout_idx,
            'env_seed': int(texture_eval_seed),
            'texture_seed': current_texture_seed,
            'scene_id': current_scene_id,
            'apply_visual_mod': True,
            'visual_mode': visual_mode,
            'visual_config': visual_config,
            'policy_seed': int(policy_seed + int(modified_value)),
            'source_split': source_split,
            'source_visual_mode': source_visual_mode,
            'source_scene_id': source_scene_id,
            'source_rollout_index': source_rollout_index,
            'video_path': (
                str(videos_dir / modified_split / video_name)
                if save_video else None
            ),
        })

    return {
        'original_specs': original_specs,
        'modified_specs': modified_specs,
        'modified_split': modified_split,
    }


def load_scene_source_records(scene_source_file: str) -> List[Dict[str, Any]]:
    source_path = pathlib.Path(resolve_path(scene_source_file))
    with open(source_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)

    if isinstance(payload, list):
        raw_records = payload
        payload_visual_mode = None
        payload_split = None
    elif isinstance(payload, dict):
        raw_records = payload.get('records')
        if raw_records is None:
            raw_records = payload.get('scene_records')
        if raw_records is None:
            raise click.ClickException(
                f'Scene source file must contain `records` or `scene_records`: {source_path}'
            )
        payload_visual_mode = payload.get('visual_mode')
        payload_split = payload.get('source_split')
    else:
        raise click.ClickException(
            f'Unsupported scene source file format: {source_path}'
        )

    records: List[Dict[str, Any]] = []
    for idx, record in enumerate(raw_records):
        if not isinstance(record, dict):
            raise click.ClickException(
                f'Invalid record at index {idx} in {source_path}: expected object.'
            )
        texture_map = record.get('texture_map')
        if texture_map is None:
            raise click.ClickException(
                f'Missing `texture_map` at index {idx} in {source_path}.'
            )
        records.append({
            'texture_map': dict(texture_map),
            'eval_seed': record.get('eval_seed'),
            'material_map': None if record.get('material_map') is None else dict(record['material_map']),
            'skybox_config': None if record.get('skybox_config') is None else dict(record['skybox_config']),
            'light_config': None if record.get('light_config') is None else dict(record['light_config']),
            'source_split': record.get('source_split', payload_split),
            'source_visual_mode': record.get('source_visual_mode', payload_visual_mode),
            'source_scene_id': record.get('source_scene_id', record.get('scene_id', idx)),
            'source_rollout_index': record.get('source_rollout_index', record.get('rollout_index', idx)),
        })
    return records


def materialize_rollout_record(spec: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    visual_config = spec.get('visual_config')
    return {
        'split': spec['split'],
        'rollout_index': int(spec['rollout_index']),
        'env_seed': int(spec['env_seed']),
        'texture_seed': spec.get('texture_seed'),
        'scene_id': spec.get('scene_id'),
        'source_split': spec.get('source_split'),
        'source_visual_mode': spec.get('source_visual_mode'),
        'source_scene_id': spec.get('source_scene_id'),
        'source_rollout_index': spec.get('source_rollout_index'),
        **result,
        'texture_map': None if visual_config is None else visual_config['texture_map'],
        'material_map': None if visual_config is None else visual_config['material_map'],
        'skybox_config': None if visual_config is None else visual_config['skybox_config'],
        'light_config': None if visual_config is None else visual_config['light_config'],
    }


def csv_record_from_rollout(record: Dict[str, Any]) -> Dict[str, Any]:
    return {
        **record,
        'source_split': record.get('source_split', '') or '',
        'source_visual_mode': record.get('source_visual_mode', '') or '',
        'source_scene_id': '' if record.get('source_scene_id') is None else int(record['source_scene_id']),
        'source_rollout_index': '' if record.get('source_rollout_index') is None else int(record['source_rollout_index']),
        'texture_map': '' if record.get('texture_map') is None else json_dumps_compact(record['texture_map']),
        'material_map': '' if record.get('material_map') is None else json_dumps_compact(record['material_map']),
        'skybox_config': '' if record.get('skybox_config') is None else json_dumps_compact(record['skybox_config']),
        'light_config': '' if record.get('light_config') is None else json_dumps_compact(record['light_config']),
        'video_path': record.get('video_path', '') or '',
    }


def print_rollout_progress(record: Dict[str, Any], total_count: int):
    split = str(record['split'])
    if split == 'original_env':
        label = f"seed={record['env_seed']}"
    else:
        if record.get('texture_seed') is None:
            label = f"scene_id={record['scene_id']}"
        else:
            label = f"texture_seed={record['texture_seed']}"
    print(
        f"[{split} {int(record['rollout_index']) + 1}/{total_count}] "
        f"{label} success={record['success_rate']:.3f} "
        f"max_reward={record['max_reward']:.3f} return={record['return']:.3f}"
    )


def build_vectorized_eval_env(
    cfg,
    env_meta: Dict[str, Any],
    chunk_specs: List[Dict[str, Any]],
    vector_backend: str = 'sync',
):
    if vector_backend not in ('sync', 'async'):
        raise ValueError(f'Unknown vector backend: {vector_backend}')
    env_fns = []
    for spec in chunk_specs:
        def _env_fn(spec=spec):
            return create_wrapped_eval_env(
                cfg=cfg,
                env_meta=env_meta,
                env_seed=int(spec['env_seed']),
                apply_visual_mod=bool(spec['apply_visual_mod']),
                visual_config=spec.get('visual_config'),
                texture_seed=spec.get('texture_seed'),
                visual_mode=spec.get('visual_mode'),
                scene_id=spec.get('scene_id'),
                enable_render=True,
                exact_visual_config=bool(spec.get('exact_visual_config', False)),
            )
        env_fns.append(_env_fn)

    if vector_backend == 'async':
        def dummy_env_fn():
            return create_wrapped_eval_env(
                cfg=cfg,
                env_meta=env_meta,
                env_seed=int(chunk_specs[0]['env_seed']),
                enable_render=False,
            )

        # Spawn avoids inheriting the policy's CUDA and simulator GL contexts.
        return AsyncVectorEnv(
            env_fns, dummy_env_fn=dummy_env_fn, context='spawn',
        )
    return SyncVectorEnv(env_fns)


def rollout_episodes_vectorized(
    policy,
    env,
    specs: List[Dict[str, Any]],
    max_steps: int,
    n_action_steps: int,
    abs_action: bool,
    rotation_transformer: Optional[RotationTransformer],
    video_fps: int,
    capture_initial_frame: bool = True,
) -> List[Dict[str, Any]]:
    device = policy.device
    obs = env.reset()
    initial_states = (
        env.call('get_state') if any(spec.get('record_initial_state', False) for spec in specs)
        else None
    )
    policy.reset()
    is_async = isinstance(env, AsyncVectorEnv)
    underlying_envs = None if is_async else env.call('get_attr', 'env')
    if len(specs) > 0:
        seed_policy_rng(int(specs[0]['policy_seed']), device)

    n_envs = len(specs)
    total_rewards = np.zeros(n_envs, dtype=np.float64)
    max_rewards = np.full(n_envs, -np.inf, dtype=np.float64)
    successes = np.zeros(n_envs, dtype=np.bool_)
    finished = np.zeros(n_envs, dtype=np.bool_)
    horizons = np.zeros(n_envs, dtype=np.int32)
    max_policy_steps = math.ceil(max_steps / n_action_steps) + 1
    policy_steps = 0
    last_progress_time = time.monotonic()

    save_video = any(spec.get('video_path') is not None for spec in specs)
    all_frames: Optional[List[List[np.ndarray]]] = None
    if save_video:
        all_frames = [[] for _ in range(n_envs)]
        if capture_initial_frame:
            initial_frames = env.render()
            for env_idx, frame in enumerate(initial_frames):
                all_frames[env_idx].append(frame)

    while not np.all(finished):
        obs_dict = dict_apply(
            dict(obs),
            lambda x: torch.from_numpy(x.astype(np.float32, copy=False)).to(device=device),
        )
        with torch.no_grad():
            action_dict = policy.predict_action(obs_dict)

        np_action_dict = dict_apply(
            action_dict,
            lambda x: x.detach().to('cpu').numpy(),
        )
        action = np_action_dict['action']
        if not np.all(np.isfinite(action)):
            raise RuntimeError('Nan or Inf action encountered during evaluation.')

        env_action = undo_transform_action(action, rotation_transformer) if abs_action else action
        obs, reward, done, info = env.step(env_action)
        # Query success inside each worker; live simulator objects cannot be sent back.
        worker_success = env.call('is_success') if is_async else None

        active_indices = np.where(~finished)[0]
        for env_idx in active_indices:
            total_rewards[env_idx] += float(reward[env_idx])
            max_rewards[env_idx] = max(max_rewards[env_idx], float(reward[env_idx]))
            horizons[env_idx] += 1
            success_info = info[env_idx]
            if is_async:
                if not isinstance(success_info, dict) or not any(
                    key in success_info for key in ('success', 'is_success', 'task')
                ):
                    success_info = {'success': worker_success[env_idx]}
                success = extract_success_flag(success_info, env=None)
            else:
                success = extract_success_flag(success_info, env=underlying_envs[env_idx])
            if success:
                successes[env_idx] = True
            if bool(done[env_idx]) or bool(successes[env_idx]) or int(horizons[env_idx]) >= max_policy_steps:
                finished[env_idx] = True

        if all_frames is not None:
            rendered_frames = env.render()
            for env_idx in active_indices:
                all_frames[env_idx].append(rendered_frames[env_idx])

        policy_steps += 1
        if time.monotonic() - last_progress_time >= 30:
            print(
                f'  policy step {policy_steps}/{max_policy_steps}, '
                f'finished {int(finished.sum())}/{n_envs}, '
                f'successes {int(successes.sum())}/{n_envs}',
                flush=True,
            )
            last_progress_time = time.monotonic()

    results: List[Dict[str, Any]] = []
    for env_idx, spec in enumerate(specs):
        saved_video_path = None
        if all_frames is not None and spec.get('video_path') is not None:
            saved_video_path = save_video_frames(
                frames=all_frames[env_idx],
                output_path=str(spec['video_path']),
                fps=video_fps,
            )
        max_reward = 0.0 if int(horizons[env_idx]) == 0 else float(max_rewards[env_idx])
        results.append({
            **(initial_state_record(initial_states[env_idx])
               if initial_states is not None and spec.get('record_initial_state', False) else {}),
            'policy_batch_seed': int(specs[0]['policy_seed']),
            'policy_batch_lane': env_idx,
            'return': float(total_rewards[env_idx]),
            'max_reward': max_reward,
            'success_rate': float(successes[env_idx]),
            'horizon': int(horizons[env_idx]),
            'video_path': saved_video_path,
        })
    return results


def evaluate_rollout_specs(
    policy,
    cfg,
    env_meta: Dict[str, Any],
    specs: List[Dict[str, Any]],
    n_envs: int,
    abs_action: bool,
    rotation_transformer: Optional[RotationTransformer],
    vector_backend: str = 'sync',
) -> List[Dict[str, Any]]:
    if vector_backend not in ('sync', 'async'):
        raise ValueError(f'Unknown vector backend: {vector_backend}')
    if len(specs) == 0:
        return []

    records: List[Dict[str, Any]] = []
    max_steps = int(cfg.task.env_runner.max_steps)
    n_action_steps = int(cfg.task.env_runner.n_action_steps)
    video_fps = int(cfg.task.env_runner.fps)

    if n_envs <= 1:
        for spec in specs:
            env = create_wrapped_eval_env(
                cfg=cfg,
                env_meta=env_meta,
                env_seed=int(spec['env_seed']),
                apply_visual_mod=bool(spec['apply_visual_mod']),
                visual_config=spec.get('visual_config'),
                texture_seed=spec.get('texture_seed'),
                visual_mode=spec.get('visual_mode'),
                scene_id=spec.get('scene_id'),
                enable_render=True,
                exact_visual_config=bool(spec.get('exact_visual_config', False)),
            )
            try:
                result = rollout_episode(
                    policy=policy,
                    env=env,
                    max_steps=max_steps,
                    abs_action=abs_action,
                    rotation_transformer=rotation_transformer,
                    policy_seed=int(spec['policy_seed']),
                    save_video_path=spec.get('video_path'),
                    video_fps=video_fps,
                    record_initial_state=bool(spec.get('record_initial_state', False)),
                )
            finally:
                env.close()
            record = materialize_rollout_record(spec, result)
            print_rollout_progress(record, total_count=len(specs))
            records.append(record)
        return records

    for start_idx in range(0, len(specs), n_envs):
        chunk_specs = specs[start_idx:start_idx + n_envs]
        print(
            f'Rollouts {start_idx + 1}-{start_idx + len(chunk_specs)}/{len(specs)}: '
            f'backend={vector_backend}, environments={len(chunk_specs)}',
            flush=True,
        )
        env = build_vectorized_eval_env(
            cfg=cfg,
            env_meta=env_meta,
            chunk_specs=chunk_specs,
            vector_backend=vector_backend,
        )
        if vector_backend == 'async':
            print(f'  environment worker PIDs: {[p.pid for p in env.processes]}', flush=True)
        try:
            chunk_results = rollout_episodes_vectorized(
                policy=policy,
                env=env,
                specs=chunk_specs,
                max_steps=max_steps,
                n_action_steps=n_action_steps,
                abs_action=abs_action,
                rotation_transformer=rotation_transformer,
                video_fps=video_fps,
            )
        finally:
            env.close()

        for spec, result in zip(chunk_specs, chunk_results):
            record = materialize_rollout_record(spec, result)
            print_rollout_progress(record, total_count=len(specs))
            records.append(record)
    return records


def summarize_records(records: List[Dict[str, Any]]) -> Dict[str, float]:
    if len(records) == 0:
        return {
            'count': 0,
            'mean_return': 0.0,
            'std_return': 0.0,
            'mean_max_reward': 0.0,
            'std_max_reward': 0.0,
            'mean_success_rate': 0.0,
            'std_success_rate': 0.0,
            'mean_horizon': 0.0,
            'std_horizon': 0.0,
        }

    def _values(key: str) -> List[float]:
        return [float(record[key]) for record in records]

    def _mean(values: List[float]) -> float:
        return float(statistics.mean(values))

    def _std(values: List[float]) -> float:
        return float(statistics.pstdev(values)) if len(values) > 1 else 0.0

    returns = _values('return')
    max_rewards = _values('max_reward')
    success_rates = _values('success_rate')
    horizons = _values('horizon')
    return {
        'count': len(records),
        'mean_return': _mean(returns),
        'std_return': _std(returns),
        'mean_max_reward': _mean(max_rewards),
        'std_max_reward': _std(max_rewards),
        'mean_success_rate': _mean(success_rates),
        'std_success_rate': _std(success_rates),
        'mean_horizon': _mean(horizons),
        'std_horizon': _std(horizons),
    }


def json_dumps_compact(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, sort_keys=True)


@click.command()
@click.option('-c', '--checkpoint', required=True, type=click.Path(exists=True, dir_okay=False))
@click.option('-o', '--output_dir', required=True, type=click.Path())
@click.option('--dataset-path', type=click.Path(dir_okay=False), default=None, help='Optional override for cfg.task.dataset.dataset_path')
@click.option('-d', '--device', default='cuda:0')
@click.option('--n-base', default=10, type=int, show_default=True)
@click.option('--n-envs', default=1, type=int, show_default=True, help='Number of environments to evaluate in parallel. Use 1 for the original sequential behavior.')
@click.option('--base-start-seed', type=int, default=None, help='Default: cfg.task.env_runner.test_start_seed')
@click.option('--texture-seeds', default='0-19', show_default=True, help='Comma list or ranges, e.g. 0-19 or 0,3,7')
@click.option('--scene-ids', default=None, help='Optional scene id list (e.g. 0-19). If set, evaluate train-scene sweeps instead of texture seeds.')
@click.option('--scene-source-file', type=click.Path(exists=True, dir_okay=False), default=None, help='Optional JSON manifest with explicit scene visual configs. If set, evaluate these scene records instead of sampling by mode.')
@click.option('--texture-eval-seed', type=int, default=None, help='Env seed used for all texture runs. Default: base-start-seed')
@click.option('--visual-mode', type=click.Choice(['train', 'eval-easy', 'eval-medium', 'eval-hard']), default='eval-easy', show_default=True)
@click.option('--scene-id', type=int, default=None, help='Optional DAD-style scene id. If set and texture seed is omitted for sampling, mode-specific scene seeds are used.')
@click.option('--policy-seed', default=0, type=int, show_default=True)
@click.option('--disable-material/--enable-material', default=True, show_default=True)
@click.option('--disable-skybox/--enable-skybox', default=True, show_default=True)
@click.option('--disable-light/--enable-light', default=True, show_default=True)
@click.option('--skip-original-env', is_flag=True, help='Skip original_env rollouts and only evaluate modified environments.')
@click.option('--save-video/--no-save-video', default=True, show_default=True, help='Save rollout videos to the eval output directory.')
@click.option('--difficulty-layout', type=click.Choice(sorted(SUPPORTED_LAYOUT_NAMES)), default=DEFAULT_LAYOUT_NAME, show_default=True)
@click.option('--overwrite', is_flag=True)
def main(
    checkpoint: str,
    output_dir: str,
    dataset_path: Optional[str],
    device: str,
    n_base: int,
    n_envs: int,
    base_start_seed: Optional[int],
    texture_seeds: str,
    scene_ids: Optional[str],
    scene_source_file: Optional[str],
    texture_eval_seed: Optional[int],
    visual_mode: str,
    scene_id: Optional[int],
    policy_seed: int,
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
    skip_original_env: bool,
    save_video: bool,
    difficulty_layout: str,
    overwrite: bool,
):
    output_dir = resolve_path(output_dir)
    out_path = pathlib.Path(output_dir)
    if out_path.exists() and (not overwrite):
        raise click.ClickException(
            f'Output path already exists: {output_dir}. Pass --overwrite to reuse it.'
        )
    out_path.mkdir(parents=True, exist_ok=True)

    checkpoint = resolve_path(checkpoint)
    device_obj = torch.device(device)
    if n_envs < 1:
        raise click.ClickException(f'--n-envs must be >= 1, got {n_envs}')
    cfg, policy = load_policy_from_checkpoint(
        checkpoint=checkpoint,
        output_dir=output_dir,
        device=device_obj,
    )

    if dataset_path is None:
        dataset_path = resolve_path(cfg.task.dataset.dataset_path)
    else:
        dataset_path = resolve_path(dataset_path)

    if not pathlib.Path(dataset_path).exists():
        raise click.ClickException(
            'Dataset file not found: '
            f'{dataset_path}\n'
            f'Checkpoint config expects: {cfg.task.dataset.dataset_path}\n'
            'Pass --dataset-path /abs/path/to/image_abs.hdf5 to override it.'
        )

    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False

    if base_start_seed is None:
        base_start_seed = int(cfg.task.env_runner.test_start_seed)
    if texture_eval_seed is None:
        texture_eval_seed = int(base_start_seed)

    texture_seed_values = parse_int_list(texture_seeds)
    scene_id_values = parse_int_list(scene_ids) if scene_ids is not None else None
    scene_source_records = (
        load_scene_source_records(scene_source_file)
        if scene_source_file is not None else None
    )
    abs_action = bool(cfg.task.env_runner.abs_action)
    rotation_transformer = None
    if abs_action:
        env_meta = copy.deepcopy(env_meta)
        env_meta['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')

    base_records: List[Dict[str, Any]] = []
    modified_records: List[Dict[str, Any]] = []
    csv_records: List[Dict[str, Any]] = []
    videos_dir = out_path / 'videos'

    print(f'Loaded checkpoint: {checkpoint}')
    print(f'Task: {env_meta["env_name"]}')
    print(f'Dataset: {dataset_path}')
    print(f"Control delta: {env_meta['env_kwargs']['controller_configs'].get('control_delta')}")
    print(f'Base seeds: {base_start_seed}..{base_start_seed + n_base - 1}')
    print(f'Texture seeds: {texture_seed_values}')
    print(f'Scene ids: {scene_id_values}')
    print(f'Scene source file: {scene_source_file}')
    print(f'Texture eval env seed: {texture_eval_seed}')
    print(f'Visual mode: {visual_mode}')
    print(f'Difficulty layout: {difficulty_layout}')
    print(f'Scene id: {scene_id}')
    print(f'Parallel envs: {n_envs}')
    print(f'Skip original env: {skip_original_env}')
    print(f'Save video: {save_video}')

    rollout_spec_bundle = build_rollout_specs(
        env_meta=env_meta,
        n_base=n_base,
        base_start_seed=int(base_start_seed),
        texture_eval_seed=int(texture_eval_seed),
        texture_seed_values=texture_seed_values,
        scene_id_values=scene_id_values,
        scene_source_records=scene_source_records,
        visual_mode=visual_mode,
        scene_id=scene_id,
        policy_seed=policy_seed,
        disable_material=disable_material,
        disable_skybox=disable_skybox,
        disable_light=disable_light,
        save_video=save_video,
        videos_dir=videos_dir,
        difficulty_layout=difficulty_layout,
    )
    modified_split = str(rollout_spec_bundle['modified_split'])

    if not skip_original_env:
        base_records = evaluate_rollout_specs(
            policy=policy,
            cfg=cfg,
            env_meta=env_meta,
            specs=rollout_spec_bundle['original_specs'],
            n_envs=n_envs,
            abs_action=abs_action,
            rotation_transformer=rotation_transformer,
        )

    modified_records = evaluate_rollout_specs(
        policy=policy,
        cfg=cfg,
        env_meta=env_meta,
        specs=rollout_spec_bundle['modified_specs'],
        n_envs=n_envs,
        abs_action=abs_action,
        rotation_transformer=rotation_transformer,
    )

    csv_records = [csv_record_from_rollout(record) for record in (base_records + modified_records)]

    summary = {
        'checkpoint': checkpoint,
        'dataset_path': dataset_path,
        'task_name': env_meta['env_name'],
        'config': {
            'n_base': n_base,
            'n_envs': n_envs,
            'base_start_seed': base_start_seed,
            'texture_seeds': texture_seed_values,
            'scene_ids': scene_id_values,
            'scene_source_file': scene_source_file,
            'texture_eval_seed': texture_eval_seed,
            'visual_mode': visual_mode,
            'scene_id': scene_id,
            'policy_seed': policy_seed,
            'disable_material': disable_material,
            'disable_skybox': disable_skybox,
            'disable_light': disable_light,
            'skip_original_env': skip_original_env,
            'save_video': save_video,
            'difficulty_layout': difficulty_layout,
        },
        'original_env_summary': summarize_records(base_records),
        'texture_summary': summarize_records(modified_records),
        'original_env_records': base_records,
        'texture_records': modified_records,
        'modified_env_split': modified_split,
        'modified_env_summary': summarize_records(modified_records),
        'modified_env_records': modified_records,
        'base_summary': summarize_records(base_records),
        'base_records': base_records,
    }

    summary_path = out_path / 'summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, sort_keys=True)

    csv_path = out_path / 'episode_results.csv'
    with open(csv_path, 'w', newline='') as f:
        fieldnames = [
            'split',
            'rollout_index',
            'env_seed',
            'texture_seed',
            'scene_id',
            'source_split',
            'source_visual_mode',
            'source_scene_id',
            'source_rollout_index',
            'success_rate',
            'max_reward',
            'return',
            'horizon',
            'texture_map',
            'material_map',
            'skybox_config',
            'light_config',
            'video_path',
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for record in csv_records:
            writer.writerow(record)

    print('')
    if not skip_original_env:
        print('Original environment summary:')
        print(json.dumps(summary['original_env_summary'], indent=2, sort_keys=True))
        print('')
    print(f'{modified_split} summary:')
    print(json.dumps(summary['texture_summary'], indent=2, sort_keys=True))
    print('')
    print(f'Wrote summary to: {summary_path}')
    print(f'Wrote episode table to: {csv_path}')


if __name__ == '__main__':
    main()
