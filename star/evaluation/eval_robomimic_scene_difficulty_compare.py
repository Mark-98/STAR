#!/usr/bin/env python3
from __future__ import annotations

import csv
import copy
import json
import os
import pathlib
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import click
import numpy as np
import torch

try:
    import yaml
except ModuleNotFoundError:  # pragma: no cover - JSON configs still work.
    yaml = None


REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

import robomimic.utils.file_utils as FileUtils

from diffusion_policy.env.robomimic.rebalanced_difficulty_layout import (
    DEFAULT_LAYOUT_NAME,
    SUPPORTED_LAYOUT_NAMES,
    build_rebalanced_scene_records,
)
from diffusion_policy.model.common.rotation_transformer import RotationTransformer
from star.evaluation.eval_robomimic_texture_generalization import (
    csv_record_from_rollout,
    evaluate_rollout_specs,
    json_dumps_compact,
    load_policy_from_checkpoint,
    load_scene_source_records,
    maybe_disable_visual_components,
    parse_int_list,
    resolve_path,
    resolve_visual_config,
)


DEFAULT_DIFFICULTY_ORDER = ['clean', 'easy', 'medium', 'hard']
DEFAULT_VISUAL_MODES = {
    'clean': 'train',
    'train': 'train',
    'easy': 'eval-easy',
    'medium': 'eval-medium',
    'hard': 'eval-hard',
}
DEFAULT_SCENE_IDS = '0-29'
DEFAULT_SEEDS = '100000-100029'


@dataclass(frozen=True)
class ModelSpec:
    label: str
    checkpoint: str
    group: str
    train_seed: Optional[int] = None


def load_config(path: Optional[str]) -> Dict[str, Any]:
    if path is None:
        return {}
    config_path = pathlib.Path(resolve_path(path))
    with open(config_path, 'r', encoding='utf-8') as f:
        if config_path.suffix.lower() in {'.yaml', '.yml'}:
            if yaml is None:
                raise click.ClickException(
                    'PyYAML is not installed; use a JSON config or install PyYAML.'
                )
            payload = yaml.safe_load(f)
        else:
            payload = json.load(f)
    if payload is None:
        return {}
    if not isinstance(payload, dict):
        raise click.ClickException(f'Config must be a mapping: {config_path}')
    return payload


def write_json(path: pathlib.Path, payload: Any):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def sanitize_name(value: str) -> str:
    result = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in value)
    while '__' in result:
        result = result.replace('__', '_')
    return result.strip('_') or 'unnamed'


def expand_int_spec(value: Any, *, field_name: str) -> List[int]:
    if value is None:
        return []
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []
        return parse_int_list(value)
    if isinstance(value, (list, tuple)):
        result: List[int] = []
        for item in value:
            result.extend(expand_int_spec(item, field_name=field_name))
        return result
    raise click.ClickException(f'Unsupported {field_name}: {value!r}')


def normalize_model_entries(
    raw_models: Any,
    cli_models: Sequence[str],
) -> List[ModelSpec]:
    specs: List[ModelSpec] = []

    def _append(label: str, value: Any):
        if isinstance(value, str):
            checkpoint = value
            group = label
            train_seed = None
        elif isinstance(value, dict):
            checkpoint = value.get('checkpoint') or value.get('path')
            group = str(value.get('group', label))
            train_seed_value = value.get('train_seed', value.get('seed'))
            train_seed = None if train_seed_value is None else int(train_seed_value)
        else:
            raise click.ClickException(f'Invalid model entry for {label}: {value!r}')
        if not checkpoint:
            raise click.ClickException(f'Missing checkpoint for model {label}.')
        checkpoint = os.path.expandvars(str(checkpoint))
        checkpoint = resolve_path(checkpoint)
        if not pathlib.Path(checkpoint).is_file():
            raise click.ClickException(f'Checkpoint not found for {label}: {checkpoint}')
        specs.append(ModelSpec(
            label=str(label),
            checkpoint=checkpoint,
            group=group,
            train_seed=train_seed,
        ))

    if isinstance(raw_models, dict):
        for label, value in raw_models.items():
            _append(str(label), value)
    elif isinstance(raw_models, list):
        for idx, item in enumerate(raw_models):
            if not isinstance(item, dict):
                raise click.ClickException(f'Invalid model entry at index {idx}: {item!r}')
            label = item.get('label') or item.get('name')
            if not label:
                raise click.ClickException(f'Missing label for model entry at index {idx}.')
            _append(str(label), item)
    elif raw_models is not None:
        raise click.ClickException('`models` must be a mapping or a list.')

    for entry in cli_models:
        if '=' not in entry:
            raise click.ClickException(
                '--model must be label=/path/to/checkpoint.ckpt or group:label=/path.'
            )
        left, checkpoint = entry.split('=', 1)
        if ':' in left:
            group, label = left.split(':', 1)
        else:
            label = left
            group = left
        _append(label, {'checkpoint': checkpoint, 'group': group})

    if len(specs) == 0:
        raise click.ClickException('No models configured. Use config `models` or --model.')
    return specs


def default_difficulty_config() -> Dict[str, Any]:
    return {
        'order': list(DEFAULT_DIFFICULTY_ORDER),
        'clean': {
            'apply_visual_mod': False,
        },
        'easy': {
            'visual_mode': 'eval-easy',
            'scene_ids': DEFAULT_SCENE_IDS,
        },
        'medium': {
            'visual_mode': 'eval-medium',
            'scene_ids': DEFAULT_SCENE_IDS,
        },
        'hard': {
            'visual_mode': 'eval-hard',
            'scene_ids': DEFAULT_SCENE_IDS,
        },
    }


def merged_difficulty_config(config: Dict[str, Any]) -> Dict[str, Any]:
    raw = (
        config.get('robustness_eval')
        or config.get('difficulties')
        or config.get('difficulty')
        or {}
    )
    if raw is None:
        raw = {}
    if not isinstance(raw, dict):
        raise click.ClickException('Difficulty config must be a mapping.')
    merged = default_difficulty_config()
    for key, value in raw.items():
        if key == 'order':
            merged['order'] = list(value)
            continue
        if not isinstance(value, dict):
            raise click.ClickException(f'Difficulty `{key}` must be a mapping.')
        base = dict(merged.get(key, {}))
        base.update(value)
        merged[key] = base
    return merged


def get_eval_config(config: Dict[str, Any]) -> Dict[str, Any]:
    eval_cfg = dict(config.get('evaluation', {}) or {})
    visual_cfg = dict(config.get('visual_components', {}) or {})
    eval_cfg.setdefault('seeds', DEFAULT_SEEDS)
    eval_cfg.setdefault('seed_strategy', 'paired')
    eval_cfg.setdefault('n_envs', 1)
    eval_cfg.setdefault('vector_backend', 'sync')
    eval_cfg.setdefault('exact_scene_config', False)
    eval_cfg.setdefault('policy_seed', 0)
    eval_cfg.setdefault('save_video', False)
    eval_cfg.setdefault('difficulty_layout', DEFAULT_LAYOUT_NAME)
    eval_cfg.setdefault('monotonic_tolerance', 0.02)
    eval_cfg.setdefault('use_record_eval_seed', False)
    eval_cfg.setdefault('disable_material', bool(visual_cfg.get('disable_material', False)))
    eval_cfg.setdefault('disable_skybox', bool(visual_cfg.get('disable_skybox', False)))
    eval_cfg.setdefault('disable_light', bool(visual_cfg.get('disable_light', False)))
    return eval_cfg


def load_env_meta(dataset_path: str) -> Dict[str, Any]:
    env_meta = FileUtils.get_env_metadata_from_dataset(dataset_path)
    env_meta['env_kwargs']['use_object_obs'] = False
    return env_meta


def apply_abs_action_env_meta(
    env_meta: Dict[str, Any],
    abs_action: bool,
) -> Tuple[Dict[str, Any], Optional[RotationTransformer]]:
    result = copy.deepcopy(env_meta)
    rotation_transformer = None
    if abs_action:
        result['env_kwargs']['controller_configs']['control_delta'] = False
        rotation_transformer = RotationTransformer('axis_angle', 'rotation_6d')
    return result, rotation_transformer


def infer_apply_visual_mod(name: str, diff_cfg: Dict[str, Any]) -> bool:
    if 'apply_visual_mod' in diff_cfg:
        return bool(diff_cfg['apply_visual_mod'])
    if name == 'clean':
        has_scene = any(key in diff_cfg for key in ('scene_id', 'scene_ids', 'scene_source_file'))
        return bool(has_scene)
    return True


def copy_record_visual_config(
    record: Dict[str, Any],
    *,
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
) -> Dict[str, Any]:
    visual_config = {
        'texture_map': dict(record['texture_map']),
        'material_map': None if record.get('material_map') is None else dict(record['material_map']),
        'skybox_config': None if record.get('skybox_config') is None else dict(record['skybox_config']),
        'light_config': None if record.get('light_config') is None else dict(record['light_config']),
    }
    if disable_material or disable_skybox or disable_light:
        visual_config = maybe_disable_visual_components(
            visual_config,
            disable_material=disable_material,
            disable_skybox=disable_skybox,
            disable_light=disable_light,
        )
    return visual_config


def resolve_scene_record(
    *,
    env_meta: Dict[str, Any],
    difficulty_name: str,
    diff_cfg: Dict[str, Any],
    visual_mode: str,
    scene_id: int,
    scene_source_records: Optional[List[Dict[str, Any]]],
    rebalanced_records: Optional[List[Dict[str, Any]]],
    difficulty_layout: str,
    disable_material: bool,
    disable_skybox: bool,
    disable_light: bool,
) -> Dict[str, Any]:
    source_split = None
    source_visual_mode = None
    source_scene_id = None
    source_rollout_index = None
    record_eval_seed = None

    if scene_source_records is not None:
        if scene_id < 0 or scene_id >= len(scene_source_records):
            raise click.ClickException(
                f'{difficulty_name}: scene_id={scene_id} is out of range for '
                f'scene_source_file with {len(scene_source_records)} records.'
            )
        record = scene_source_records[scene_id]
        visual_config = copy_record_visual_config(
            record,
            disable_material=disable_material,
            disable_skybox=disable_skybox,
            disable_light=disable_light,
        )
        source_split = record.get('source_split')
        source_visual_mode = record.get('source_visual_mode')
        source_scene_id = record.get('source_scene_id')
        source_rollout_index = record.get('source_rollout_index')
        record_eval_seed = record.get('eval_seed')
    elif rebalanced_records is not None:
        if scene_id < 0 or scene_id >= len(rebalanced_records):
            raise click.ClickException(
                f'{difficulty_name}: scene_id={scene_id} is out of range for '
                f'{env_meta["env_name"]} {visual_mode} {difficulty_layout}. '
                f'Available range is 0..{len(rebalanced_records) - 1}.'
            )
        record = rebalanced_records[scene_id]
        visual_config = copy_record_visual_config(
            record,
            disable_material=disable_material,
            disable_skybox=disable_skybox,
            disable_light=disable_light,
        )
        source_split = record.get('source_split')
        source_visual_mode = record.get('source_visual_mode')
        source_scene_id = record.get('source_scene_id')
        source_rollout_index = record.get('source_rollout_index')
        record_eval_seed = record.get('eval_seed')
    else:
        texture_seed = diff_cfg.get('texture_seed')
        texture_seed = None if texture_seed is None else int(texture_seed)
        visual_config = resolve_visual_config(
            env_meta=env_meta,
            apply_visual_mod=True,
            texture_seed=texture_seed,
            visual_mode=visual_mode,
            scene_id=scene_id,
            disable_material=disable_material,
            disable_skybox=disable_skybox,
            disable_light=disable_light,
        )

    return {
        'visual_config': visual_config,
        'source_split': source_split,
        'source_visual_mode': source_visual_mode,
        'source_scene_id': source_scene_id,
        'source_rollout_index': source_rollout_index,
        'record_eval_seed': record_eval_seed,
    }


def choose_scene_ids(
    diff_cfg: Dict[str, Any],
    *,
    cli_scene_ids: Optional[str],
    cli_scene_id: Sequence[int],
    source_count: Optional[int],
) -> List[int]:
    if len(cli_scene_id) > 0:
        return [int(value) for value in cli_scene_id]
    if cli_scene_ids:
        return expand_int_spec(cli_scene_ids, field_name='--scene-ids')
    if 'scene_id' in diff_cfg:
        return expand_int_spec(diff_cfg['scene_id'], field_name='scene_id')
    if 'scene_ids' in diff_cfg:
        return expand_int_spec(diff_cfg['scene_ids'], field_name='scene_ids')
    if source_count is not None:
        return list(range(source_count))
    return expand_int_spec(DEFAULT_SCENE_IDS, field_name='default scene_ids')


def seeds_for_scene(
    *,
    seed_values: List[int],
    seed_strategy: str,
    scene_index: int,
    scene_count: int,
    record_eval_seed: Optional[int],
    use_record_eval_seed: bool,
) -> List[int]:
    if use_record_eval_seed and record_eval_seed is not None:
        return [int(record_eval_seed)]
    if seed_strategy == 'cross_product':
        return list(seed_values)
    if seed_strategy == 'paired':
        if len(seed_values) == 0:
            raise click.ClickException('evaluation.seeds cannot be empty.')
        if scene_count == len(seed_values):
            return [int(seed_values[scene_index])]
        if len(seed_values) == 1:
            return [int(seed_values[0])]
        return [int(seed_values[scene_index % len(seed_values)])]
    raise click.ClickException(
        f'Unsupported seed_strategy={seed_strategy!r}. Use paired or cross_product.'
    )


def build_difficulty_specs(
    *,
    env_meta: Dict[str, Any],
    difficulty_name: str,
    diff_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    seed_values: List[int],
    cli_scene_ids: Optional[str],
    cli_scene_id: Sequence[int],
    save_video: bool,
    videos_dir: pathlib.Path,
    start_policy_offset: int,
) -> List[Dict[str, Any]]:
    apply_visual_mod = infer_apply_visual_mod(difficulty_name, diff_cfg)
    visual_mode = str(
        diff_cfg.get('visual_mode')
        or DEFAULT_VISUAL_MODES.get(difficulty_name, difficulty_name)
    )
    seed_strategy = str(eval_cfg.get('seed_strategy', 'paired'))
    policy_seed_base = int(eval_cfg.get('policy_seed', 0))

    specs: List[Dict[str, Any]] = []
    if not apply_visual_mod:
        for episode_idx, env_seed in enumerate(seed_values):
            video_path = None
            if save_video:
                video_path = videos_dir / difficulty_name / f'episode_{episode_idx:03d}_seed_{env_seed}.mp4'
            specs.append({
                'split': difficulty_name,
                'difficulty': difficulty_name,
                'rollout_index': len(specs),
                'difficulty_episode': episode_idx,
                'scene_episode': episode_idx,
                'env_seed': int(env_seed),
                'texture_seed': None,
                'scene_id': None,
                'apply_visual_mod': False,
                'visual_mode': visual_mode,
                'visual_config': None,
                'policy_seed': int(policy_seed_base + start_policy_offset + len(specs)),
                'source_split': None,
                'source_visual_mode': None,
                'source_scene_id': None,
                'source_rollout_index': None,
                'video_path': None if video_path is None else str(video_path),
            })
        return specs

    scene_source_records = None
    if diff_cfg.get('scene_source_file'):
        scene_source_records = load_scene_source_records(str(diff_cfg['scene_source_file']))

    difficulty_layout = str(eval_cfg.get('difficulty_layout', DEFAULT_LAYOUT_NAME))
    if difficulty_layout not in SUPPORTED_LAYOUT_NAMES:
        raise click.ClickException(
            f'Unsupported difficulty_layout={difficulty_layout!r}. '
            f'Supported: {sorted(SUPPORTED_LAYOUT_NAMES)}'
        )

    rebalanced_records = None
    if scene_source_records is None and difficulty_layout != DEFAULT_LAYOUT_NAME:
        rebalanced_records = build_rebalanced_scene_records(
            task_name=env_meta['env_name'],
            visual_mode=visual_mode,
            layout_name=difficulty_layout,
        )

    source_count = None
    if scene_source_records is not None:
        source_count = len(scene_source_records)
    elif rebalanced_records is not None:
        source_count = len(rebalanced_records)

    scene_ids = choose_scene_ids(
        diff_cfg,
        cli_scene_ids=cli_scene_ids,
        cli_scene_id=cli_scene_id,
        source_count=source_count,
    )
    if len(scene_ids) == 0:
        raise click.ClickException(f'{difficulty_name}: no scene ids configured.')

    for scene_index, scene_id in enumerate(scene_ids):
        scene_payload = resolve_scene_record(
            env_meta=env_meta,
            difficulty_name=difficulty_name,
            diff_cfg=diff_cfg,
            visual_mode=visual_mode,
            scene_id=int(scene_id),
            scene_source_records=scene_source_records,
            rebalanced_records=rebalanced_records,
            difficulty_layout=difficulty_layout,
            disable_material=bool(eval_cfg.get('disable_material', False)),
            disable_skybox=bool(eval_cfg.get('disable_skybox', False)),
            disable_light=bool(eval_cfg.get('disable_light', False)),
        )
        env_seeds = seeds_for_scene(
            seed_values=seed_values,
            seed_strategy=seed_strategy,
            scene_index=scene_index,
            scene_count=len(scene_ids),
            record_eval_seed=scene_payload.get('record_eval_seed'),
            use_record_eval_seed=bool(eval_cfg.get('use_record_eval_seed', False)),
        )
        for scene_episode, env_seed in enumerate(env_seeds):
            episode_idx = len(specs)
            video_path = None
            if save_video:
                video_path = (
                    videos_dir
                    / difficulty_name
                    / f'scene_{int(scene_id):03d}'
                    / f'episode_{scene_episode:03d}_seed_{int(env_seed)}.mp4'
                )
            specs.append({
                'split': difficulty_name,
                'difficulty': difficulty_name,
                'rollout_index': episode_idx,
                'difficulty_episode': episode_idx,
                'scene_episode': scene_episode,
                'env_seed': int(env_seed),
                'texture_seed': diff_cfg.get('texture_seed'),
                'scene_id': int(scene_id),
                'apply_visual_mod': True,
                'visual_mode': visual_mode,
                'visual_config': scene_payload['visual_config'],
                'exact_visual_config': bool(eval_cfg.get('exact_scene_config', False)),
                'policy_seed': int(policy_seed_base + start_policy_offset + len(specs)),
                'source_split': scene_payload.get('source_split'),
                'source_visual_mode': scene_payload.get('source_visual_mode'),
                'source_scene_id': scene_payload.get('source_scene_id'),
                'source_rollout_index': scene_payload.get('source_rollout_index'),
                'video_path': None if video_path is None else str(video_path),
            })
    return specs


def build_all_specs(
    *,
    env_meta: Dict[str, Any],
    difficulty_cfg: Dict[str, Any],
    eval_cfg: Dict[str, Any],
    difficulty_filter: Sequence[str],
    seed_values: List[int],
    cli_scene_ids: Optional[str],
    cli_scene_id: Sequence[int],
    save_video: bool,
    videos_dir: pathlib.Path,
) -> List[Dict[str, Any]]:
    configured_order = list(difficulty_cfg.get('order', DEFAULT_DIFFICULTY_ORDER))
    if len(difficulty_filter) > 0:
        wanted = set(difficulty_filter)
        difficulty_order = [name for name in configured_order if name in wanted]
        missing = sorted(wanted.difference(difficulty_order))
        if missing:
            raise click.ClickException(f'Unknown difficulty filter(s): {missing}')
    else:
        difficulty_order = configured_order

    specs: List[Dict[str, Any]] = []
    for difficulty_name in difficulty_order:
        if difficulty_name not in difficulty_cfg:
            raise click.ClickException(f'Missing difficulty config: {difficulty_name}')
        current_specs = build_difficulty_specs(
            env_meta=env_meta,
            difficulty_name=str(difficulty_name),
            diff_cfg=dict(difficulty_cfg[difficulty_name]),
            eval_cfg=eval_cfg,
            seed_values=seed_values,
            cli_scene_ids=cli_scene_ids,
            cli_scene_id=cli_scene_id,
            save_video=save_video,
            videos_dir=videos_dir,
            start_policy_offset=len(specs),
        )
        specs.extend(current_specs)
    for spec in specs:
        if eval_cfg.get('fixed_env_seed') is not None:
            spec['env_seed'] = int(eval_cfg['fixed_env_seed'])
        spec['record_initial_state'] = bool(eval_cfg.get('record_initial_state', False))
    return specs


def enrich_records(
    *,
    model: ModelSpec,
    specs: List[Dict[str, Any]],
    records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    enriched: List[Dict[str, Any]] = []
    for spec, record in zip(specs, records):
        item = {
            **record,
            'model': model.label,
            'model_group': model.group,
            'train_seed': model.train_seed,
            'checkpoint': model.checkpoint,
            'difficulty': spec['difficulty'],
            'difficulty_episode': int(spec['difficulty_episode']),
            'scene_episode': int(spec['scene_episode']),
            'visual_mode': spec.get('visual_mode'),
            'success': int(float(record.get('success_rate', 0.0)) >= 1.0),
        }
        enriched.append(item)
    return enriched


def mean(values: Sequence[float]) -> float:
    return float(statistics.mean(values)) if values else 0.0


def pstdev(values: Sequence[float]) -> float:
    return float(statistics.pstdev(values)) if len(values) > 1 else 0.0


def group_by(records: Iterable[Dict[str, Any]], keys: Sequence[str]) -> Dict[Tuple[Any, ...], List[Dict[str, Any]]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = {}
    for record in records:
        key = tuple(record.get(name) for name in keys)
        grouped.setdefault(key, []).append(record)
    return grouped


def build_scene_summary(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    keys = ['model', 'model_group', 'train_seed', 'difficulty', 'scene_id']
    for key, group_records in sorted(group_by(records, keys).items(), key=lambda item: str(item[0])):
        successes = [float(record['success']) for record in group_records]
        rows.append({
            'model': key[0],
            'model_group': key[1],
            'train_seed': '' if key[2] is None else key[2],
            'difficulty': key[3],
            'scene_id': 'original_env' if key[4] is None else key[4],
            'num_episodes': len(group_records),
            'success_rate': mean(successes),
            'std_success_rate': pstdev(successes),
            'mean_return': mean([float(record['return']) for record in group_records]),
            'mean_max_reward': mean([float(record['max_reward']) for record in group_records]),
            'mean_horizon': mean([float(record['horizon']) for record in group_records]),
        })
    return rows


def build_difficulty_summary(
    records: List[Dict[str, Any]],
    scene_rows: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    scene_rates: Dict[Tuple[Any, ...], List[float]] = {}
    for row in scene_rows:
        key = (
            row['model'],
            row['model_group'],
            row['train_seed'],
            row['difficulty'],
        )
        scene_rates.setdefault(key, []).append(float(row['success_rate']))

    rows: List[Dict[str, Any]] = []
    keys = ['model', 'model_group', 'train_seed', 'difficulty']
    for key, group_records in sorted(group_by(records, keys).items(), key=lambda item: str(item[0])):
        successes = [float(record['success']) for record in group_records]
        scene_key = (
            key[0],
            key[1],
            '' if key[2] is None else key[2],
            key[3],
        )
        rates = scene_rates.get(scene_key, [])
        rows.append({
            'model': key[0],
            'model_group': key[1],
            'train_seed': '' if key[2] is None else key[2],
            'difficulty': key[3],
            'num_scenes': len(rates),
            'num_episodes': len(group_records),
            'success_rate': mean(successes),
            'std_success_rate': pstdev(rates if len(rates) > 1 else successes),
            'mean_return': mean([float(record['return']) for record in group_records]),
            'mean_max_reward': mean([float(record['max_reward']) for record in group_records]),
            'mean_horizon': mean([float(record['horizon']) for record in group_records]),
        })
    return rows


def build_group_difficulty_summary(difficulty_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    grouped: Dict[Tuple[Any, Any], List[Dict[str, Any]]] = {}
    for row in difficulty_rows:
        grouped.setdefault((row['model_group'], row['difficulty']), []).append(row)
    for key, group_rows in sorted(grouped.items(), key=lambda item: str(item[0])):
        rates = [float(row['success_rate']) for row in group_rows]
        rows.append({
            'model_group': key[0],
            'difficulty': key[1],
            'num_models': len(group_rows),
            'num_episodes': sum(int(row['num_episodes']) for row in group_rows),
            'mean_success_rate': mean(rates),
            'std_success_rate': pstdev(rates),
        })
    return rows


def row_lookup(
    rows: List[Dict[str, Any]],
    *,
    key_fields: Sequence[str],
    value_field: str,
) -> Dict[Tuple[Any, ...], float]:
    return {
        tuple(row[field] for field in key_fields): float(row[value_field])
        for row in rows
    }


def build_comparison_summary(
    group_rows: List[Dict[str, Any]],
    *,
    difficulty_order: Sequence[str],
    base_group: Optional[str],
    target_groups: Sequence[str],
) -> List[Dict[str, Any]]:
    groups = sorted({str(row['model_group']) for row in group_rows})
    if not base_group:
        base_candidates = [group for group in groups if 'base' in group.lower()]
        base_group = base_candidates[0] if base_candidates else (groups[0] if groups else None)
    if base_group is None:
        return []
    targets = list(target_groups) if target_groups else [group for group in groups if group != base_group]
    rates = row_lookup(group_rows, key_fields=['model_group', 'difficulty'], value_field='mean_success_rate')
    base_clean = rates.get((base_group, 'clean'))
    rows: List[Dict[str, Any]] = []
    for target in targets:
        target_clean = rates.get((target, 'clean'))
        for difficulty in difficulty_order:
            base_rate = rates.get((base_group, difficulty))
            target_rate = rates.get((target, difficulty))
            if base_rate is None or target_rate is None:
                continue
            base_degradation = None if base_clean is None else base_clean - base_rate
            target_degradation = None if target_clean is None else target_clean - target_rate
            rows.append({
                'base_group': base_group,
                'model_group': target,
                'difficulty': difficulty,
                'base_success_rate': base_rate,
                'model_success_rate': target_rate,
                'robustness_gain': target_rate - base_rate,
                'base_clean_degradation': base_degradation,
                'model_clean_degradation': target_degradation,
                'degradation_reduction': (
                    None
                    if base_degradation is None or target_degradation is None
                    else base_degradation - target_degradation
                ),
            })
    return rows


def check_monotonicity(
    group_rows: List[Dict[str, Any]],
    *,
    difficulty_order: Sequence[str],
    tolerance: float,
) -> List[Dict[str, Any]]:
    rates_by_group: Dict[str, Dict[str, float]] = {}
    for row in group_rows:
        rates_by_group.setdefault(str(row['model_group']), {})[str(row['difficulty'])] = float(row['mean_success_rate'])

    checks: List[Dict[str, Any]] = []
    for group, rates in sorted(rates_by_group.items()):
        ordered = [(name, rates[name]) for name in difficulty_order if name in rates]
        warnings: List[str] = []
        for (prev_name, prev_rate), (next_name, next_rate) in zip(ordered, ordered[1:]):
            if next_rate > prev_rate + tolerance:
                warnings.append(
                    f'{next_name} success rate ({next_rate:.4f}) is higher than '
                    f'{prev_name} ({prev_rate:.4f}).'
                )
        checks.append({
            'model_group': group,
            'passed': len(warnings) == 0,
            'rates': {name: rate for name, rate in ordered},
            'warnings': warnings,
        })
    return checks


def write_csv(path: pathlib.Path, rows: List[Dict[str, Any]], fieldnames: Sequence[str]):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(fieldnames))
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, '') for field in fieldnames})


def episode_csv_row(record: Dict[str, Any]) -> Dict[str, Any]:
    row = csv_record_from_rollout(record)
    return {
        'model': record['model'],
        'model_group': record['model_group'],
        'train_seed': '' if record.get('train_seed') is None else record.get('train_seed'),
        'difficulty': record['difficulty'],
        'scene_id': 'original_env' if record.get('scene_id') is None else record.get('scene_id'),
        'seed': record['env_seed'],
        'initial_state_hash': record.get('initial_state_hash', ''),
        'policy_batch_seed': record.get('policy_batch_seed', ''),
        'policy_batch_lane': record.get('policy_batch_lane', ''),
        'episode': record['difficulty_episode'],
        'scene_episode': record['scene_episode'],
        'success': record['success'],
        'success_rate': record['success_rate'],
        'max_reward': record['max_reward'],
        'return': record['return'],
        'horizon': record['horizon'],
        'visual_mode': record.get('visual_mode', ''),
        'source_split': row.get('source_split', ''),
        'source_visual_mode': row.get('source_visual_mode', ''),
        'source_scene_id': row.get('source_scene_id', ''),
        'source_rollout_index': row.get('source_rollout_index', ''),
        'texture_map': row.get('texture_map', ''),
        'material_map': row.get('material_map', ''),
        'skybox_config': row.get('skybox_config', ''),
        'light_config': row.get('light_config', ''),
        'video_path': row.get('video_path', ''),
        'checkpoint': record['checkpoint'],
    }


def json_ready_records(records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result: List[Dict[str, Any]] = []
    for record in records:
        row = dict(record)
        for key in ('texture_map', 'material_map', 'skybox_config', 'light_config'):
            if key in row and row[key] is not None:
                row[key] = json.loads(json_dumps_compact(row[key]))
        result.append(row)
    return result


def write_plot(
    path: pathlib.Path,
    group_rows: List[Dict[str, Any]],
    *,
    difficulty_order: Sequence[str],
) -> Optional[str]:
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        return None

    groups = sorted({str(row['model_group']) for row in group_rows})
    if len(groups) == 0:
        return None

    rate_map = {
        (str(row['model_group']), str(row['difficulty'])): float(row['mean_success_rate'])
        for row in group_rows
    }
    std_map = {
        (str(row['model_group']), str(row['difficulty'])): float(row['std_success_rate'])
        for row in group_rows
    }
    x_values = list(range(len(difficulty_order)))
    plt.figure(figsize=(8, 5))
    for group in groups:
        y_values = [rate_map.get((group, difficulty), float('nan')) * 100.0 for difficulty in difficulty_order]
        y_errors = [std_map.get((group, difficulty), 0.0) * 100.0 for difficulty in difficulty_order]
        plt.errorbar(x_values, y_values, yerr=y_errors, marker='o', capsize=3, label=group)
    plt.xticks(x_values, [name.capitalize() for name in difficulty_order])
    plt.ylabel('Success Rate (%)')
    plt.xlabel('Visual Difficulty')
    plt.ylim(0, 105)
    plt.grid(True, axis='y', alpha=0.3)
    plt.legend()
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=160)
    plt.close()
    return str(path)


def print_monotonicity(checks: List[Dict[str, Any]]):
    for check in checks:
        rates_text = ' > '.join(
            f'{name.capitalize()} {rate * 100.0:.1f}'
            for name, rate in check['rates'].items()
        )
        if check['passed']:
            print(f'[PASS] {check["model_group"]} difficulty trend: {rates_text}')
        else:
            print(f'[WARNING] {check["model_group"]} difficulty trend: {rates_text}')
            for warning in check['warnings']:
                print(f'  {warning}')


@click.command()
@click.option('--config', 'config_path', type=click.Path(exists=True, dir_okay=False), default=None)
@click.option('--model', 'cli_models', multiple=True, help='label=/path.ckpt or group:label=/path.ckpt')
@click.option('--model-seed', 'model_seed_filter', multiple=True, type=int, help='Select configured model train_seed(s).')
@click.option('--output-dir', required=True, type=click.Path())
@click.option('--task', 'task_name_override', default=None, help='Optional task label for metadata only.')
@click.option('--dataset-path', default=None, type=click.Path(dir_okay=False))
@click.option('--device', default=None, help='Overrides config.runtime.device. Default: cuda:0')
@click.option('--seeds', default=None, help='Comma/range list, e.g. 100000-100029.')
@click.option('--difficulty', 'difficulty_filter', multiple=True, help='Limit to one or more configured difficulties.')
@click.option('--scene-id', 'cli_scene_id', multiple=True, type=int, help='Direct scene id override.')
@click.option('--scene-ids', 'cli_scene_ids', default=None, help='Direct scene id range override, e.g. 0-29.')
@click.option('--n-envs', default=None, type=int)
@click.option('--vector-backend', type=click.Choice(['sync', 'async']), default=None)
@click.option('--difficulty-layout', default=None)
@click.option('--save-video/--no-save-video', default=None)
@click.option('--overwrite', is_flag=True)
@click.option('--monotonic-tolerance', default=None, type=float)
def main(
    config_path: Optional[str],
    cli_models: Sequence[str],
    model_seed_filter: Sequence[int],
    output_dir: str,
    task_name_override: Optional[str],
    dataset_path: Optional[str],
    device: Optional[str],
    seeds: Optional[str],
    difficulty_filter: Sequence[str],
    cli_scene_id: Sequence[int],
    cli_scene_ids: Optional[str],
    n_envs: Optional[int],
    vector_backend: Optional[str],
    difficulty_layout: Optional[str],
    save_video: Optional[bool],
    overwrite: bool,
    monotonic_tolerance: Optional[float],
):
    config = load_config(config_path)
    models = normalize_model_entries(config.get('models'), cli_models)
    if model_seed_filter:
        models = [model for model in models if model.train_seed in model_seed_filter]
        if {model.train_seed for model in models} != set(model_seed_filter):
            raise click.ClickException('One or more --model-seed values have no configured models.')
    difficulty_cfg = merged_difficulty_config(config)
    eval_cfg = get_eval_config(config)

    runtime_cfg = dict(config.get('runtime', {}) or {})
    task_cfg = dict(config.get('task', {}) or {})
    comparison_cfg = dict(config.get('comparison', {}) or {})

    if device is not None:
        runtime_cfg['device'] = device
    runtime_cfg.setdefault('device', 'cuda:0')

    if dataset_path is not None:
        task_cfg['dataset_path'] = dataset_path
    if task_name_override is not None:
        task_cfg['name'] = task_name_override
    if seeds is not None:
        eval_cfg['seeds'] = seeds
    if n_envs is not None:
        eval_cfg['n_envs'] = int(n_envs)
    if vector_backend is not None:
        eval_cfg['vector_backend'] = vector_backend
    if difficulty_layout is not None:
        eval_cfg['difficulty_layout'] = difficulty_layout
    if save_video is not None:
        eval_cfg['save_video'] = bool(save_video)
    if monotonic_tolerance is not None:
        eval_cfg['monotonic_tolerance'] = float(monotonic_tolerance)
    if eval_cfg['vector_backend'] not in ('sync', 'async'):
        raise click.ClickException('evaluation.vector_backend must be sync or async.')

    output_dir = resolve_path(output_dir)
    out_path = pathlib.Path(output_dir)
    if out_path.exists() and not overwrite:
        raise click.ClickException(f'Output path already exists: {out_path}. Pass --overwrite.')
    out_path.mkdir(parents=True, exist_ok=True)

    seed_values = expand_int_spec(eval_cfg.get('seeds', DEFAULT_SEEDS), field_name='evaluation.seeds')
    if len(seed_values) == 0:
        raise click.ClickException('evaluation.seeds cannot be empty.')
    if int(eval_cfg.get('n_envs', 1)) < 1:
        raise click.ClickException('evaluation.n_envs must be >= 1.')

    device_obj = torch.device(str(runtime_cfg['device']))
    all_records: List[Dict[str, Any]] = []
    reference_environment_protocol = None
    dataset_path_resolved: Optional[str] = None
    base_env_meta: Optional[Dict[str, Any]] = None
    task_name = task_cfg.get('name')

    run_config = {
        'created_at': datetime.now().isoformat(),
        'config_path': None if config_path is None else resolve_path(config_path),
        'output_dir': output_dir,
        'task': task_cfg,
        'runtime': runtime_cfg,
        'evaluation': eval_cfg,
        'difficulty': difficulty_cfg,
        'difficulty_filter': list(difficulty_filter),
        'cli_scene_ids': cli_scene_ids,
        'cli_scene_id': list(cli_scene_id),
        'models': [
            {
                'label': model.label,
                'group': model.group,
                'train_seed': model.train_seed,
                'checkpoint': model.checkpoint,
            }
            for model in models
        ],
    }
    write_json(out_path / 'run_config.json', run_config)

    print(f'Output: {out_path}')
    print(f'Models: {len(models)}')
    print(f'Device: {device_obj}')
    print(f'Seeds: {seed_values}')
    print(f'Seed strategy: {eval_cfg["seed_strategy"]}')
    print(f'Environment backend: {eval_cfg["vector_backend"]}, n_envs={eval_cfg["n_envs"]}')
    print(f'Difficulty layout: {eval_cfg["difficulty_layout"]}')

    for model in models:
        print('')
        print(f'Loading model: {model.label}')
        print(f'  group:      {model.group}')
        print(f'  checkpoint: {model.checkpoint}')
        model_output_dir = out_path / 'workspace_load' / sanitize_name(model.label)
        cfg, policy = load_policy_from_checkpoint(
            checkpoint=model.checkpoint,
            output_dir=str(model_output_dir),
            device=device_obj,
        )

        if dataset_path_resolved is None:
            configured_dataset = task_cfg.get('dataset_path') or cfg.task.dataset.dataset_path
            dataset_path_resolved = resolve_path(str(configured_dataset))
            if not pathlib.Path(dataset_path_resolved).is_file():
                raise click.ClickException(f'Dataset not found: {dataset_path_resolved}')
            base_env_meta = load_env_meta(dataset_path_resolved)
            task_name = task_name or base_env_meta['env_name']
            print(f'Dataset: {dataset_path_resolved}')
            print(f'Task: {task_name}')

        assert base_env_meta is not None
        abs_action = bool(cfg.task.env_runner.abs_action)
        env_meta, rotation_transformer = apply_abs_action_env_meta(
            base_env_meta,
            abs_action=abs_action,
        )
        environment_protocol = {
            'env_name': env_meta['env_name'],
            'max_steps': int(cfg.task.env_runner.max_steps),
            'n_action_steps': int(cfg.task.env_runner.n_action_steps),
            'abs_action': abs_action,
        }
        if eval_cfg.get('require_identical_initial_state', False):
            if reference_environment_protocol is None:
                reference_environment_protocol = environment_protocol
            elif reference_environment_protocol != environment_protocol:
                raise click.ClickException(f'Environment protocol differs for {model.label}.')
            print(f'  fixed-state environment protocol: {environment_protocol}')
        specs = build_all_specs(
            env_meta=env_meta,
            difficulty_cfg=difficulty_cfg,
            eval_cfg=eval_cfg,
            difficulty_filter=difficulty_filter,
            seed_values=seed_values,
            cli_scene_ids=cli_scene_ids,
            cli_scene_id=cli_scene_id,
            save_video=bool(eval_cfg.get('save_video', False)),
            videos_dir=out_path / 'videos' / sanitize_name(model.label),
        )
        print(f'  rollout specs: {len(specs)}')
        records = evaluate_rollout_specs(
            policy=policy,
            cfg=cfg,
            env_meta=env_meta,
            specs=specs,
            n_envs=int(eval_cfg.get('n_envs', 1)),
            abs_action=abs_action,
            rotation_transformer=rotation_transformer,
            vector_backend=str(eval_cfg['vector_backend']),
        )
        model_records = enrich_records(model=model, specs=specs, records=records)
        if eval_cfg.get('record_initial_state', False):
            for record in model_records:
                record['environment_protocol'] = environment_protocol
            model_records_dir = out_path / 'model_records'
            model_records_dir.mkdir(exist_ok=True)
            write_json(model_records_dir / f'{sanitize_name(model.label)}.json',
                       json_ready_records(model_records))
        all_records.extend(model_records)
        if eval_cfg.get('require_identical_initial_state', False):
            validate_fixed_initial_conditions(all_records, int(eval_cfg['fixed_env_seed']))
        del policy
        if device_obj.type == 'cuda':
            torch.cuda.empty_cache()

    write_evaluation_report(
        out_path=out_path, all_records=all_records, difficulty_cfg=difficulty_cfg,
        difficulty_filter=difficulty_filter, eval_cfg=eval_cfg,
        comparison_cfg=comparison_cfg, task_name=task_name,
        dataset_path_resolved=dataset_path_resolved,
    )


def validate_fixed_initial_conditions(records, seed):
    if not records:
        raise ValueError('No episodes to validate.')
    reference = np.asarray(records[0].get('initial_state', []), dtype=float)
    if reference.size == 0:
        raise ValueError('Missing recorded simulator initial state.')
    max_delta = 0.0
    reference_protocol = records[0].get('environment_protocol')
    episode_protocols = {}
    for row in records:
        state = np.asarray(row.get('initial_state', []), dtype=float)
        if row['env_seed'] != seed or state.shape != reference.shape:
            raise ValueError(f'Initial-condition protocol mismatch: {row["model"]}')
        if not np.allclose(state, reference, rtol=0, atol=1e-10):
            raise ValueError(f'Initial states differ: {row["model"]}, scene {row.get("scene_id")}')
        if row.get('environment_protocol') != reference_protocol:
            raise ValueError(f'Environment protocols differ: {row["model"]}')
        if 'difficulty_episode' in row:
            key = (row['difficulty'], row['difficulty_episode'])
            protocol = {name: row.get(name) for name in (
                'scene_id', 'policy_batch_seed', 'policy_batch_lane',
                'texture_map', 'material_map', 'skybox_config', 'light_config')}
            if episode_protocols.setdefault(key, protocol) != protocol:
                raise ValueError(f'Paired evaluation protocol differs: {key}, {row["model"]}')
        max_delta = max(max_delta, float(np.max(np.abs(state - reference))))
    return {'passed': True, 'env_seed': seed, 'num_episodes': len(records),
            'max_absolute_state_difference': max_delta,
            'environment_protocol': reference_protocol,
            'note': 'One physical initial state; repeated trials are not independent initial states.'}


def write_evaluation_report(
    *, out_path, all_records, difficulty_cfg, difficulty_filter, eval_cfg,
    comparison_cfg, task_name, dataset_path_resolved,
):
    initial_condition_check = None
    if eval_cfg.get('require_identical_initial_state', False):
        initial_condition_check = validate_fixed_initial_conditions(
            all_records, int(eval_cfg['fixed_env_seed']))
    difficulty_order = [
        str(name)
        for name in difficulty_cfg.get('order', DEFAULT_DIFFICULTY_ORDER)
        if len(difficulty_filter) == 0 or str(name) in set(difficulty_filter)
    ]
    scene_rows = build_scene_summary(all_records)
    difficulty_rows = build_difficulty_summary(all_records, scene_rows)
    group_rows = build_group_difficulty_summary(difficulty_rows)
    comparison_rows = build_comparison_summary(
        group_rows,
        difficulty_order=difficulty_order,
        base_group=comparison_cfg.get('base_group'),
        target_groups=list(comparison_cfg.get('target_groups', []) or []),
    )
    monotonicity = check_monotonicity(
        group_rows,
        difficulty_order=difficulty_order,
        tolerance=float(eval_cfg.get('monotonic_tolerance', 0.02)),
    )

    episode_rows = [episode_csv_row(record) for record in all_records]
    write_csv(
        out_path / 'episode_results.csv',
        episode_rows,
        [
            'model',
            'model_group',
            'train_seed',
            'difficulty',
            'scene_id',
            'seed',
            'initial_state_hash',
            'policy_batch_seed',
            'policy_batch_lane',
            'episode',
            'scene_episode',
            'success',
            'success_rate',
            'max_reward',
            'return',
            'horizon',
            'visual_mode',
            'source_split',
            'source_visual_mode',
            'source_scene_id',
            'source_rollout_index',
            'texture_map',
            'material_map',
            'skybox_config',
            'light_config',
            'video_path',
            'checkpoint',
        ],
    )
    write_csv(
        out_path / 'scene_summary.csv',
        scene_rows,
        [
            'model',
            'model_group',
            'train_seed',
            'difficulty',
            'scene_id',
            'num_episodes',
            'success_rate',
            'std_success_rate',
            'mean_return',
            'mean_max_reward',
            'mean_horizon',
        ],
    )
    write_csv(
        out_path / 'difficulty_summary.csv',
        difficulty_rows,
        [
            'model',
            'model_group',
            'train_seed',
            'difficulty',
            'num_scenes',
            'num_episodes',
            'success_rate',
            'std_success_rate',
            'mean_return',
            'mean_max_reward',
            'mean_horizon',
        ],
    )
    write_csv(
        out_path / 'group_difficulty_summary.csv',
        group_rows,
        [
            'model_group',
            'difficulty',
            'num_models',
            'num_episodes',
            'mean_success_rate',
            'std_success_rate',
        ],
    )
    write_csv(
        out_path / 'comparison_summary.csv',
        comparison_rows,
        [
            'base_group',
            'model_group',
            'difficulty',
            'base_success_rate',
            'model_success_rate',
            'robustness_gain',
            'base_clean_degradation',
            'model_clean_degradation',
            'degradation_reduction',
        ],
    )

    plot_path = write_plot(
        out_path / 'success_rate_by_difficulty.png',
        group_rows,
        difficulty_order=difficulty_order,
    )
    summary = {
        'task_name': task_name,
        'dataset_path': dataset_path_resolved,
        'episode_results_csv': str(out_path / 'episode_results.csv'),
        'scene_summary_csv': str(out_path / 'scene_summary.csv'),
        'difficulty_summary_csv': str(out_path / 'difficulty_summary.csv'),
        'group_difficulty_summary_csv': str(out_path / 'group_difficulty_summary.csv'),
        'comparison_summary_csv': str(out_path / 'comparison_summary.csv'),
        'plot': plot_path,
        'scene_summary': scene_rows,
        'difficulty_summary': difficulty_rows,
        'group_difficulty_summary': group_rows,
        'comparison_summary': comparison_rows,
        'monotonicity': monotonicity,
        'episode_records': json_ready_records(all_records),
        'initial_condition_check': initial_condition_check,
    }
    write_json(out_path / 'summary.json', summary)

    print('')
    print_monotonicity(monotonicity)
    print('')
    print(f'Wrote episode results: {out_path / "episode_results.csv"}')
    print(f'Wrote group summary:   {out_path / "group_difficulty_summary.csv"}')
    print(f'Wrote comparison:      {out_path / "comparison_summary.csv"}')
    if plot_path is not None:
        print(f'Wrote plot:            {plot_path}')


if __name__ == '__main__':
    main()
