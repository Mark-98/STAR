#!/usr/bin/env python3
"""Evaluate existing checkpoints using the original demo-state runner."""
import argparse
import csv
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics

import h5py
import dill
import numpy as np
import torch
import yaml

from diffusion_policy.env_runner.robomimic_image_runner import RobomimicImageRunner
from diffusion_policy.env.robomimic.robomimic_image_wrapper import RobomimicVisualModWrapper
from star.utils.paths import resource
from star.environments.scenes import assign_scene_episodes
from star.evaluation.eval_robomimic_texture_generalization import (
    load_policy_from_checkpoint, replace_frozen_actor_with_base_ema,
)


ROOT = Path(__file__).resolve().parent


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def records_from_log(log, model, task, checkpoint, demo_states, specs=None, demo_start=0):
    records = []
    difficulty = 'clean' if specs is None else specs[0]['difficulty']
    prefix = 'train/' if specs is None else f'eval_{difficulty}/'
    if specs is not None and len(specs) != len(demo_states):
        raise ValueError('One scene per demo is required.')
    for idx, state in enumerate(demo_states):
        demo_id = demo_start + idx
        key = f'{prefix}sim_max_reward_{demo_id}'
        if key not in log:
            raise ValueError(f'Missing legacy episode result: {key}')
        reward = float(log[key])
        if reward not in (0., 1.):
            raise ValueError(f'Expected sparse success reward, got {reward}')
        records.append({
            'task': task, 'model': model['label'], 'model_group': model['group'],
            'train_seed': model['train_seed'], 'checkpoint': checkpoint,
            'protocol': 'legacy_demo', 'difficulty': difficulty,
            'scene_id': None if specs is None else specs[idx]['scene_id'],
            'env_seed': None, 'demo_id': demo_id, 'episode': idx,
            'demo_state_hash': hashlib.sha256(np.asarray(state, dtype='<f8').tobytes()).hexdigest(),
            'success': int(reward), 'max_reward': reward,
            'visual_config': None if specs is None else specs[idx]['visual_config'],
            'exact_visual_config': None if specs is None else specs[idx]['exact_visual_config'],
        })
    if not np.isclose(np.mean([r['success'] for r in records]), log[prefix + 'mean_score']):
        raise ValueError('Legacy mean score disagrees with per-episode rewards.')
    return records


def visual_specs(scene_config, dataset_path, count, assignment='exact'):
    from star.evaluation.eval_robomimic_scene_difficulty_compare import (
        build_all_specs, get_eval_config, load_env_meta, merged_difficulty_config,
    )
    evaluation = get_eval_config(scene_config)
    evaluation.update({'use_record_eval_seed': False, 'seed_strategy': 'paired'})
    specs = build_all_specs(
        env_meta=load_env_meta(dataset_path), difficulty_cfg=merged_difficulty_config(scene_config),
        eval_cfg=evaluation, difficulty_filter=['easy', 'medium', 'hard'],
        seed_values=list(range(count)), cli_scene_ids=None, cli_scene_id=[],
        save_video=False, videos_dir=resource('outputs', 'videos'),
    )
    return assign_scene_episodes(specs, count, assignment)


def configure_visual_demo_rollouts(runner, specs, demo_states, demo_start=0):
    if len(specs) != len(demo_states):
        raise ValueError('Visual scenes must match the demo count.')
    initializers = []
    for spec, state in zip(specs, demo_states):
        def initialize(env, spec=copy.deepcopy(spec), state=np.array(state, copy=True)):
            wrapped = env.env.env
            if not isinstance(wrapped, RobomimicVisualModWrapper):
                raise TypeError('Visual demo evaluation requires the visual wrapper.')
            wrapped.init_state = state
            wrapped._seed = None
            for key in ('texture_map', 'material_map', 'skybox_config', 'light_config'):
                setattr(wrapped, key, copy.deepcopy(spec['visual_config'][key]))
            wrapped.visual_config_override = (
                copy.deepcopy(spec['visual_config']) if spec['exact_visual_config'] else None)
            wrapped.texture_seed = spec.get('texture_seed')
            wrapped.visual_mode = spec['visual_mode']
            wrapped.scene_id = spec['scene_id']
            wrapped.randomize_on_reset = False
            wrapped._has_applied_visual_mod = False
            wrapped.reset()
            actual = np.asarray(wrapped.get_state()['states'])
            if not np.allclose(actual, state, rtol=0, atol=1e-10):
                raise ValueError(f'Visual scene changed physical initial state: {spec["scene_id"]}')
        initializers.append(dill.dumps(initialize))
    runner.env_init_fn_dills = initializers
    # Log keys carry the demo ID so held-out ranges never collide with demo_0..N-1 results.
    runner.env_seeds = list(range(demo_start, demo_start + len(specs)))
    runner.env_prefixs = [f'eval_{spec["difficulty"]}/' for spec in specs]


def seed_policy(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate(args):
    config = yaml.safe_load(args.config.read_text())
    models = [(label, value) for label, value in config['models'].items()
              if value['train_seed'] == args.model_seed]
    if len(models) != 2 or not any(m['group'] == 'base' for _, m in models):
        raise ValueError('Expected one Base and one Ours for the selected model seed.')
    base_checkpoint = next(m['checkpoint'] for _, m in models if m['group'] == 'base')
    if args.models != 'both':
        models = [(label, model) for label, model in models
                  if (model['group'] == 'base') == (args.models == 'base')]
    args.output_dir.mkdir(parents=True, exist_ok=False)
    dataset_path = config['task']['dataset_path']
    demo_ids = list(range(args.demo_start, args.demo_start + args.num_episodes))
    with h5py.File(dataset_path, 'r') as dataset:
        if any(f'demo_{idx}' not in dataset['data'] for idx in demo_ids):
            raise ValueError(f'Dataset has no demo range {demo_ids[0]}..{demo_ids[-1]}.')
        demo_states = [dataset[f'data/demo_{idx}/states'][0] for idx in demo_ids]
    scene_config = yaml.safe_load(args.scene_config.read_text()) if args.scene_config else None
    specs = None
    if scene_config is not None:
        if scene_config['task'] != config['task'] or scene_config['models'] != config['models']:
            raise ValueError('Scene config must use the same task and checkpoints.')
        specs = visual_specs(scene_config, dataset_path, args.num_episodes, args.scene_assignment)
    evaluation = {
        'protocol': 'legacy_demo', 'n_train': args.num_episodes if specs is None else 0,
        'train_start_idx': args.demo_start, 'demo_start': args.demo_start,
        'demo_ids': demo_ids,
        'n_test': 0, 'test_start_seed': 100000, 'n_envs': args.n_envs,
        'policy_rng_seed': args.policy_rng_seed, 'clean_scene_id': None,
        'policy_rng_scope': 'reset_before_each_model_and_difficulty',
        'model_filter': args.models,
        'base_weight_source': args.base_weights,
        'ours_weight_source': args.ours_weights,
        'ours_actor_weights': args.ours_actor_weights,
        'scene_assignment': args.scene_assignment,
        'visual_episode_counts': {} if specs is None else {
            level: args.num_episodes for level in ('easy', 'medium', 'hard')},
        'phase': 'original' if specs is None else 'visual',
        'num_episodes_per_difficulty': args.num_episodes,
        'scene_config': scene_config,
        'note': 'Original runner and demo resets; policy RNG is reset before each model. '
                'Not a bitwise replay of the RNG state in historical training logs.',
    }
    write_json(args.output_dir / 'run_config.json', {
        'config_path': str(args.config), 'task': config['task'],
        'evaluation': evaluation, 'models': dict(models),
        'runtime': {'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')},
    })
    records = []
    reference_protocol = None
    for label, model in models:
        print(f'Loading {label}: {model["checkpoint"]}', flush=True)
        output_dir = args.output_dir / label
        output_dir.mkdir()
        cfg, policy = load_policy_from_checkpoint(
            model['checkpoint'], str(output_dir / 'workspace_load'), torch.device(args.device),
            weight_source=args.base_weights if model['group'] == 'base' else args.ours_weights)
        requested_weights = args.base_weights if model['group'] == 'base' else args.ours_weights
        selected_weights = (
            ('ema' if cfg.training.use_ema else 'model')
            if requested_weights == 'auto' else requested_weights)
        actor_source = selected_weights
        actor_checkpoint = model['checkpoint']
        if model['group'] != 'base' and args.ours_actor_weights == 'base_ema':
            actor_audit = replace_frozen_actor_with_base_ema(policy, base_checkpoint)
            write_json(output_dir / 'actor_override.json', actor_audit)
            actor_source = 'base_ema'
            actor_checkpoint = base_checkpoint
            print('Ours EMA actor loaded; all non-actor tensors are unchanged.', flush=True)
        write_json(output_dir / 'checkpoint_selection.json', {
            'checkpoint': model['checkpoint'], 'requested': requested_weights,
            'selected': selected_weights, 'training_seed': int(cfg.training.seed),
            'actor_weight_source': actor_source, 'actor_checkpoint': actor_checkpoint,
        })
        runner_cfg = cfg.task.env_runner
        protocol = {
            'max_steps': int(runner_cfg.max_steps),
            'n_obs_steps': int(runner_cfg.n_obs_steps),
            'n_action_steps': int(runner_cfg.n_action_steps),
            'abs_action': bool(runner_cfg.abs_action),
            'past_action': bool(runner_cfg.get('past_action', False)),
        }
        if reference_protocol is None:
            reference_protocol = protocol
        elif protocol != reference_protocol:
            raise ValueError('Base/Ours environment protocol mismatch.')
        print(f'Legacy {evaluation["phase"]}: demo_{demo_ids[0]}..demo_{demo_ids[-1]}, '
              f'{args.n_envs} environments, {protocol}', flush=True)
        runner = RobomimicImageRunner(
            output_dir=str(output_dir), dataset_path=dataset_path,
            shape_meta=runner_cfg.shape_meta, n_train=args.num_episodes if specs is None else 0,
            n_eval_easy=0 if specs is None else args.num_episodes,
            n_train_vis=0, train_start_idx=args.demo_start, n_test=0, n_test_vis=0,
            test_start_seed=100000, n_envs=args.n_envs,
            render_obs_key=runner_cfg.render_obs_key, fps=int(runner_cfg.fps),
            tqdm_interval_sec=30, **protocol,
        )
        model_records = []
        log = {}
        try:
            for difficulty in (['clean'] if specs is None else ['easy', 'medium', 'hard']):
                current_specs = None if specs is None else [s for s in specs if s['difficulty'] == difficulty]
                if current_specs is not None:
                    configure_visual_demo_rollouts(runner, current_specs, demo_states, args.demo_start)
                seed_policy(args.policy_rng_seed)
                current_log = {key: float(value) for key, value in runner.run(policy).items()}
                log.update(current_log)
                current_records = records_from_log(
                    current_log, {'label': label, **model}, config['task']['name'],
                    model['checkpoint'], demo_states, current_specs, args.demo_start)
                for record in current_records:
                    record['weight_source'] = selected_weights
                    record['actor_weight_source'] = actor_source
                    record['actor_checkpoint'] = actor_checkpoint
                model_records.extend(current_records)
                write_json(output_dir / f'{difficulty}_episodes.json', current_records)
                print(f'{label} {difficulty}: {sum(r["success"] for r in current_records)}'
                      f'/{args.num_episodes}', flush=True)
        finally:
            runner.env.close()
        log = {key: float(value) for key, value in log.items()}
        write_json(output_dir / 'eval_log.json', log)
        records.extend(model_records)
        write_json(output_dir / 'episode_results.json', model_records)
        del runner, policy
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    write_json(args.output_dir / 'summary.json', {
        'task': config['task']['name'], 'evaluation': evaluation,
        'environment_protocol': reference_protocol, 'episode_records': records,
    })


def summarize(root):
    all_records, rows = [], []
    lines = ['# Legacy Original Evaluation', '',
             'Original RobomimicImageRunner; shared demo initial states; no scene ID.', '',
             'Demo ranges are stated per task below the table.', '',
             '| Task | Model | train0 | train1 | train2 | Mean +/- SD (%) |',
             '| --- | --- | ---: | ---: | ---: | ---: |']
    for task in ('can', 'square'):
        protocol = None
        expected_models = None
        task_records = []
        for seed in range(3):
            directory = root / 'shards' / task / f'seed{seed}'
            result = json.loads((directory / 'summary.json').read_text())
            current = (result['evaluation'], result['environment_protocol'])
            if protocol is None:
                protocol = current
                config = yaml.safe_load((root / 'configs' / f'{task}.yaml').read_text())
                expected_models = config['models']
            elif protocol != current:
                raise ValueError(f'Protocols differ for {task}.')
            task_records.extend(result['episode_records'])
        num_episodes = protocol[0]['n_train']
        demo_start = protocol[0].get('demo_start', 0)
        groups = {}
        reference_states = {}
        for label, model in expected_models.items():
            selected = [r for r in task_records if r['model'] == label]
            expected_ids = set(range(demo_start, demo_start + num_episodes))
            if len(selected) != num_episodes or {r['demo_id'] for r in selected} != expected_ids:
                raise ValueError(f'Incomplete legacy results for {label}.')
            for record in selected:
                if record['checkpoint'] != model['checkpoint'] or record['scene_id'] is not None:
                    raise ValueError('Checkpoint or scene mismatch.')
                key = record['demo_id']
                if reference_states.setdefault(key, record['demo_state_hash']) != record['demo_state_hash']:
                    raise ValueError('Base/Ours demo states differ.')
            count = sum(r['success'] for r in selected)
            groups.setdefault(model['group'], {})[model['train_seed']] = count / num_episodes
        for group, values in groups.items():
            rates = [values[seed] * 100 for seed in range(3)]
            avg, std = statistics.mean(rates), statistics.pstdev(rates)
            rows.append({'task': task, 'model_group': group, 'num_models': 3,
                         'num_episodes_per_model': num_episodes,
                         'mean_success_rate': avg / 100, 'std_success_rate': std / 100})
            lines.append('| ' + ' | '.join([task, group] + [f'{r:.1f}' for r in rates]
                                          + [f'{avg:.1f} +/- {std:.1f}']) + ' |')
        all_records.extend(task_records)
        lines.append(f'{task}: demo_{demo_start}..demo_{demo_start + num_episodes - 1}')
    write_json(root / 'summary.json', {'group_summary': rows, 'episode_records': all_records})
    for name, items in [('group_summary.csv', rows), ('episode_results.csv', all_records)]:
        with (root / name).open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(items[0]))
            writer.writeheader()
            writer.writerows(items)
    (root / 'RESULTS.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    run = sub.add_parser('run')
    run.add_argument('--config', type=Path, required=True)
    run.add_argument('--model-seed', type=int, required=True, choices=[0, 1, 2])
    run.add_argument('--output-dir', type=Path, required=True)
    run.add_argument('--num-episodes', type=int, default=30)
    run.add_argument('--demo-start', type=int, default=0,
                     help='First demo index; use 50 for held-out demo_50..99 after calibrating on demo_0..49.')
    run.add_argument('--n-envs', type=int, default=10)
    run.add_argument('--policy-rng-seed', type=int, default=42)
    run.add_argument('--device', default='cuda:0')
    run.add_argument('--models', choices=['both', 'base', 'ours'], default='both')
    run.add_argument('--base-weights', choices=['auto', 'model', 'ema'], default='auto',
                     help='Override Base checkpoint weights; auto preserves the saved EMA setting.')
    run.add_argument('--ours-weights', choices=['auto', 'model', 'ema'], default='auto',
                     help='Select the saved Ours policy weights, before any optional actor override.')
    run.add_argument('--ours-actor-weights', choices=['checkpoint', 'base_ema'], default='checkpoint',
                     help='Optionally replace only the frozen Ours actor with its paired Base EMA actor.')
    run.add_argument('--scene-config', type=Path,
                     help='Evaluate configured visual scenes on the same demo states, after Original.')
    run.add_argument('--scene-assignment', choices=['exact', 'cycle'], default='exact',
                     help='Cycle existing scenes when episode count differs; never changes visual parameters.')
    merge = sub.add_parser('summarize')
    merge.add_argument('--output-root', type=Path, required=True)
    args = parser.parse_args()
    if args.command == 'run':
        if args.num_episodes < 1 or args.n_envs < 1:
            parser.error('Episode and environment counts must be positive.')
        if args.demo_start < 0:
            parser.error('Demo start must be non-negative.')
        if args.models == 'base' and args.ours_actor_weights != 'checkpoint':
            parser.error('--ours-actor-weights requires evaluating Ours.')
        evaluate(args)
    else:
        summarize(args.output_root)


if __name__ == '__main__':
    main()
