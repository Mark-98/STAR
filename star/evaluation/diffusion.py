"""Evaluate trusted Base/Ours checkpoints on the frozen STAR visual benchmark."""
import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from star.environments.scenes import LEVELS, TASKS, load_scenes
from star.utils.paths import resource


def build_config(task, dataset, base, ours, seed):
    levels = {'clean': {'apply_visual_mod': False}}
    for level in LEVELS:
        levels[level] = {
            'apply_visual_mod': True,
            # Preserve v6's exact wrapper mode, including explicit null overrides.
            'visual_mode': 'eval-easy',
            'scene_source_file': str(resource('configs', 'scenes', task + '.json')),
            'scene_ids': [r['scene_id'] for r in load_scenes(task, level)],
        }
    return {
        'task': {'name': task + '_image_abs', 'dataset_path': str(Path(dataset).expanduser().resolve())},
        'models': {
            f'base_seed{seed}': {'group': 'base', 'train_seed': seed, 'checkpoint': str(Path(base).expanduser().resolve())},
            f'ours_seed{seed}': {'group': 'ours', 'train_seed': seed, 'checkpoint': str(Path(ours).expanduser().resolve())},
        },
        'robustness_eval': {'order': ['clean', *LEVELS], **levels},
        'evaluation': {'seeds': [42] * 30, 'seed_strategy': 'paired', 'n_envs': 10,
                       'policy_seed': 42, 'difficulty_layout': 'default', 'use_record_eval_seed': False,
                       'disable_material': False, 'disable_skybox': False, 'disable_light': False,
                       'exact_scene_config': True, 'fixed_env_seed': 42},
        'calibration': {'selection_kind': 'outcome_calibrated_v1_block_subset',
                        'source_version': 'v6', 'note': 'Outcome-selected scenes; demo150..199 reused from v4/v5, not untouched test data.'},
    }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task', required=True, choices=TASKS)
    p.add_argument('--dataset', required=True)
    p.add_argument('--base-checkpoint', required=True)
    p.add_argument('--ours-checkpoint', required=True)
    p.add_argument('--model-seed', type=int, choices=[0, 1, 2], default=0)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--phase', choices=['all', 'clean', 'visual'], default='all')
    p.add_argument('--episodes', '--num-episodes', dest='num_episodes', type=int, default=50,
                   help='Episodes per model and condition; visual scenes cycle automatically.')
    p.add_argument('--demo-start', type=int, default=150)
    p.add_argument('--n-envs', type=int, default=10)
    p.add_argument('--dry-run', action='store_true', help='Print plan without importing simulators or loading weights.')
    args = p.parse_args(argv)
    if args.num_episodes <= 0 or args.n_envs <= 0 or args.demo_start < 0:
        p.error('Episode/environment counts must be positive and demo-start nonnegative.')
    return args


def main(argv=None):
    args = parse_args(argv)
    cfg = build_config(args.task, args.dataset, args.base_checkpoint, args.ours_checkpoint, args.model_seed)
    if args.dry_run:
        conditions = ['clean', *LEVELS] if args.phase == 'all' else (
            ['clean'] if args.phase == 'clean' else list(LEVELS))
        print(json.dumps({'task': args.task, 'conditions': conditions,
                          'episodes_per_model_per_condition': args.num_episodes,
                          'total_episodes': 2 * len(conditions) * args.num_episodes,
                          'scene_assignment': 'automatic cycle, 10 fixed scenes per visual condition',
                          'protocol': vars(args), 'calibration': cfg['calibration']},
                         indent=2, default=str))
        return
    for path in [cfg['task']['dataset_path'], *[m['checkpoint'] for m in cfg['models'].values()]]:
        if not Path(path).is_file():
            raise FileNotFoundError(path)
    from star.evaluation.eval_robomimic_legacy_original import evaluate
    args.output_dir.mkdir(parents=True, exist_ok=False)
    config_path = args.output_dir / 'resolved_config.json'
    config_path.write_text(json.dumps(cfg, indent=2) + '\n')  # JSON is also valid YAML.
    for phase in (['clean', 'visual'] if args.phase == 'all' else [args.phase]):
        evaluate(SimpleNamespace(
            config=config_path, scene_config=config_path if phase == 'visual' else None,
            model_seed=args.model_seed, output_dir=args.output_dir / phase,
            num_episodes=args.num_episodes, demo_start=args.demo_start, n_envs=args.n_envs,
            policy_rng_seed=42, device=args.device, models='both', base_weights='model',
            ours_weights='model', ours_actor_weights='checkpoint', scene_assignment='cycle'))


if __name__ == '__main__':
    main()
