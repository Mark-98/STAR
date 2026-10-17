"""Evaluate a trusted legacy RL agent through the retained environment wrappers.

This lightweight runner does not reproduce every historical visualization script's
RNG trajectory. It preserves agent.act and reports explicit episode seeds/settings.
"""
import argparse
import gc
import json
import os
from pathlib import Path

GEOMETRY_MODES = ['rotate_easy', 'rotate_hard', 'shift_easy', 'shift_hard']


def scene_for_episode(episode):
    """Balance successive episodes across the ten benchmark scenes."""
    return episode % 10


def make_robosuite(args, episode):
    from star.rl.wrappers.robo_wrapper import robo_make
    return robo_make(name=args.task, action_repeat=args.action_repeat,
                     frame_stack=args.frame_stack, seed=args.seed + episode,
                     mode='eval' if args.mode == 'clean' else 'eval-' + args.mode,
                     scene_id=scene_for_episode(episode))


def close_environment(env):
    # dm_env.Environment.close is a no-op inherited by several legacy wrappers;
    # find the Gym adapter and explicitly release its real simulator.
    seen = set()
    while id(env) not in seen:
        seen.add(id(env))
        state = vars(env)
        if '_gym_env' in state:
            state['_gym_env'].close()
            return
        if '_env' not in state:
            env.close()
            return
        env = state['_env']


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--env', choices=['dmc', 'robosuite'], required=True)
    p.add_argument('--task', required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--mode', default='clean', choices=['clean', 'easy', 'medium', 'hard', 'color_easy', 'color_hard', 'video_easy', 'video_hard'] + GEOMETRY_MODES)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--episodes', type=int, default=30,
                   help='Number of full episodes (default: 30). Robosuite scenes are selected automatically.')
    p.add_argument('--action-repeat', type=int, required=True)
    p.add_argument('--frame-stack', type=int, default=3)
    p.add_argument('--max-steps', type=int, help='Smoke testing only; results are explicitly marked truncated.')
    args = p.parse_args(argv)
    if args.episodes <= 0:
        p.error('episodes must be positive')
    if args.max_steps is not None and args.max_steps <= 0:
        p.error('max-steps must be positive')
    if args.env == 'dmc' and args.mode not in ['clean', 'color_easy', 'color_hard', 'video_easy', 'video_hard'] + GEOMETRY_MODES:
        p.error('DMC uses clean/color_easy/color_hard/video_easy/video_hard.')
    if args.env == 'robosuite' and args.mode not in ('clean', 'easy', 'medium', 'hard'):
        p.error('Robosuite uses clean/easy/medium/hard.')
    return args


def main():
    args = parse_args()
    os.environ.setdefault('MUJOCO_GL', 'egl')
    import torch
    from star.rl.checkpoint import load_checkpoint
    from star.rl import utils
    if not torch.cuda.is_available():
        raise RuntimeError('Legacy RL agents retain CUDA-specific tensors; use the documented GPU environment.')
    utils.set_seed_everywhere(args.seed)
    payload = load_checkpoint(args.checkpoint, map_location='cuda')
    agent = payload['agent']
    saved = getattr(agent, 'args', None)
    if saved is not None and getattr(saved, 'task_name', args.task) != args.task:
        raise ValueError('Checkpoint task does not match --task.')
    if saved is not None:
        for key in ('action_repeat', 'frame_stack'):
            if getattr(saved, key, getattr(args, key)) != getattr(args, key):
                raise ValueError('Checkpoint ' + key + ' does not match evaluation settings.')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    rows = []
    for episode in range(args.episodes):
        episode_seed = args.seed + episode
        utils.set_seed_everywhere(episode_seed)
        scene = scene_for_episode(episode) if args.env == 'robosuite' else None
        if args.mode in GEOMETRY_MODES:
            import numpy as np
            from star.environments.geometric.make_env import make_env
            from star.rl.wrappers.loco_wrapper import Gym2DMC
            from star.rl.wrappers.dmc import ActionDTypeWrapper, ExtendedTimeStepWrapper
            from dm_control.suite.wrappers import action_scale
            domain, task = args.task.split('_', 1)
            gym_env = make_env(domain, task, episode_seed, action_repeat=args.action_repeat,
                               frame_stack=args.frame_stack, camera_rotation_mode=args.mode)
            env = ExtendedTimeStepWrapper(action_scale.Wrapper(
                ActionDTypeWrapper(Gym2DMC(gym_env), np.float32), minimum=-1., maximum=1.))
        elif args.env == 'dmc':
            from star.rl.wrappers.loco_wrapper import make
            kind, difficulty = ('original', 'easy') if args.mode == 'clean' else args.mode.split('_')
            env = make(args.task, args.frame_stack, args.action_repeat, args.seed + episode,
                       type=kind, difficulty=difficulty, object_color=difficulty if kind == 'color' else 'original')
        else:
            env = make_robosuite(args, episode)
        try:
            ts, total, steps = env.reset(), 0.0, 0
            while not ts.last():
                with torch.no_grad(), utils.eval_mode(agent):
                    action = agent.act(ts.observation, payload.get('_global_step', 0), eval_mode=True)
                ts = env.step(action)
                total += float(ts.reward or 0)
                steps += 1
                if args.max_steps is not None and steps >= args.max_steps:
                    break
            rows.append({'episode': episode, 'seed': args.seed + episode,
                         'scene_id': scene,
                         'return': total, 'steps': steps, 'complete_episode': bool(ts.last())})
            print(f'Episode {episode + 1}/{args.episodes}: return={total:.3f}, '
                  f'steps={steps}, complete={bool(ts.last())}', flush=True)
        finally:
            close_environment(env)
            # Robosuite modders can retain render-context reference cycles.
            # Collect them now, while EGL is alive, not during Python shutdown.
            del env
            gc.collect()
    result = {'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'scene_selection': 'automatic_round_robin_10' if args.env == 'robosuite' else None,
              'full_episodes': all(r['complete_episode'] for r in rows),
              'episodes': rows, 'mean_return': sum(r['return'] for r in rows) / len(rows)}
    (args.output_dir / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print('mean_return:', result['mean_return'])


if __name__ == '__main__':
    main()
