"""Train tiny fresh agents and evaluate one full episode; NOT a score benchmark."""
import argparse
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['cheetah_run', 'Door', 'TwoArmPegInHole'], default='cheetah_run')
    parser.add_argument('--run-name', default='smoke_pipeline')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--overlay-type', choices=['random', 'random_rotate_shift'], default='random_rotate_shift')
    args = parser.parse_args()
    if not re.fullmatch(r'[A-Za-z0-9_-]+', args.run_name):
        parser.error('run-name must contain only letters, digits, underscores or hyphens')
    if args.overlay_type == 'random' and not os.environ.get('PLACES365_ROOT'):
        parser.error('random overlay requires PLACES365_ROOT; see docs/ASSETS.md')
    is_dmc = args.task == 'cheetah_run'
    benchmark, repeat, feature, dynamics = ('dmc', 4, 100, 5) if is_dmc else ('robosuite', 1, 256, 1)
    episode_steps = 250 if is_dmc else 500
    # Legacy replay merge discards 1000 rows. Collect enough complete episodes
    # for four nonempty replay workers, then take exactly ten learner steps.
    collect = int(math.ceil(1250 / episode_steps)) * episode_steps
    total = collect + 10
    teacher_name, student_name = args.run_name + '_teacher', args.run_name + '_student'
    report_dir = ROOT / 'outputs' / args.run_name
    for path in (ROOT / 'logs' / teacher_name, ROOT / 'logs' / student_name, report_dir):
        if path.exists():
            parser.error('Use a new run-name; output already exists: ' + str(path))
    report_dir.mkdir(parents=True)
    env = dict(os.environ, MUJOCO_GL=os.environ.get('MUJOCO_GL', 'egl'), WANDB_MODE='disabled')
    def run(label, tokens):
        print('START', label, flush=True)
        with (report_dir / (label + '.log')).open('w') as log:
            subprocess.run([sys.executable] + tokens, cwd=ROOT, env=env,
                           stdout=log, stderr=subprocess.STDOUT, check=True)
        print('PASS', label, flush=True)
    common = ['--env', benchmark, '--task_name', args.task, '--seed', str(args.seed),
              '--action_repeat', str(repeat), '--repr_dim', str(feature), '--dyna_alpha', str(dynamics),
              '--batch_size', '2', '--wandb_mode', 'disabled']
    run('stage1', ['train_stage1.py'] + common + ['--exp_name', teacher_name,
        '--num_train_frames', str(total), '--num_seed_frames', str(collect),
        '--init_steps', str(collect), '--replay_buffer_num_workers', '0', '--skip_eval', 'true'])
    teacher = ROOT / 'logs' / teacher_name / 'model' / ('snapshot_' + str(total * repeat) + '.pt')
    run('stage2', ['train_stage2.py'] + common + ['--exp_name', student_name,
        '--teacher_work_dir', str(teacher), '--buffer_dir', str(ROOT / 'logs' / teacher_name / 'buffer'),
        '--expected_teacher_dyna_alpha', str(dynamics), '--num_train_frames', '2',
        '--overlay_type', args.overlay_type])
    student = ROOT / 'logs' / student_name / 'model/snapshot.pt'
    mode = ('rotate_easy' if args.overlay_type == 'random_rotate_shift' else 'video_hard') if is_dmc else 'medium'
    run('evaluation', ['-m', 'star.evaluation.rl', '--env', benchmark, '--task', args.task,
        '--checkpoint', str(student), '--mode', mode, '--action-repeat', str(repeat),
        '--seed', str(args.seed), '--episodes', '1', '--output-dir', str(report_dir / 'evaluation')])
    import torch
    from star.rl.checkpoint import load_checkpoint
    torch.set_num_threads(1)
    teacher_payload = load_checkpoint(teacher, map_location='cpu')
    student_payload = load_checkpoint(student, map_location='cpu')
    steps = lambda optimizer: max(float(v['step']) for v in optimizer.state.values())
    teacher_steps = steps(teacher_payload['agent'].critic_opt)
    student_steps = steps(student_payload['agent'].adapter_opt)
    assert teacher_steps == 10 and student_steps == 2
    evaluation = json.loads((report_dir / 'evaluation/summary.json').read_text())
    assert evaluation['full_episodes']
    result = dict(task=args.task, seed=args.seed, augmentation=args.overlay_type,
                  fresh_teacher=True, fresh_replay=True, environment_steps=total,
                  initial_collection_steps=collect, initial_consecutive_updates=1,
                  gradient_clipping_enabled=False,
                  teacher_optimizer_steps=teacher_steps, student_optimizer_steps=student_steps,
                  checkpoints_reload=True, full_evaluation_episode=True,
                  paper_score_reproduction=False)
    (report_dir / 'pipeline.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
