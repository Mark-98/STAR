"""Launch retained RL workspaces with YAML presets plus original CLI overrides."""
import argparse
import os
import sys
from pathlib import Path
import yaml


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage', choices=['stage1', 'stage2', 'teacher'], required=True,
                        help='STAR stage1 (legacy dad) or stage2 (legacy dad2); teacher is a compatibility alias.')
    parser.add_argument('--config', type=Path)
    parser.add_argument('--print-config', action='store_true')
    options, rest = parser.parse_known_args()
    stage = 'stage1' if options.stage == 'teacher' else options.stage
    if options.config is None:
        options.config = Path(__file__).resolve().parents[2] / 'configs/rl' / (stage + '.yaml')
    values = yaml.safe_load(options.config.read_text()) if options.config else {}
    tokens = []
    for key, value in (values or {}).items():
        if value is None:
            continue
        if key == 'save_video':
            if value:
                tokens.append('--save_video')
            continue
        tokens.extend(['--' + key, str(value).lower() if isinstance(value, bool) else str(value)])
    from star.rl.arguments import parse_args
    cfg = parse_args(tokens + rest)
    cfg.stage = stage
    if (stage == 'stage1' and cfg.algo == 'dad2') or (stage == 'stage2' and cfg.algo != 'dad2'):
        parser.error('STAR stage1 uses dad; STAR stage2 uses dad2.')
    if cfg.env not in ('dmc', 'robosuite') or cfg.algo not in ('dad', 'dad2'):
        parser.error('Release scope: DMC/Robosuite STAR Stage 1 and Stage 2 only.')
    if cfg.causal_config or cfg.mechanism_config or cfg.diagnostic_hook:
        parser.error('Experimental diagnostic hooks are not included in the release.')
    if cfg.algo in ('dad', 'dad2') and (cfg.dyna != 'fwd' or cfg.use_ib or cfg.encoder_type != 'default'):
        parser.error('STAR uses forward dynamics and the default non-IB encoder only.')
    if stage == 'stage2' and (cfg.gen_type != 'mlp' or cfg.rc or cfg.overlay_type not in ('random', 'random_rotate_shift')
                             or cfg.q_act_sample or cfg.multi_act):
        parser.error('STAR Stage 2 supports MLP, random/random_rotate_shift augmentation and single-action inference.')
    if options.print_config:
        print(yaml.safe_dump(vars(cfg), sort_keys=True))
        return
    if stage == 'stage2' and (not cfg.buffer_dir or not cfg.teacher_work_dir):
        parser.error('stage2 requires --buffer_dir and --teacher_work_dir.')
    if cfg.wandb_mode is not None:
        os.environ['WANDB_MODE'] = cfg.wandb_mode
    else:
        os.environ.setdefault('WANDB_MODE', 'offline')
    from importlib import import_module
    import_module('star.rl.train_' + ('teacher' if stage == 'stage1' else 'stage2')).main(cfg)


if __name__ == '__main__':
    main()
