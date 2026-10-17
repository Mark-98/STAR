import argparse


def str2bool(v):
    """Safe boolean parsing for argparse."""
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        vv = v.strip().lower()
        if vv in ("true", "t", "1", "yes", "y", "on"):
            return True
        if vv in ("false", "f", "0", "no", "n", "off"):
            return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


def parse_args(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--actor_load', default=True, type=str2bool)
    parser.add_argument('--actor_update', default=True, type=str2bool)
    parser.add_argument('--causal_config', default=None, type=str)
    parser.add_argument('--causal_condition', default=None, type=str)
    parser.add_argument('--causal_smoke', default=False, type=str2bool)
    parser.add_argument('--weak_aug', default=True, type=str2bool)
    parser.add_argument('--curr_overlay', default=False, type=str2bool)
    parser.add_argument('--train_type', default="offline", type=str)
    parser.add_argument('--ln', default=True, type=str2bool)
    parser.add_argument('--last_ln', default=False, type=str2bool)
    parser.add_argument('--ad_ln', default=False, type=str2bool)
    parser.add_argument('--model_load_dir', default="none", type=str)
    parser.add_argument('--exp_name', default="test", type=str)
    parser.add_argument('--wandb_mode', choices=['online', 'offline', 'disabled'], default=None)
    parser.add_argument('--wandb_project', default=None, type=str)
    parser.add_argument('--wandb_entity', default=None, type=str)
    parser.add_argument('--wandb_group', default=None, type=str)
    parser.add_argument('--expected_teacher_dyna_alpha', default=None, type=float,
                        help='Optional check of loaded teacher metadata; does not add a stage-2 loss.')
    parser.add_argument('--require_full_replay', default=False, type=str2bool,
                        help='Reject subset filters, empty worker partitions and replay capacity truncation.')
    parser.add_argument('--mechanism_config', default=None, type=str,
                        help='Opt-in uniform mechanism diagnostics; prefer experiments.mechanism.run')
    parser.add_argument('--data_load_dir', default="none", type=str)
    parser.add_argument('--log_interval', default=1000, type=int)
    parser.add_argument('--encoder_type', default="default", type=str)
    parser.add_argument('--overlay_ratio', default=0.5, type=float)
    parser.add_argument('--num_train_frames', default=500_000, type=int)
    parser.add_argument('--env', default='dmc', type=str)
    parser.add_argument('--num_eval_episodes', default=10, type=int)
    parser.add_argument("--buffer_dir", default=None, type=str)
    parser.add_argument("--replay_dataset_index", default=-1, type=int)
    parser.add_argument("--replay_dataset_start", default=-1, type=int)
    parser.add_argument("--replay_dataset_end", default=-1, type=int)
    parser.add_argument("--replay_dataset_tail_count", default=-1, type=int)
    parser.add_argument("--replay_dataset_start_ratio", default=-1.0, type=float)
    parser.add_argument("--replay_dataset_end_ratio", default=-1.0, type=float)
    parser.add_argument("--replay_dataset_tail_ratio", default=-1.0, type=float)
    parser.add_argument('--dyna', default="fwd", choices=['fwd'])
    parser.add_argument('--overlay_type', default="random", choices=['random', 'random_rotate_shift'])
    parser.add_argument('--q_act_sample', default=False, type=str2bool)
    parser.add_argument('--use_target_enc', default=True, type=str2bool)
    parser.add_argument('--load_enc', default=True, type=str2bool)
    parser.add_argument('--gen_layer', default=6, type=int)

    parser.add_argument('--critic_target_tau', default=0.01, type=float)
    parser.add_argument('--dyna_alpha', default=1.0, type=float)
    parser.add_argument('--save_snapshot', default=True, type=str2bool)
    parser.add_argument('--skip_eval', default=False, type=str2bool)
    parser.add_argument('--resume', default=False, type=str2bool)

    # Optional 'module:Class' diagnostic hook. Empty keeps the native mechanism hook.
    parser.add_argument('--diagnostic_hook', default='', type=str)

    # environment
    parser.add_argument('--task_name', default='walker_walk', type=str)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--action_repeat', default=4, type=int)  # finger 2, cartpole 8, else 4
    parser.add_argument('--episode_length', default=1000, type=int)
    parser.add_argument('--eval_mode', default='video_hard', type=str)

    # agent
    parser.add_argument('--algo', default='star_stage1', type=str)
    parser.add_argument('--train_steps', default='500k', type=str)
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_steps', default=1000, type=int,
                        help='Minimum collection steps before updates; never an update-burst size.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--hidden_dim', default=1024, type=int)

    # actor
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)

    # critic
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.01, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)

    # architecture
    parser.add_argument('--num_shared_layers', default=11, type=int)
    parser.add_argument('--num_head_layers', default=0, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--feature_dim', default=256, type=int)
    parser.add_argument('--repr_dim', default=256, type=int)

    parser.add_argument('--encoder_tau', default=0.05, type=float)
    parser.add_argument('--nstep', default=1, type=int)

    parser.add_argument('--random_stack', default=False, type=str2bool)
    parser.add_argument('--buffer_save', default=True, type=str2bool)
    # True (native) merges the retained replay into buffer/merged at the end of
    # training. False retains the raw per-episode npz only and defers the merge,
    # which halves peak replay disk; nothing is deleted either way.
    parser.add_argument('--buffer_merge_on_finish', default=True, type=str2bool)
    parser.add_argument('--multi_act', default=False, type=str2bool)
    parser.add_argument('--max_alpha', default=1.0, type=float)
    parser.add_argument('--rc', default=False, type=str2bool)

    parser.add_argument('--robo_camera_name', default='agentview', type=str)
    parser.add_argument('--replay_buffer_size', default=1_000_000, type=int)
    parser.add_argument('--replay_buffer_num_workers', default=4, type=int)
    parser.add_argument('--num_seed_frames', default=1000, type=int,
                        help='Initial random-action collection steps (legacy name; not multiplied by action repeat).')
    parser.add_argument('--use_ib', default=False, type=str2bool)
    parser.add_argument('--reward_scale', default=1.0, type=float)
    parser.add_argument('--teacher_work_dir', default=None, type=str)
    parser.add_argument('--gen_type', default='mlp', choices=['mlp'])
    parser.add_argument('--freeze_aug_encoder_mlp', default=False, type=str2bool)
    parser.add_argument('--clean_loss', default=False, type=str2bool)
    parser.add_argument('--clean_loss_weight', default=1.0, type=float)
    # entropy maximization
    parser.add_argument('--init_temperature', default=0.1, type=float)
    parser.add_argument('--alpha_lr', default=1e-4, type=float)
    parser.add_argument('--lr', default=1e-3, type=float)

    parser.add_argument('--alpha_beta', default=0.5, type=float)






    # eval
    parser.add_argument('--save_freq', default='50k', type=str)
    parser.add_argument('--eval_every_frames', default='100k', type=str)
    # 0 keeps the native behavior: the shifted suites run on the eval_every_frames grid.
    parser.add_argument('--eval_shift_every_frames', default='0', type=str)
    parser.add_argument('--eval_episodes', default=10, type=int)
    parser.add_argument('--distracting_cs_intensity', default=0., type=float)

    # misc
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--log_dir', default='logs', type=str)
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--save_train_video', default=False, type=str2bool)

    args = parser.parse_args(argv)

    args.algo = {'star_stage1': 'dad', 'star_stage2': 'dad2'}.get(args.algo, args.algo)

    assert args.algo in {'dad', 'dad2'}, f'specified algorithm "{args.algo}" is not supported'

    assert args.env in {'dmc', 'robosuite'}, \
        f'specified env "{args.env}" is not supported'

    assert args.eval_mode in {
        'train', 'color_easy', 'color_hard', 'video_easy', 'video_hard',
        'distracting_cs', 'none'
    }, f'specified mode "{args.eval_mode}" is not supported'

    assert args.seed is not None, 'must provide seed for experiment'
    assert args.log_dir is not None, 'must provide a log directory for experiment'

    intensities = {0., 0.025, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5}
    assert args.distracting_cs_intensity in intensities, \
        f'distracting_cs has only been implemented for intensities: {intensities}'

    args.train_steps = int(args.train_steps.replace('k', '000'))
    args.save_freq = int(args.save_freq.replace('k', '000'))
    args.eval_every_frames = int(args.eval_every_frames.replace('k', '000'))
    args.eval_shift_every_frames = int(str(args.eval_shift_every_frames).replace('k', '000'))

    if args.eval_mode == 'none':
        args.eval_mode = None

    args.image_size = 84
    args.image_crop_size = 84

    return args
