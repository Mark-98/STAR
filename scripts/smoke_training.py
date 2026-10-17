"""Exercise fresh Stage-1/Stage-2 construction and updates on real replay prefixes."""
import argparse
import ast
import copy
import json
from pathlib import Path
import zipfile
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from star.rl import utils
from star.rl.arguments import parse_args
from star.rl.algos.stage1 import STARStage1
from star.rl.algos.stage2 import STARStage2
from star.rl.algos.utils.networks import ImageDecoder
from star.rl.checkpoint import load_checkpoint
from verify_checkpoint import digest


def read_prefix(path, count):
    arrays = {}
    with zipfile.ZipFile(path) as archive:
        for key in ('observation', 'action', 'reward', 'discount'):
            with archive.open(key + '.npy') as stream:
                version = np.lib.format.read_magic(stream)
                reader = np.lib.format.read_array_header_1_0 if version == (1, 0) else np.lib.format.read_array_header_2_0
                shape, fortran, dtype = reader(stream)
                if fortran or dtype.hasobject or shape[0] < count:
                    raise ValueError('Unsupported replay prefix format')
                raw = stream.read(count * int(np.prod(shape[1:])) * dtype.itemsize)
                arrays[key] = np.frombuffer(raw, dtype=dtype).reshape(count, *shape[1:]).copy()
    return arrays


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--teacher', type=Path, required=True)
    parser.add_argument('--replay-file', type=Path, required=True)
    parser.add_argument('--original-stage1', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    data = read_prefix(args.replay_file, 16)
    batch = (data['observation'][2:6], data['action'][3:7], data['reward'][3:7],
             data['discount'][3:7] * .99, data['observation'][3:7], data['action'][4:8])
    teacher = load_checkpoint(args.teacher, map_location='cpu')['agent']
    saved = teacher.args
    cfg = parse_args(['--algo', 'star_stage1'])
    for key in ['env', 'task_name', 'seed', 'action_repeat', 'repr_dim', 'hidden_dim', 'lr', 'dyna_alpha', 'ln']:
        setattr(cfg, key, getattr(saved, key))
    cfg.obs_shape, cfg.action_shape = batch[0].shape[1:], batch[1].shape[1]
    cfg.batch_size, cfg.rc, cfg.use_ib, cfg.encoder_type = 4, False, False, 'default'
    cfg.dyna, cfg.gen_type, cfg.teacher_work_dir = 'fwd', 'mlp', str(args.teacher)
    cfg.expected_teacher_dyna_alpha = cfg.dyna_alpha
    cfg.gen_layer, cfg.load_enc, cfg.ad_ln, cfg.weak_aug = 4, True, False, True
    del teacher
    utils.set_seed_everywhere(731)
    fresh = STARStage1(copy.deepcopy(cfg))
    tree = ast.parse(args.original_stage1.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SACFWAgent')
    cls.body = [n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name != '__init__']
    namespace = dict(torch=torch, np=np, F=F, utils=utils, nn=nn)
    exec(compile(ast.Module(body=[cls], type_ignores=[]), str(args.original_stage1), 'exec'), namespace)
    legacy = namespace['SACFWAgent'].__new__(namespace['SACFWAgent'])
    legacy.__dict__.update(copy.deepcopy(fresh.__dict__))
    # The source trains an auxiliary detached image decoder. It does not affect
    # the policy/critic/dynamics parameters retained in the public implementation.
    legacy.image_decoder = ImageDecoder(cfg.repr_dim).cuda()
    legacy.image_decoder_opt = torch.optim.Adam(legacy.image_decoder.parameters(), lr=cfg.lr)
    before = digest(fresh.critic)
    for step in (1, 2):
        utils.set_seed_everywhere(900 + step)
        a = legacy.update(iter([batch[:5]]), step)
        utils.set_seed_everywhere(900 + step)
        b = fresh.update(iter([batch[:5]]), step)
        for key in b:
            assert np.array_equal(torch.as_tensor(a[key]).detach().cpu().numpy(),
                                  torch.as_tensor(b[key]).detach().cpu().numpy()), key
        for name in ['actor', 'critic', 'critic_target', 'dyna']:
            assert digest(getattr(fresh, name)) == digest(getattr(legacy, name)), name
    assert before != digest(fresh.critic)
    report = {'stage1': {'fresh_construction': True, 'real_replay_updates': 2,
                         'original_controlled_updates_bitwise_equal': True}}
    del fresh, legacy
    cfg.algo = 'dad2'
    for mode in ('random', 'random_rotate_shift'):
        cfg.overlay_type = mode
        utils.set_seed_everywhere(731)
        agent = STARStage2(copy.deepcopy(cfg))
        teacher_before = digest(agent.teacher_agent.actor)
        before = digest(agent.adapter)
        values = [agent.update(iter([batch]), step) for step in (1, 2)]
        assert all(np.isfinite(v['G/recon_loss']) for v in values)
        assert digest(agent.adapter) != before
        assert digest(agent.teacher_agent.actor) == teacher_before
        checkpoint = args.output_dir / (mode + '.pt')
        torch.save({'agent': agent, '_global_step': 2}, checkpoint)
        restored = load_checkpoint(checkpoint, map_location='cuda')['agent']
        assert np.array_equal(agent.act(batch[0][0], 2, True), restored.act(batch[0][0], 2, True))
        report[mode] = dict(fresh_construction=True, real_replay_updates=2,
                            loss=values, teacher_unchanged=True, save_reload=True)
        del agent, restored
    # Small real replay copies allow a bounded native Workspace run with normal workers.
    prefix = args.output_dir / 'replay/merged'
    prefix.mkdir(parents=True)
    for index in range(4):
        np.savez_compressed(prefix / f'prefix_{index}_15.npz', **data)
    report.update(full_training=False, batch_size=4,
                  inputs='16 real replay rows; actual Places ImageFolder pipeline for random overlay')
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
