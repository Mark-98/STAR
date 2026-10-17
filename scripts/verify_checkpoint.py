"""Bounded GPU regression against trusted original source, without importing that project.

Loads the requested trusted checkpoint and compares the actual original act/update
methods (via AST) with the public implementation. This is not full retraining.
"""
import argparse
import ast
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from PIL import Image
from star.rl import utils, augmentations
from star.rl.checkpoint import load_checkpoint
from star.rl.arguments import parse_args


def digest(module):
    result = hashlib.sha256()
    for key, value in module.state_dict().items():
        result.update(key.encode())
        result.update(value.detach().cpu().contiguous().numpy().tobytes())
    return result.hexdigest()


def original_method(path, name):
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'DiffusionGAN_Agent')
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == name)
    namespace = dict(torch=torch, F=F, np=np, utils=utils, a=augmentations,
                     random_overlay=utils.random_overlay)
    exec(compile(ast.Module(body=[fn], type_ignores=[]), str(path), 'exec'), namespace)
    return namespace[name]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--original-source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--real-observation', type=Path, help='84px RGB initial image; repeats across frame stack.')
    p.add_argument('--places-image', type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    utils.set_seed_everywhere(731)
    agent = load_checkpoint(args.checkpoint, map_location='cuda')['agent']
    agent.train(False)
    original = original_method(args.original_source, 'act')
    legacy = copy.copy(agent)
    legacy.generator_diffusion = agent.adapter
    rng = np.random.RandomState(731)
    observations = [rng.randint(0, 256, size=(9, 84, 84), dtype=np.uint8) for _ in range(3)]
    if args.real_observation:
        rgb = np.asarray(Image.open(args.real_observation).convert('RGB'))
        assert rgb.shape == (84, 84, 3)
        observations.append(np.tile(np.moveaxis(rgb, -1, 0), (3, 1, 1)))
    for obs in observations:
        utils.set_seed_everywhere(71)
        expected = original(legacy, obs, 0, eval_mode=True)
        utils.set_seed_everywhere(71)
        actual = agent.act(obs, 0, eval_mode=True)
        assert np.array_equal(expected, actual), np.max(abs(expected - actual))
    # Explicit current recipe defaults fill metadata absent from older snapshots;
    # this tests the update implementation, not unknown historical hyperparameters.
    cfg = parse_args(['--algo', 'star_stage2'])
    vars(cfg).update(vars(agent.args))
    cfg.curr_overlay, cfg.clean_loss, cfg.freeze_aug_encoder_mlp = False, False, False
    cfg.gen_type = 'mlp'
    cfg.weak_aug = True
    agent.args = cfg
    agent.freeze_aug_encoder_mlp = False
    legacy = copy.deepcopy(agent)
    legacy.generator_diffusion = legacy.adapter
    legacy.generator_diffusion_opt = legacy.adapter_opt
    old_update = original_method(args.original_source, 'update')
    obs = np.stack(observations[:2])
    batch = (obs, np.zeros((2, agent.action_shape), dtype=np.float32),
             np.zeros((2, 1), dtype=np.float32), np.ones((2, 1), dtype=np.float32), obs, np.zeros((2, agent.action_shape), dtype=np.float32))
    # random_ratio_overlay needs 84 + 2 * 3 stacked frames * shift_max(6).
    fixture = np.asarray(Image.open(args.places_image).convert('RGB').resize((120, 120))).copy()
    fixture = torch.as_tensor(fixture, device='cuda').permute(2, 0, 1).float().unsqueeze(0).repeat(2, 1, 1, 1) / 255.
    augmentations.places_dataloader = object()
    augmentations._get_places_batch = lambda batch_size: fixture[:batch_size]
    before_teacher = digest(agent.teacher_agent.actor)
    before_encoder = digest(agent.aug_encoder)
    losses = []
    for step in range(1, 4):
        utils.set_seed_everywhere(800 + step)
        expected = old_update(legacy, iter([batch]), step)
        rng_expected = torch.cuda.get_rng_state()
        utils.set_seed_everywhere(800 + step)
        actual = agent.update(iter([batch]), step)
        assert expected == actual, (expected, actual)
        assert torch.equal(rng_expected, torch.cuda.get_rng_state())
        assert digest(legacy.aug_encoder) == digest(agent.aug_encoder)
        assert digest(legacy.adapter) == digest(agent.adapter)
        losses.append(actual)
    assert before_teacher == digest(agent.teacher_agent.actor)
    assert before_encoder != digest(agent.aug_encoder)
    import io
    stream = io.BytesIO()
    torch.save({'agent': agent}, stream)
    stream.seek(0)
    restored = load_checkpoint(stream, map_location='cuda')['agent']
    assert np.array_equal(agent.act(obs[0], 0, True), restored.act(obs[0], 0, True))
    utils.set_seed_everywhere(900)
    first = agent.update(iter([batch]), 4)
    utils.set_seed_everywhere(900)
    second = restored.update(iter([batch]), 4)
    assert first == second and digest(agent.adapter) == digest(restored.adapter)
    result = dict(task=cfg.task_name, training_seed=cfg.seed, augmentation=cfg.overlay_type,
                  inference_inputs=len(observations), original_actions_bitwise_equal=True,
                  original_three_updates_bitwise_equal=True, teacher_unchanged=True,
                  student_changed=True, checkpoint_roundtrip=True, optimizer_resume_equal=True,
                  batch_size=2, losses=losses, inputs='synthetic observations and fixed real Places image',
                  full_training=False, original_source_sha256=hashlib.sha256(args.original_source.read_bytes()).hexdigest())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
