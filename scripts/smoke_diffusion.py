"""Small model/data check, not training reproduction or a simulator test.

Uses the real Base/STAR workspace, reduced UNet widths and four denoising steps.
Creates fresh trusted checkpoints; never needs or downloads pretrained weights.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=['can', 'square'], default='can')
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dataset', type=Path, help='Optional absolute-action HDF5; reads only a tiny copy.')
    parser.add_argument('--places-root', type=Path, help='Optional Places365 source instead of batch overlays.')
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=False)

    import hydra
    import torch
    from omegaconf import OmegaConf
    from diffusion_policy.common.normalize_util import get_image_range_normalizer
    from diffusion_policy.model.common.normalizer import LinearNormalizer
    from diffusion_policy.common.pytorch_util import dict_apply
    from diffusion_policy.workspace.train_diffusion_unet_hybrid_workspace import TrainDiffusionUnetHybridWorkspace
    from star.evaluation.eval_robomimic_texture_generalization import load_policy_from_checkpoint

    torch.set_num_threads(1)
    torch.manual_seed(42)
    with hydra.initialize_config_dir(config_dir=str(ROOT / 'configs/diffusion'), version_base=None):
        cfg = hydra.compose(config_name='train_diffusion_unet_hybrid_workspace',
                            overrides=[f'task={args.task}_image_abs', 'training.use_ema=false'])
    cfg.policy.down_dims = [32, 64, 128]
    cfg.policy.num_inference_steps = 4
    cfg.training.device = args.device
    OmegaConf.resolve(cfg)
    obs = {key: torch.rand(2, 2, *spec.shape, device=args.device)
           for key, spec in cfg.shape_meta.obs.items()}
    actions = torch.rand(2, cfg.horizon, 10, device=args.device)
    batch = {'obs': obs, 'action': actions}
    normalizer = LinearNormalizer()
    normalizer.fit({'action': actions, **{key: value for key, value in obs.items()
                    if cfg.shape_meta.obs[key].get('type', 'low_dim') != 'rgb'}})
    for key, spec in cfg.shape_meta.obs.items():
        if spec.get('type') == 'rgb':
            normalizer[key] = get_image_range_normalizer()

    if args.dataset is not None:
        import h5py
        from torch.utils.data import DataLoader
        from diffusion_policy.dataset.robomimic_replay_image_dataset import RobomimicReplayImageDataset
        fixture = args.output_dir / 'tiny_demonstrations.hdf5'
        with h5py.File(args.dataset, 'r') as source, h5py.File(fixture, 'x') as dest:
            data_group = dest.create_group('data')
            data_group.attrs.update(source['data'].attrs)
            for index in range(2):
                original = source[f'data/demo_{index}']
                group = dest.create_group(f'data/demo_{index}')
                group.attrs.update(original.attrs)
                group.attrs['num_samples'] = min(32, len(original['actions']))
                group.create_dataset('actions', data=original['actions'][:32])
                if 'states' in original:
                    group.create_dataset('states', data=original['states'][:32])
                images = group.create_group('obs')
                for key in cfg.shape_meta.obs:
                    images.create_dataset(key, data=original['obs'][key][:32])
            data_group.attrs['total'] = sum(len(dest[f'data/demo_{i}/actions']) for i in range(2))
        dataset = RobomimicReplayImageDataset(
            shape_meta=cfg.shape_meta, dataset_path=str(fixture), horizon=16,
            pad_before=1, pad_after=7, n_obs_steps=2, abs_action=True, use_cache=False)
        batch = dict_apply(next(iter(DataLoader(dataset, batch_size=2))), lambda x: x.to(args.device))
        obs = batch['obs']
        normalizer = dataset.get_normalizer()

    def update(workspace):
        model = workspace.model
        model.set_normalizer(normalizer)
        model.to(args.device)
        model.train()
        trainable = {name: value.detach().clone() for name, value in model.named_parameters()
                     if value.requires_grad}
        frozen = {name: value.detach().clone() for name, value in model.named_parameters()
                  if not value.requires_grad}
        workspace.optimizer.zero_grad(set_to_none=True)
        loss = model.compute_loss(batch)
        assert torch.isfinite(loss), 'Non-finite loss'
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        workspace.optimizer.step()
        params = dict(model.named_parameters())
        assert any(not torch.equal(value, params[name]) for name, value in trainable.items())
        assert all(torch.equal(value, params[name]) for name, value in frozen.items())
        return float(loss.detach())

    base_dir = args.output_dir / 'base'
    base_dir.mkdir()
    base = TrainDiffusionUnetHybridWorkspace(cfg, output_dir=str(base_dir))
    base_loss = update(base)
    base_path = base.save_checkpoint(use_thread=False)
    stage_cfg = OmegaConf.load(ROOT / 'configs/diffusion' / f'v6_{args.task}.yaml')
    stage_cfg.pretrained_checkpoint = base_path
    stage_cfg.policy.down_dims = [32, 64, 128]
    stage_cfg.policy.num_inference_steps = 4
    # Explicit model-only fixture: do not pretend batch overlays are Places365.
    stage_cfg.policy.overlay_mode = 'batch'
    stage_cfg.policy.overlay_root = None
    if args.places_root is not None:
        from diffusion_policy.policy.diffusion_unet_hybrid_image_dad_policy import Places365Overlay
        if not Places365Overlay(root=str(args.places_root)).available():
            raise FileNotFoundError('Places365 train partition not found.')
        stage_cfg.policy.overlay_mode = 'places365'
        stage_cfg.policy.overlay_root = str(args.places_root.resolve())
        stage_cfg.policy.overlay_num_workers = 0
    stage_cfg.training.device = args.device
    OmegaConf.resolve(stage_cfg)
    star_dir = args.output_dir / 'star'
    star_dir.mkdir()
    ours = TrainDiffusionUnetHybridWorkspace(stage_cfg, output_dir=str(star_dir))
    assert all(not p.requires_grad for p in ours.model.model.parameters())
    assert all(not p.requires_grad for p in ours.model.teacher_obs_encoder.parameters())
    stage_loss = update(ours)
    ours_path = ours.save_checkpoint(use_thread=False)
    ours.model.eval()
    torch.manual_seed(42)
    expected = ours.model.predict_action(obs)['action']
    _, restored = load_policy_from_checkpoint(ours_path, str(args.output_dir / 'reload'),
                                              torch.device(args.device), weight_source='model')
    torch.manual_seed(42)
    actual = restored.predict_action(obs)['action']
    assert actual.shape == (2, 8, 10) and torch.isfinite(actual).all()
    assert torch.equal(expected, actual), 'Checkpoint reload changed predictions'
    report = {'passed': True, 'task': args.task, 'device': args.device,
              'base_loss': base_loss, 'star_loss': stage_loss,
              'base_updates': 1, 'star_updates': 1, 'frozen_parameters_unchanged': True,
              'checkpoint_predictions_bitwise_equal': True,
              'base_checkpoint': str(Path(base_path).relative_to(args.output_dir)),
              'star_checkpoint': str(Path(ours_path).relative_to(args.output_dir)),
              'batch_source': 'two demonstrations, first 32 rows each' if args.dataset else 'synthetic',
              'overlay_source': 'places365' if args.places_root else 'batch',
              'scope': 'Reduced UNet, one minibatch update per stage, four denoising steps. '
                       'Not a simulator, full training, or benchmark test.'}
    (args.output_dir / 'report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
