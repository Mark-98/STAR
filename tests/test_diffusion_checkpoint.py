"""Checkpoint naming and legacy resume compatibility (no simulator needed)."""
from importlib.util import find_spec
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from omegaconf import OmegaConf
import torch

BaseWorkspace = None
if find_spec('dill') is not None:
    from diffusion_policy.workspace.base_workspace import BaseWorkspace


@unittest.skipIf(BaseWorkspace is None, 'Install the optional Diffusion Policy environment (dill).')
class DiffusionCheckpointTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.workspace = BaseWorkspace(OmegaConf.create({}), output_dir=str(self.root))
        self.workspace.model = torch.nn.Linear(2, 2)

    def test_default_save_and_load_roundtrip(self):
        expected = self.workspace.model.weight.detach().clone()
        path = Path(self.workspace.save_checkpoint(use_thread=False))
        self.assertEqual(path, self.root / 'checkpoints/last_snapshot.ckpt')
        self.assertFalse((path.parent / 'latest.ckpt').exists())
        with torch.no_grad():
            self.workspace.model.weight.zero_()
        self.workspace.load_checkpoint()
        self.assertTrue(torch.equal(expected, self.workspace.model.weight))

    def test_async_save_uses_same_default(self):
        path = Path(self.workspace.save_checkpoint())
        self.workspace._saving_thread.join(timeout=10)
        self.assertFalse(self.workspace._saving_thread.is_alive())
        self.assertEqual(path, self.workspace.get_checkpoint_path())
        self.assertTrue(path.is_file())
        self.workspace.load_checkpoint()

    def test_legacy_latest_is_resumed_if_new_file_is_absent(self):
        expected = self.workspace.model.weight.detach().clone()
        path = Path(self.workspace.save_checkpoint(tag='latest', use_thread=False))
        self.assertEqual(self.workspace.get_resume_checkpoint_path(), path)
        with torch.no_grad():
            self.workspace.model.weight.zero_()
        self.workspace.load_checkpoint()
        self.assertTrue(torch.equal(expected, self.workspace.model.weight))
        # Saving a legacy run must still produce the new filename.
        new_path = Path(self.workspace.save_checkpoint(use_thread=False))
        self.assertEqual(new_path.name, 'last_snapshot.ckpt')
        self.assertTrue(path.is_file())

    def test_new_file_takes_precedence_over_legacy(self):
        self.workspace.save_checkpoint(tag='latest', use_thread=False)
        with torch.no_grad():
            self.workspace.model.weight.fill_(2)
        path = Path(self.workspace.save_checkpoint(use_thread=False))
        self.assertEqual(self.workspace.get_resume_checkpoint_path(), path)
        with torch.no_grad():
            self.workspace.model.weight.zero_()
        self.workspace.load_checkpoint()
        self.assertTrue(torch.equal(torch.full((2, 2), 2.), self.workspace.model.weight))

    def test_explicit_paths_and_tags_are_unchanged(self):
        tagged = Path(self.workspace.save_checkpoint(tag='custom', use_thread=False))
        self.assertEqual(tagged, self.root / 'checkpoints/custom.ckpt')
        self.workspace.load_checkpoint(tag='custom')
        explicit = self.root / 'old_epoch=0019.ckpt'
        self.assertEqual(Path(self.workspace.save_checkpoint(path=explicit, use_thread=False)), explicit)
        self.workspace.load_checkpoint(path=explicit)
        with self.assertRaises(FileNotFoundError):
            self.workspace.load_checkpoint(tag='missing')

    def test_all_recipes_save_last_without_score_named_files(self):
        root = Path(__file__).resolve().parents[1] / 'configs/diffusion'
        for name in ('train_diffusion_unet_hybrid_workspace',
                     'train_diffusion_unet_hybrid_dad_workspace', 'v6_can', 'v6_square'):
            with self.subTest(config=name):
                cfg = OmegaConf.load(root / f'{name}.yaml')
                self.assertTrue(cfg.checkpoint.save_last_ckpt)
                self.assertFalse(cfg.checkpoint.save_last_snapshot)
                self.assertEqual(cfg.checkpoint.topk.k, 0)


if __name__ == '__main__':
    unittest.main()
