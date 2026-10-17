import copy
from types import SimpleNamespace
import unittest
import sys
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from star.rl.algos.stage2 import STARStage2, RandomRotateShiftAug
from star.rl.algos.factory import algorithm
from star.rl.arguments import parse_args
from star.environments.geometric.rotations import build_camera_views


class CoreTests(unittest.TestCase):
    def test_robosuite_keyword_argument_order(self):
        from star.evaluation.rl import make_robosuite
        from unittest.mock import Mock
        factory = Mock(return_value='environment')
        module = SimpleNamespace(robo_make=factory)
        args = SimpleNamespace(task='Door', action_repeat=1, frame_stack=3, seed=1, mode='medium')
        with patch.dict(sys.modules, {'star.rl.wrappers.robo_wrapper': module}):
            self.assertEqual(make_robosuite(args, 16), 'environment')
        factory.assert_called_once_with(name='Door', action_repeat=1, frame_stack=3,
                                        seed=17, mode='eval-medium', scene_id=6)

    def test_close_reaches_simulator(self):
        from star.evaluation.rl import close_environment
        from unittest.mock import Mock
        simulator = Mock()
        env = SimpleNamespace(_env=SimpleNamespace(_gym_env=simulator))
        close_environment(env)
        simulator.close.assert_called_once_with()

    def test_only_star_agents(self):
        self.assertEqual(set(algorithm), {'dad', 'dad2', 'star_stage1', 'star_stage2'})

    def test_geometric_training_argument(self):
        args = parse_args(['--algo', 'star_stage2', '--overlay_type', 'random_rotate_shift'])
        self.assertEqual(args.algo, 'dad2')
        self.assertEqual(args.overlay_type, 'random_rotate_shift')

    def test_transform_shape_and_seed(self):
        transform = RandomRotateShiftAug()
        images = torch.arange(2 * 9 * 84 * 84).reshape(2, 9, 84, 84).float()
        torch.manual_seed(3)
        first = transform(images)
        torch.manual_seed(3)
        second = transform(images)
        self.assertEqual(first.shape, images.shape)
        self.assertTrue(torch.equal(first, second))

    def test_feature_loss_and_parameter_freezing(self):
        torch.set_num_threads(1)
        for frozen in (False, True):
            for clean in (False, True):
                torch.manual_seed(71)
                a = STARStage2.__new__(STARStage2)
                a.args = SimpleNamespace(gen_type='mlp', max_alpha=1., weak_aug=True,
                    overlay_type='random', freeze_aug_encoder_mlp=frozen,
                    clean_loss=clean, clean_loss_weight=.5)
                a.aug = nn.Identity()
                reference = nn.Linear(3, 3).requires_grad_(False)
                a.teacher_agent = SimpleNamespace(critic=SimpleNamespace(
                    encoder=SimpleNamespace(shared_cnn=reference)))
                a.aug_encoder, a.adapter = nn.Linear(3, 3), nn.Linear(3, 3)
                a.aug_encoder_opt = torch.optim.Adam(a.aug_encoder.parameters(), lr=1e-3)
                a.adapter_opt = torch.optim.Adam(a.adapter.parameters(), lr=1e-3)
                x = torch.randn(8, 3)
                before = copy.deepcopy(a.aug_encoder.state_dict())
                with torch.no_grad():
                    loss = nn.functional.mse_loss(a.adapter(a.aug_encoder(x + .3)), reference(x))
                    if clean:
                        loss += .5 * nn.functional.mse_loss(a.adapter(a.aug_encoder(x)), reference(x))
                batch = (x, None, None, None, x, None)
                with patch('star.rl.algos.stage2.utils.to_torch', side_effect=lambda x, d: x), \
                     patch('star.rl.algos.stage2.augmentations.random_ratio_overlay', side_effect=lambda x, **kw: x + .3):
                    result = a.update(iter([batch]), 1)
                self.assertEqual(result['G/recon_loss'], loss.item())
                changed = any(not torch.equal(v, a.aug_encoder.state_dict()[k]) for k, v in before.items())
                self.assertEqual(changed, not frozen)
                self.assertTrue(all(p.grad is None for p in reference.parameters()))

    def test_legacy_state_keeps_geometric_transform(self):
        a = STARStage2.__new__(STARStage2)
        transform = RandomRotateShiftAug()
        encoder, adapter = nn.Linear(3, 3), nn.Linear(3, 3)
        a.__setstate__({'args': SimpleNamespace(gen_type='mlp'), 'aug_encoder': encoder,
                        'generator_diffusion': adapter, 'random_rotate_shift_aug': transform})
        self.assertIs(a.adapter, adapter)
        self.assertIs(a.random_rotate_shift_aug, transform)


if __name__ == '__main__':
    unittest.main()
