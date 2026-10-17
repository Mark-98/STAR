"""Regression checks for runtime files previously hidden by broad ignore rules."""
from pathlib import Path
import unittest


class DistributionTests(unittest.TestCase):
    def test_required_runtime_sources_exist(self):
        root = Path(__file__).resolve().parents[1]
        for relative in ('star/datasets/__init__.py', 'star/datasets/replay.py',
                         'vendor/dm_control/dm_control/mujoco/wrapper/core.py',
                         'vendor/robosuitevgb/robosuitevgb/secant/envs/robosuite/core.py'):
            with self.subTest(path=relative):
                self.assertTrue((root / relative).is_file(), relative)

    def test_replay_helpers_import(self):
        from star.datasets.replay import stage_replay, validate_full_replay
        self.assertTrue(callable(stage_replay))
        self.assertTrue(callable(validate_full_replay))

    def test_hard_mode_requires_video_even_when_yaml_disables_it(self):
        from unittest.mock import patch
        import robosuitevgb.utils as factory
        with patch.object(factory.suite, 'make'), patch.object(factory, 'VGBWrapper') as wrapper:
            factory.make_env_eval_hard(task_name='Door', seed=1)
        self.assertTrue(wrapper.call_args.kwargs['video_background'])
        self.assertEqual(wrapper.call_args.kwargs['mode'], 'eval-hard')
