import contextlib
import io
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from star.evaluation.rl import make_robosuite, parse_args


class EvaluationTests(unittest.TestCase):
    arguments = ['--env', 'robosuite', '--task', 'Door', '--checkpoint', 'model.pt',
                 '--output-dir', 'outputs/test', '--action-repeat', '1', '--mode', 'medium']

    def test_episode_count_without_scene_argument(self):
        args = parse_args(self.arguments + ['--episodes', '12'])
        self.assertEqual(args.episodes, 12)
        self.assertFalse(hasattr(args, 'scene_id'))
        self.assertEqual(parse_args(self.arguments).episodes, 30)

    def test_invalid_episode_count_rejected(self):
        for count in ('0', '-1'):
            with self.subTest(count=count), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args(self.arguments + ['--episodes', count])
                self.assertEqual(error.exception.code, 2)

    def test_manual_scene_option_removed(self):
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parse_args(self.arguments + ['--scene-id', '6'])
        self.assertEqual(error.exception.code, 2)

    def test_automatic_scene_cycle_and_seeds(self):
        args = parse_args(self.arguments + ['--episodes', '25', '--seed', '400'])
        factory = Mock()
        with patch.dict(sys.modules, {'star.rl.wrappers.robo_wrapper': SimpleNamespace(robo_make=factory)}):
            for episode in range(args.episodes):
                make_robosuite(args, episode)
        self.assertEqual([c.kwargs['scene_id'] for c in factory.call_args_list],
                         list(range(10)) * 2 + list(range(5)))
        self.assertEqual([c.kwargs['seed'] for c in factory.call_args_list], list(range(400, 425)))
