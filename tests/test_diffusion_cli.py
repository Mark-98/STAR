"""Public DP interface is usable without installing the optional DP stack."""
import contextlib
import io
import json
import unittest
from star.evaluation.diffusion import main, parse_args


class DiffusionCLI(unittest.TestCase):
    required = ['--task', 'can', '--dataset', 'data.hdf5',
                '--base-checkpoint', 'base.ckpt', '--ours-checkpoint', 'ours.ckpt',
                '--output-dir', 'outputs/test']

    def test_episode_flag_and_legacy_alias(self):
        for flag in ('--episodes', '--num-episodes'):
            self.assertEqual(parse_args(self.required + [flag, '12']).num_episodes, 12)

    def test_no_manual_scene_flag(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            parse_args(self.required + ['--scene-id', '3'])

    def test_invalid_counts(self):
        for flags in (['--episodes', '0'], ['--episodes', '-1'],
                      ['--n-envs', '0'], ['--demo-start', '-1']):
            with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parse_args(self.required + flags)

    def test_dry_run_counts_all_conditions_without_loading_weights(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(self.required + ['--episodes', '3', '--dry-run'])
        plan = json.loads(output.getvalue())
        self.assertEqual(plan['total_episodes'], 24)
        self.assertEqual(plan['conditions'], ['clean', 'easy', 'medium', 'hard'])
        self.assertNotIn('scene_ids', output.getvalue())

    def test_visual_only_count(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            main(self.required + ['--episodes', '2', '--phase', 'visual', '--dry-run'])
        self.assertEqual(json.loads(output.getvalue())['total_episodes'], 12)
