import copy
import json
import unittest
from star.environments.scenes import assign_scene_episodes, get_scene, load_scenes, visual_digest
from star.evaluation.diffusion import build_config
from star.utils.paths import resource


class FrozenScenes(unittest.TestCase):
    def test_contiguous_ids_and_original_visual_hashes(self):
        provenance = json.loads(resource('docs', 'scene_provenance.json').read_text())
        for task in ('can', 'square'):
            records = load_scenes(task)
            self.assertEqual([r['scene_id'] for r in records], list(range(30)))
            for i, level in enumerate(('easy', 'medium', 'hard')):
                self.assertEqual([r['scene_id'] for r in load_scenes(task, level)], list(range(i*10, i*10+10)))
            for record, source in zip(records, provenance[task]['mapping']):
                self.assertEqual(visual_digest(record), source['visual_sha256'])

    def test_no_mutation_or_silent_fallback(self):
        before = get_scene('can', 0)
        get_scene('can', 0)['texture_map']['floor'] = 'CHANGED'
        self.assertEqual(before, get_scene('can', 0))
        for invalid in (-1, 30, 0.5, True):
            with self.assertRaises(ValueError):
                get_scene('can', invalid)
        with self.assertRaises(ValueError):
            load_scenes('unknown')

    def test_protocol_defaults(self):
        cfg = build_config('square', 'datasets/square/ph/image_abs.hdf5', 'base.ckpt', 'ours.ckpt', 0)
        self.assertEqual(cfg['evaluation']['seeds'], [42] * 30)
        self.assertTrue(cfg['evaluation']['exact_scene_config'])
        for level in ('easy', 'medium', 'hard'):
            self.assertEqual(len(cfg['robustness_eval'][level]['scene_ids']), 10)
        self.assertEqual(cfg['robustness_eval']['hard']['scene_ids'], list(range(20, 30)))

    def test_episode_cycle_preserves_visuals(self):
        originals = load_scenes('can')
        episodes = assign_scene_episodes(originals, 12, 'cycle')
        self.assertEqual(len(episodes), 36)
        for index in range(3):
            group = episodes[index * 12:(index + 1) * 12]
            self.assertEqual([r['scene_id'] for r in group],
                             [index * 10 + i % 10 for i in range(12)])
            self.assertEqual([r['rollout_index'] for r in group], list(range(12)))
            self.assertEqual(visual_digest(group[0]), visual_digest(group[10]))
        episodes[0]['texture_map']['floor'] = 'changed'
        self.assertEqual(originals, load_scenes('can'))

    def test_invalid_assignment_is_rejected(self):
        for count, assignment in ((0, 'cycle'), (12, 'exact'), (10, 'random')):
            with self.assertRaises(ValueError):
                assign_scene_episodes(load_scenes('can'), count, assignment)

    def test_exact_assignment(self):
        original = load_scenes('square')
        self.assertEqual(assign_scene_episodes(original, 10, 'exact'), original)


if __name__ == '__main__':
    unittest.main()
