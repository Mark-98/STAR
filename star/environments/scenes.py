"""Frozen v6 visual definitions with contiguous public scene IDs.

No simulator imports, random draws, checkpoint access, or score-based selection.
"""
import argparse
import copy
import hashlib
import json
from star.utils.paths import resource

TASKS = ('can', 'square')
LEVELS = ('easy', 'medium', 'hard')
VISUAL_KEYS = ('texture_map', 'material_map', 'skybox_config', 'light_config')


def load_scenes(task, difficulty=None):
    if task not in TASKS:
        raise ValueError(f'Unknown task {task!r}; choose {TASKS}.')
    if difficulty is not None and difficulty not in LEVELS:
        raise ValueError(f'Unknown difficulty {difficulty!r}.')
    data = json.loads(resource('configs', 'scenes', task + '.json').read_text())
    records = data['records']
    if [r['scene_id'] for r in records] != list(range(30)):
        raise ValueError('Frozen scene IDs must be exactly 0..29.')
    return copy.deepcopy([r for r in records if difficulty is None or r['difficulty'] == difficulty])


def get_scene(task, scene_id):
    if type(scene_id) is not int or not 0 <= scene_id < 30:
        raise ValueError('scene_id must be an integer in 0..29.')
    return load_scenes(task)[scene_id]


def visual_config(record):
    return {key: copy.deepcopy(record[key]) for key in VISUAL_KEYS}


def visual_digest(record):
    return hashlib.sha256(json.dumps(visual_config(record), sort_keys=True).encode()).hexdigest()


def assign_scene_episodes(specs, count, assignment='exact'):
    """Assign episode numbers without sampling or modifying the fixed visuals."""
    if count < 1 or assignment not in ('exact', 'cycle'):
        raise ValueError('Expected a positive count and exact/cycle scene assignment.')
    result = []
    for level in LEVELS:
        templates = [spec for spec in specs if spec['difficulty'] == level]
        if not templates or (assignment == 'exact' and len(templates) != count):
            raise ValueError(f'{level}: expected {count} scenes, found {len(templates)}; '
                             'use cycle explicitly to reuse configured scenes.')
        if assignment == 'exact':
            result.extend(templates)
            continue
        for episode in range(count):
            spec = copy.deepcopy(templates[episode % len(templates)])
            spec.update({'difficulty_episode': episode, 'rollout_index': episode,
                         'scene_episode': episode // len(templates), 'env_seed': episode})
            result.append(spec)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=TASKS, required=True)
    parser.add_argument('--scene-id', type=int)
    parser.add_argument('--difficulty', choices=LEVELS)
    args = parser.parse_args()
    print(json.dumps(get_scene(args.task, args.scene_id) if args.scene_id is not None
                     else load_scenes(args.task, args.difficulty), indent=2))


if __name__ == '__main__':
    main()
