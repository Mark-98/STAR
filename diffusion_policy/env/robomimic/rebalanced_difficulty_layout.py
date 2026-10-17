"""Compatibility adapter for immutable public scenes, without result directories.

Published evaluations use STAR's frozen registry. The ordinary upstream training
layout remains available; obsolete outcome-generated layout names fail explicitly.
"""
from star.environments.scenes import load_scenes

DEFAULT_LAYOUT_NAME = 'default'
REBALANCED_LAYOUT_NAME = 'star_v1'
SUPPORTED_LAYOUT_NAMES = {DEFAULT_LAYOUT_NAME, REBALANCED_LAYOUT_NAME}
_TASKS = {'PickPlaceCan': 'can', 'NutAssemblySquare': 'square'}


def has_rebalanced_layout(task_name, layout_name):
    return layout_name == REBALANCED_LAYOUT_NAME and task_name in _TASKS


def build_rebalanced_scene_records(task_name, visual_mode, layout_name):
    if layout_name == DEFAULT_LAYOUT_NAME:
        return None
    if not has_rebalanced_layout(task_name, layout_name):
        raise ValueError(f'Unsupported frozen layout {layout_name!r} for {task_name!r}.')
    return load_scenes(_TASKS[task_name], visual_mode.replace('eval-', ''))
