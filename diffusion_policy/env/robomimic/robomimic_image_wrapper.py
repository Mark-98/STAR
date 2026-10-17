from typing import Any, Dict, List, Optional
import pathlib
import re
import numpy as np
import gym
from gym import spaces
from omegaconf import OmegaConf
from robomimic.envs.env_robosuite import EnvRobosuite
from diffusion_policy.env.robomimic.robomimic_reward_mixin import RobomimicRewardMixin
import xml.etree.ElementTree as ET

from robosuite.utils.mjcf_utils import TEXTURES, xml_path_completion


ROBOMIMIC_TEXTURE_ALIAS = {
    'floor': 'floorplane',
    'wall': 'walls_mat',
    'table_legs': 'table_legs_metal',
    'table': 'table_ceramic',
    'door': 'Door_MatDarkWood',
    'door_handle': 'Door_MatMetal',
    'lift_object': 'cube_redwood_mat',
    'na_metal1': 'smetal',
    'na_metal2': 'bmetal',
    'pp_table1': 'light-wood',
    'pp_table2': 'dark-wood',
    'stack_object1': 'greenwood_mat',
    'stack_object2': 'redwood_mat',
    'handoff_hammer_head': 'metal_mat',
    'handoff_hammer_body': 'wood_mat',
    'ta_lift_pot': 'pot_mat',
    'ta_lift_handle1': 'handle1_mat',
    'ta_lift_handle2': 'handle2_mat',
    'ta_pih_plate': 'plate_mat',
    'ta_pih_stick': 'greenwood_mat',
}

ROBOMIMIC_TEXTURE_TYPES = {
    'floor': '2d',
    'wall': '2d',
    'table_legs': 'cube',
    'table': '2d',
    'door': 'cube',
    'door_handle': 'cube',
    'lift_object': 'cube',
    'na_metal1': 'cube',
    'na_metal2': 'cube',
    'pp_table1': '2d',
    'pp_table2': '2d',
    'stack_object1': 'cube',
    'stack_object2': 'cube',
    'handoff_hammer_head': 'cube',
    'handoff_hammer_body': 'cube',
    'ta_lift_pot': 'cube',
    'ta_lift_handle1': 'cube',
    'ta_lift_handle2': 'cube',
    'ta_pih_plate': 'cube',
    'ta_pih_stick': 'cube',
}

ROBOMIMIC_TASK_TEXTURE_LIST = {
    'Door': ['floor', 'wall', 'table_legs', 'table', 'door', 'door_handle'],
    'Lift': ['floor', 'wall', 'table_legs', 'table', 'lift_object'],
    'NutAssembly': ['floor', 'wall', 'table_legs', 'table', 'na_metal1', 'na_metal2'],
    'NutAssemblyRound': ['floor', 'wall', 'table_legs', 'table', 'na_metal1', 'na_metal2'],
    'NutAssemblySingle': ['floor', 'wall', 'table_legs', 'table', 'na_metal1', 'na_metal2'],
    'NutAssemblySquare': ['floor', 'wall', 'table_legs', 'table', 'na_metal1', 'na_metal2'],
    'PickPlace': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceBread': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceCan': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceCereal': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceMilk': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceSingle': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'Stack': ['floor', 'wall', 'table_legs', 'table', 'stack_object1', 'stack_object2'],
    'TwoArmHandover': ['floor', 'wall', 'table_legs', 'table', 'handoff_hammer_head', 'handoff_hammer_body'],
    'TwoArmLift': ['floor', 'wall', 'table_legs', 'table', 'ta_lift_pot', 'ta_lift_handle1', 'ta_lift_handle2'],
    'TwoArmPegInHole': ['floor', 'wall', 'ta_pih_plate', 'ta_pih_stick'],
    'TwoArmTransport': ['floor', 'wall', 'table_legs', 'table'],
    'ToolHang': ['floor', 'wall', 'table_legs', 'table'],
    'Wipe': ['floor', 'wall', 'table_legs', 'table'],
}

ROBOMIMIC_TRAIN_BACKGROUND_ONLY_TEXTURE_LIST = {
    'Door': ['floor', 'wall', 'table_legs', 'table'],
    'Lift': ['floor', 'wall', 'table_legs', 'table'],
    'PickPlace': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceBread': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceCan': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceCereal': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceMilk': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'PickPlaceSingle': ['floor', 'wall', 'table_legs', 'pp_table1', 'pp_table2'],
    'NutAssembly': ['floor', 'wall', 'table_legs', 'table'],
    'NutAssemblyRound': ['floor', 'wall', 'table_legs', 'table'],
    'NutAssemblySingle': ['floor', 'wall', 'table_legs', 'table'],
    'NutAssemblySquare': ['floor', 'wall', 'table_legs', 'table'],
    'Stack': ['floor', 'wall', 'table_legs', 'table'],
    'TwoArmHandover': ['floor', 'wall', 'table_legs', 'table'],
    'TwoArmLift': ['floor', 'wall', 'table_legs', 'table'],
    'TwoArmPegInHole': ['floor', 'wall'],
    'TwoArmTransport': ['floor', 'wall', 'table_legs', 'table'],
    'ToolHang': ['floor', 'wall', 'table_legs', 'table'],
    'Wipe': ['floor', 'wall', 'table_legs', 'table'],
}

# Difficulty policy:
# - train: background-only, near in-domain
# - eval-easy: background-only, mild OOD
# - eval-medium: full-scene texture changes, moderate OOD
# - eval-hard: full-scene texture changes, strongly OOD / custom-heavy
ROBOMIMIC_BACKGROUND_ONLY_VISUAL_MODES = {'train', 'eval-easy'}

ROBOMIMIC_TEXTURE_PRESET = {
    'floor': ['WoodTiles', 'WoodPanels', 'WoodDark', 'WoodLight', 'WoodgrainGray', 'FeltGray', 'Dirt'],
    'wall': ['PlasterCream', 'PlasterPink', 'PlasterYellow', 'PlasterGray', 'PlasterWhite', 'BricksWhite', 'Clay'],
    'table_legs': ['Metal', 'SteelBrushed', 'SteelScratched'],
    'table': ['Ceramic', 'PlasterCream', 'PlasterYellow', 'PlasterGray', 'WoodDark', 'WoodLight'],
    'door': ['WoodDark', 'WoodLight', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'door_handle': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
    'lift_object': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'na_metal1': ['Brass', 'SteelBrushed', 'Metal', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'na_metal2': ['Brass', 'SteelBrushed', 'Metal', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'pp_table1': ['WoodDark', 'Ceramic', 'PlasterCream', 'PlasterYellow', 'PlasterGray', 'WoodLight', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'pp_table2': ['WoodDark', 'Ceramic', 'PlasterCream', 'PlasterYellow', 'PlasterGray', 'WoodLight', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'stack_object1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'stack_object2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'handoff_hammer_head': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
    'handoff_hammer_body': ['WoodDark', 'WoodLight', 'WoodGreen', 'WoodRed', 'WoodBlue'],
    'ta_lift_pot': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'ta_lift_handle1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'ta_lift_handle2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'ta_pih_plate': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    'ta_pih_stick': ['WoodGreen', 'WoodRed', 'WoodBlue'],
}

ROBOMIMIC_TEXTURE_DIFFICULTY_CANDIDATES = {
    'train': {
        'floor': ['WoodTiles', 'WoodPanels', 'WoodLight'],
        'wall': ['PlasterCream', 'PlasterGray', 'PlasterWhite'],
        'table_legs': ['Metal', 'SteelBrushed'],
        'table': ['Ceramic', 'PlasterCream', 'WoodLight'],
        'door': ['WoodDark', 'WoodLight'],
        'door_handle': ['Brass', 'Metal'],
        'lift_object': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'na_metal1': ['Brass', 'Metal'],
        'na_metal2': ['Brass', 'Metal'],
        'pp_table1': ['WoodLight', 'WoodDark', 'Ceramic'],
        'pp_table2': ['WoodDark', 'WoodLight', 'Ceramic'],
        'stack_object1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'stack_object2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'handoff_hammer_head': ['Metal', 'SteelBrushed'],
        'handoff_hammer_body': ['WoodDark', 'WoodLight'],
        'ta_lift_pot': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_lift_handle1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_lift_handle2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_pih_plate': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_pih_stick': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    },
    'eval-easy': {
        'floor': ['WoodTiles', 'WoodPanels', 'WoodLight', 'WoodgrainGray', 'FeltGray'],
        'wall': ['PlasterCream', 'PlasterGray', 'PlasterWhite', 'BricksWhite', 'Clay'],
        'table_legs': ['Metal', 'SteelBrushed'],
        'table': ['Ceramic', 'PlasterCream', 'WoodLight'],
        'door': ['WoodDark', 'WoodLight', 'WoodPanels'],
        'door_handle': ['Brass', 'Metal'],
        'lift_object': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'na_metal1': ['Brass', 'Metal', 'SteelBrushed'],
        'na_metal2': ['Brass', 'Metal', 'SteelBrushed'],
        'pp_table1': ['WoodLight', 'WoodDark', 'Ceramic', 'PlasterCream', 'PlasterGray'],
        'pp_table2': ['WoodDark', 'WoodLight', 'Ceramic', 'PlasterCream', 'PlasterGray'],
        'stack_object1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'stack_object2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'handoff_hammer_head': ['Metal', 'SteelBrushed'],
        'handoff_hammer_body': ['WoodDark', 'WoodLight'],
        'ta_lift_pot': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_lift_handle1': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_lift_handle2': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_pih_plate': ['WoodGreen', 'WoodRed', 'WoodBlue'],
        'ta_pih_stick': ['WoodGreen', 'WoodRed', 'WoodBlue'],
    },
    'eval-medium': {
        'floor': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'wall': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'table_legs': ['Metal', 'SteelBrushed', 'SteelScratched', 'Brass'],
        'table': ['PlasterGray', 'WoodDark','WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'door': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'door_handle': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
        'lift_object': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'na_metal1': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'na_metal2': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'pp_table1': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'pp_table2': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'stack_object1': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'stack_object2': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'handoff_hammer_head': ['Metal', 'SteelBrushed', 'SteelScratched', 'Brass'],
        'handoff_hammer_body': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'ta_lift_pot': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'ta_lift_handle1': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'ta_lift_handle2': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'ta_pih_plate': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        'ta_pih_stick': ['WoodGreen', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
    },
    'eval-hard': {
        'floor': [f'Custom{i:02d}' for i in range(11, 41)],
        'wall': [f'Custom{i:02d}' for i in range(11, 41)],
        'table_legs': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
        'table': [f'Custom{i:02d}' for i in range(11, 41)],
        'door': [f'Custom{i:02d}' for i in range(11, 41)],
        'door_handle': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
        'lift_object': [f'Custom{i:02d}' for i in range(11, 41)],
        'na_metal1': [f'Custom{i:02d}' for i in range(11, 41)],
        'na_metal2': [f'Custom{i:02d}' for i in range(11, 41)],
        'pp_table1': [f'Custom{i:02d}' for i in range(11, 41)],
        'pp_table2': [f'Custom{i:02d}' for i in range(11, 41)],
        'stack_object1': [f'Custom{i:02d}' for i in range(11, 41)],
        'stack_object2': [f'Custom{i:02d}' for i in range(11, 41)],
        'handoff_hammer_head': ['Brass', 'Metal', 'SteelBrushed', 'SteelScratched'],
        'handoff_hammer_body': [f'Custom{i:02d}' for i in range(11, 41)],
        'ta_lift_pot': [f'Custom{i:02d}' for i in range(11, 41)],
        'ta_lift_handle1': [f'Custom{i:02d}' for i in range(11, 41)],
        'ta_lift_handle2': [f'Custom{i:02d}' for i in range(11, 41)],
        'ta_pih_plate': [f'Custom{i:02d}' for i in range(11, 41)],
        'ta_pih_stick': [f'Custom{i:02d}' for i in range(11, 41)],
    },
}

# Task-specific nudges to make selected eval buckets slightly harder without
# shifting the default texture policy for every robomimic task.
ROBOMIMIC_TASK_MODE_TEXTURE_OVERRIDES = {
    'NutAssemblySquare': {
        'eval-easy': {
            'floor': ['WoodTiles', 'WoodPanels', 'WoodLight', 'WoodgrainGray', 'FeltGray', 'Dirt'],
            'table_legs': ['Metal', 'SteelBrushed', 'SteelScratched'],
            'table': ['Ceramic', 'PlasterCream', 'WoodLight', 'WoodDark'],
        },
        'eval-medium': {
            'floor': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
            'wall': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
            'table_legs': ['Metal', 'SteelBrushed', 'SteelBrushed', 'SteelScratched', 'Brass'],
            'table': ['WoodDark', 'WoodGreen', 'WoodRed', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        },
    },
    'PickPlaceCan': {
        'eval-easy': {
            'floor': ['WoodTiles', 'WoodPanels', 'WoodLight', 'WoodgrainGray', 'FeltGray', 'Dirt'],
            'table_legs': ['Metal', 'SteelBrushed', 'SteelScratched'],
            'pp_table1': ['WoodLight', 'WoodDark', 'Ceramic', 'PlasterCream', 'PlasterGray', 'WoodRed'],
            'pp_table2': ['WoodDark', 'WoodLight', 'Ceramic', 'PlasterCream', 'PlasterGray', 'WoodRed'],
        },
        'eval-medium': {
            'floor': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
            'wall': ['WoodGreen', 'WoodRed', 'WoodBlue', 'WoodgrainGray', 'FeltGray', 'Dirt', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
            'table_legs': ['Metal', 'SteelBrushed', 'SteelBrushed', 'SteelScratched', 'Brass'],
            'pp_table1': ['WoodGreen', 'WoodRed', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
            'pp_table2': ['WoodGreen', 'WoodRed', 'WoodRed', 'WoodBlue', 'Custom01', 'Custom02', 'Custom03', 'Custom04', 'Custom05'],
        },
    },
}

ROBOMIMIC_DIFFICULTY_SCENE_SEEDS = {
    'train': list(range(20)),
    'eval-easy': [105, 106, 113, 114, 120, 130, 132, 163, 224, 281],
    'eval-medium': [16, 22, 102, 171, 190, 235, 309, 338, 343, 369],
    'eval-hard': [26, 29, 31, 100, 112, 148, 229, 275, 291, 369],
}

ROBOMIMIC_TEXTURE_DIFF_CONSTRAINTS = [
    ['na_metal1', 'na_metal2'],
    ['pp_table1', 'pp_table2'],
    ['stack_object1', 'stack_object2'],
    ['ta_lift_pot', 'ta_lift_handle1', 'ta_lift_handle2'],
    ['ta_pih_plate', 'ta_pih_stick'],
]

ROBOMIMIC_MATERIAL_PRESET = {
    'floor': [
        {'shininess': 0.05, 'specular': 0.02, 'reflectance': 0.02},
        {'shininess': 0.1, 'specular': 0.05, 'reflectance': 0.05},
    ],
    'wall': [
        {'shininess': 0.02, 'specular': 0.01, 'reflectance': 0.0},
        {'shininess': 0.08, 'specular': 0.03, 'reflectance': 0.02},
    ],
    'table_legs': [
        {'shininess': 0.7, 'specular': 0.6, 'reflectance': 0.15},
        {'shininess': 0.9, 'specular': 0.8, 'reflectance': 0.25},
    ],
    'table': [
        {'shininess': 0.15, 'specular': 0.08, 'reflectance': 0.02},
        {'shininess': 0.3, 'specular': 0.15, 'reflectance': 0.05},
    ],
    'door': [
        {'shininess': 0.12, 'specular': 0.08, 'reflectance': 0.03},
        {'shininess': 0.22, 'specular': 0.12, 'reflectance': 0.06},
    ],
    'door_handle': [
        {'shininess': 0.8, 'specular': 0.75, 'reflectance': 0.2},
        {'shininess': 0.95, 'specular': 0.9, 'reflectance': 0.3},
    ],
    'lift_object': [
        {'shininess': 0.18, 'specular': 0.1, 'reflectance': 0.03},
        {'shininess': 0.28, 'specular': 0.15, 'reflectance': 0.06},
    ],
}

ROBOMIMIC_SKYBOX_PRESET = [
    {'builtin_texture': 'WoodDark', 'rgb1': [0.85, 0.9, 0.95], 'rgb2': [0.55, 0.65, 0.78]},
    {'builtin_texture': 'PlasterGray', 'rgb1': [0.95, 0.9, 0.82], 'rgb2': [0.68, 0.58, 0.5]},
    {'builtin_texture': 'WoodLight', 'rgb1': [0.82, 0.9, 0.82], 'rgb2': [0.42, 0.55, 0.46]},
]

ROBOMIMIC_LIGHT_PRESET = [
    {
        'ambient': [0.35, 0.35, 0.35],
        'diffuse': [0.75, 0.75, 0.75],
        'specular': [0.25, 0.25, 0.25],
        'castshadow': False,
    },
    {
        'ambient': [0.22, 0.22, 0.22],
        'diffuse': [0.95, 0.88, 0.8],
        'specular': [0.4, 0.35, 0.3],
        'castshadow': True,
    },
    {
        'ambient': [0.18, 0.2, 0.24],
        'diffuse': [0.72, 0.8, 0.95],
        'specular': [0.2, 0.25, 0.35],
        'castshadow': True,
    },
]

_CUSTOM_TEXTURE_PATTERN = re.compile(r'^Custom(\d{2})$')
from star.utils.paths import texture_root


def unwrap_robosuite_env(env):
    current = env
    visited = set()
    while hasattr(current, 'env') and id(current) not in visited:
        visited.add(id(current))
        next_env = getattr(current, 'env')
        if next_env is None:
            break
        current = next_env
    return current


def get_robomimic_task_name(env, fallback: Optional[str] = None) -> Optional[str]:
    candidates = [env, unwrap_robosuite_env(env)]
    for candidate in candidates:
        for attr in ('name', 'env_name'):
            value = getattr(candidate, attr, None)
            if isinstance(value, str) and len(value) > 0:
                return value
    return fallback


def get_robomimic_env_xml(env) -> str:
    base_env = unwrap_robosuite_env(env)
    root = getattr(getattr(base_env, 'model', None), 'root', None)
    if root is None:
        raise RuntimeError('Unable to access MuJoCo XML root from the given env.')
    return ET.tostring(root, encoding='unicode')


def resolve_texture_candidates_for_mode(
    task_name: str,
    visual_mode: Optional[str] = None,
) -> Dict[str, List[str]]:
    if visual_mode in ROBOMIMIC_BACKGROUND_ONLY_VISUAL_MODES:
        aliases = ROBOMIMIC_TRAIN_BACKGROUND_ONLY_TEXTURE_LIST.get(
            task_name,
            ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, []),
        )
    else:
        aliases = ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, [])
    if visual_mode is None:
        return {
            alias: list(ROBOMIMIC_TEXTURE_PRESET.get(alias, []))
            for alias in aliases
            if len(ROBOMIMIC_TEXTURE_PRESET.get(alias, [])) > 0
        }

    if visual_mode not in ROBOMIMIC_TEXTURE_DIFFICULTY_CANDIDATES:
        raise ValueError(f'Unsupported visual_mode: {visual_mode}')

    mode_candidates = ROBOMIMIC_TEXTURE_DIFFICULTY_CANDIDATES[visual_mode]
    fallback_candidates = ROBOMIMIC_TEXTURE_PRESET
    task_overrides = (
        ROBOMIMIC_TASK_MODE_TEXTURE_OVERRIDES
        .get(task_name, {})
        .get(visual_mode, {})
    )
    resolved = dict()
    for alias in aliases:
        candidates = task_overrides.get(alias)
        if candidates is None:
            candidates = mode_candidates.get(alias)
        if candidates is None:
            candidates = fallback_candidates.get(alias, [])
        if len(candidates) > 0:
            resolved[alias] = list(candidates)
    return resolved


def get_visual_mode_seed(
    visual_mode: Optional[str] = None,
    scene_id: Optional[int] = None,
    seed: Optional[int] = None,
) -> Optional[int]:
    if seed is not None:
        return int(seed)
    if visual_mode is None:
        return None
    scene_seeds = ROBOMIMIC_DIFFICULTY_SCENE_SEEDS.get(visual_mode)
    if scene_seeds is None:
        raise ValueError(f'Unsupported visual_mode: {visual_mode}')
    if scene_id is None:
        scene_id = 0
    scene_count = len(scene_seeds)
    cycle_index, scene_index = divmod(int(scene_id), scene_count)
    base_seed = int(scene_seeds[scene_index])
    if cycle_index == 0:
        return base_seed
    # Extend scene sweeps beyond the preset table without repeating identical
    # visual seeds. This keeps the first cycle unchanged while making
    # `scene_id=10..29` deterministic and distinct for post-train evaluation.
    return base_seed + cycle_index * 1000


def sample_task_texture_map(
    task_name: str,
    seed: Optional[int] = None,
    texture_candidates: Optional[Dict[str, List[str]]] = None,
    visual_mode: Optional[str] = None,
    scene_id: Optional[int] = None,
) -> Dict[str, str]:
    resolved_seed = get_visual_mode_seed(
        visual_mode=visual_mode,
        scene_id=scene_id,
        seed=seed,
    )
    rng = np.random.RandomState(resolved_seed)
    texture_candidates = texture_candidates or resolve_texture_candidates_for_mode(
        task_name=task_name,
        visual_mode=visual_mode,
    )
    if visual_mode in ROBOMIMIC_BACKGROUND_ONLY_VISUAL_MODES:
        aliases = ROBOMIMIC_TRAIN_BACKGROUND_ONLY_TEXTURE_LIST.get(
            task_name,
            ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, []),
        )
    else:
        aliases = ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, [])
    result = dict()
    for alias in aliases:
        candidates = texture_candidates.get(alias, [])
        if len(candidates) == 0:
            continue
        result[alias] = str(rng.choice(candidates))

    for alias_group in ROBOMIMIC_TEXTURE_DIFF_CONSTRAINTS:
        chosen = set()
        for alias in alias_group:
            if alias not in result:
                continue
            candidates = list(texture_candidates.get(alias, []))
            if len(candidates) == 0:
                continue
            unique_candidates = [candidate for candidate in candidates if candidate not in chosen]
            if len(unique_candidates) == 0:
                unique_candidates = candidates
            result[alias] = str(rng.choice(unique_candidates))
            chosen.add(result[alias])
    return result


def sample_task_material_map(
    task_name: str,
    seed: Optional[int] = None,
    material_candidates: Optional[Dict[str, List[Dict[str, Any]]]] = None
) -> Dict[str, Dict[str, Any]]:
    rng = np.random.RandomState(seed)
    material_candidates = material_candidates or ROBOMIMIC_MATERIAL_PRESET
    aliases = ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, [])
    result = dict()
    for alias in aliases:
        candidates = material_candidates.get(alias, [])
        if len(candidates) == 0:
            continue
        idx = int(rng.randint(len(candidates)))
        result[alias] = dict(candidates[idx])
    return result


def sample_skybox_config(seed: Optional[int] = None) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    idx = int(rng.randint(len(ROBOMIMIC_SKYBOX_PRESET)))
    return dict(ROBOMIMIC_SKYBOX_PRESET[idx])


def sample_light_config(seed: Optional[int] = None) -> Dict[str, Any]:
    rng = np.random.RandomState(seed)
    idx = int(rng.randint(len(ROBOMIMIC_LIGHT_PRESET)))
    return dict(ROBOMIMIC_LIGHT_PRESET[idx])


def sample_task_visual_config(
    task_name: str,
    seed: Optional[int] = None,
    visual_mode: Optional[str] = None,
    scene_id: Optional[int] = None,
) -> Dict[str, Any]:
    base_seed = get_visual_mode_seed(
        visual_mode=visual_mode,
        scene_id=scene_id,
        seed=seed,
    )
    if base_seed is None:
        base_seed = 0
    if visual_mode in ROBOMIMIC_BACKGROUND_ONLY_VISUAL_MODES:
        return {
            'texture_map': sample_task_texture_map(
                task_name=task_name,
                seed=base_seed,
                visual_mode=visual_mode,
                scene_id=scene_id,
            ),
            'material_map': None,
            'skybox_config': None,
            'light_config': None,
        }
    return {
        'texture_map': sample_task_texture_map(
            task_name=task_name,
            seed=base_seed,
            visual_mode=visual_mode,
            scene_id=scene_id,
        ),
        'material_map': sample_task_material_map(task_name=task_name, seed=base_seed + 1),
        'skybox_config': sample_skybox_config(seed=base_seed + 2),
        'light_config': None,
    }


def _to_mjcf_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (list, tuple, np.ndarray)):
        return ' '.join(str(float(x)) for x in value)
    return str(value)


def _find_asset_element(root: ET.Element, tag: str, name: str) -> Optional[ET.Element]:
    for elem in root.iter(tag):
        if elem.get('name') == name:
            return elem
    return None


def _ensure_asset_parent(root: ET.Element) -> ET.Element:
    asset = root.find('asset')
    if asset is None:
        asset = ET.SubElement(root, 'asset')
    return asset


def _ensure_texture_asset(
    root: ET.Element,
    texture_name: str,
    texture_type: str,
    texture_attributes: Optional[Dict[str, Any]] = None,
) -> str:
    asset = _ensure_asset_parent(root)
    existing = _find_asset_element(root, 'texture', texture_name)
    if existing is not None:
        if texture_attributes is not None:
            for key, value in texture_attributes.items():
                existing.set(key, _to_mjcf_value(value))
        return texture_name

    custom_match = _CUSTOM_TEXTURE_PATTERN.match(texture_name)
    if custom_match is not None:
        custom_index = custom_match.group(1).lower()
        texture_file = str(texture_root() / f'custom{custom_index}.png')
        if not pathlib.Path(texture_file).is_file():
            raise FileNotFoundError(f'Missing frozen texture {texture_file}; see assets/README.md and STAR_TEXTURE_ROOT.')
    elif texture_name in TEXTURES:
        texture_file = xml_path_completion('textures/' + TEXTURES[texture_name])
    else:
        texture_file = texture_name

    texture_attrib = {
        'name': texture_name,
        'type': texture_type,
        'file': texture_file,
    }
    if texture_attributes is not None:
        for key, value in texture_attributes.items():
            texture_attrib[key] = _to_mjcf_value(value)
    asset.append(ET.Element('texture', texture_attrib))
    return texture_name


def _apply_material_attributes(
    material_elem: ET.Element,
    material_attributes: Optional[Dict[str, Any]]
):
    if material_attributes is None:
        return
    for key, value in material_attributes.items():
        material_elem.set(key, _to_mjcf_value(value))


def _apply_skybox_config(root: ET.Element, skybox_config: Optional[Dict[str, Any]]):
    if skybox_config is None:
        return

    skybox_texture = None
    for texture_elem in root.iter('texture'):
        if texture_elem.get('type') == 'skybox':
            skybox_texture = texture_elem
            break

    if skybox_texture is None:
        asset = _ensure_asset_parent(root)
        skybox_texture = ET.SubElement(asset, 'texture', {'type': 'skybox', 'name': 'auto_skybox'})

    builtin_texture = skybox_config.get('builtin_texture')
    if builtin_texture in TEXTURES:
        skybox_texture.set('file', xml_path_completion('textures/' + TEXTURES[builtin_texture]))
        skybox_texture.set('builtin', 'none')
    elif builtin_texture is not None:
        skybox_texture.set('file', _to_mjcf_value(builtin_texture))

    for key in ('rgb1', 'rgb2', 'markrgb', 'random', 'width', 'height'):
        if key in skybox_config:
            skybox_texture.set(key, _to_mjcf_value(skybox_config[key]))


def _apply_light_config(root: ET.Element, light_config: Optional[Dict[str, Any]]):
    if light_config is None:
        return
    for light_elem in root.iter('light'):
        for key, value in light_config.items():
            light_elem.set(key, _to_mjcf_value(value))


def build_visual_mod_xml(
    xml_string: str,
    texture_map: Dict[str, str],
    task_name: Optional[str] = None,
    material_map: Optional[Dict[str, Dict[str, Any]]] = None,
    skybox_config: Optional[Dict[str, Any]] = None,
    light_config: Optional[Dict[str, Any]] = None,
) -> str:
    root = ET.fromstring(xml_string)
    aliases = ROBOMIMIC_TASK_TEXTURE_LIST.get(task_name, list(texture_map.keys()))

    for alias in aliases:
        texture_name = texture_map.get(alias)
        material_name = ROBOMIMIC_TEXTURE_ALIAS.get(alias)
        if texture_name is None or material_name is None:
            continue

        material_elem = _find_asset_element(root, 'material', material_name)
        if material_elem is None:
            continue

        texture_type = ROBOMIMIC_TEXTURE_TYPES.get(alias, '2d')
        resolved_texture_name = _ensure_texture_asset(
            root=root,
            texture_name=texture_name,
            texture_type=texture_type,
        )
        material_elem.set('texture', resolved_texture_name)
        _apply_material_attributes(material_elem, (material_map or {}).get(alias))

    for alias, material_attributes in (material_map or {}).items():
        material_name = ROBOMIMIC_TEXTURE_ALIAS.get(alias)
        if material_name is None:
            continue
        material_elem = _find_asset_element(root, 'material', material_name)
        if material_elem is None:
            continue
        _apply_material_attributes(material_elem, material_attributes)

    _apply_skybox_config(root, skybox_config)
    _apply_light_config(root, light_config)

    return ET.tostring(root, encoding='unicode')


def apply_visual_mod_to_env(
    env,
    texture_map: Dict[str, str],
    task_name: Optional[str] = None,
    material_map: Optional[Dict[str, Dict[str, Any]]] = None,
    skybox_config: Optional[Dict[str, Any]] = None,
    light_config: Optional[Dict[str, Any]] = None,
) -> str:
    task_name = task_name or get_robomimic_task_name(env)
    xml_string = get_robomimic_env_xml(env)
    modified_xml = build_visual_mod_xml(
        xml_string=xml_string,
        texture_map=texture_map,
        task_name=task_name,
        material_map=material_map,
        skybox_config=skybox_config,
        light_config=light_config,
    )
    base_env = unwrap_robosuite_env(env)
    if not hasattr(base_env, 'reset_from_xml_string'):
        raise RuntimeError('Underlying robosuite env does not support reset_from_xml_string.')
    base_env.reset_from_xml_string(modified_xml)
    return modified_xml




class RobomimicImageWrapper(RobomimicRewardMixin, gym.Env):
    def __init__(self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray]=None,
        render_obs_key='agentview_image',
        ):

        self.env = env
        self.render_obs_key = render_obs_key
        self.init_state = init_state
        self.seed_state_map = dict()
        self._seed = None
        self.shape_meta = shape_meta
        self.render_cache = None
        self.has_reset_before = False
        self.episode_success = False

        # setup spaces
        action_shape = shape_meta['action']['shape']
        action_space = spaces.Box(
            low=-1,
            high=1,
            shape=action_shape,
            dtype=np.float32
        )
        self.action_space = action_space

        observation_space = spaces.Dict()
        for key, value in shape_meta['obs'].items():
            shape = value['shape']
            min_value, max_value = -1, 1
            if key.endswith('image'):
                min_value, max_value = 0, 1
            elif key.endswith('quat'):
                min_value, max_value = -1, 1
            elif key.endswith('qpos'):
                min_value, max_value = -1, 1
            elif key.endswith('pos'):
                # better range?
                min_value, max_value = -1, 1
            else:
                raise RuntimeError(f"Unsupported type {key}")

            this_space = spaces.Box(
                low=min_value,
                high=max_value,
                shape=shape,
                dtype=np.float32
            )
            observation_space[key] = this_space
        self.observation_space = observation_space


    def get_observation(self, raw_obs=None):
        if raw_obs is None:
            raw_obs = self.env.get_observation()

        self.render_cache = raw_obs[self.render_obs_key]

        obs = dict()
        for key in self.observation_space.keys():
            obs[key] = raw_obs[key]
        return obs

    def seed(self, seed=None):
        np.random.seed(seed=seed)
        self._seed = seed

    def reset(self):
        if self.init_state is not None:
            if not self.has_reset_before:
                # the env must be fully reset at least once to ensure correct rendering
                self.env.reset()
                self.has_reset_before = True

            # always reset to the same state
            # to be compatible with gym
            raw_obs = self.env.reset_to({'states': self.init_state})
        elif self._seed is not None:
            # reset to a specific seed
            seed = self._seed
            if seed in self.seed_state_map:
                # env.reset is expensive, use cache
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                # robosuite's initializes all use numpy global random state
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
        else:
            # random reset
            raw_obs = self.env.reset()
        self._reset_episode_success()

        # return obs
        obs = self.get_observation(raw_obs)
        return obs

    def step(self, action):
        raw_obs, reward, done, info = self.env.step(action)
        self._update_episode_success()
        obs = self.get_observation(raw_obs)
        return obs, reward, done, info

    def render(self, mode='rgb_array'):
        if self.render_cache is None:
            raise RuntimeError('Must run reset or step before render.')
        img = np.moveaxis(self.render_cache, 0, -1)
        img = (img * 255).astype(np.uint8)
        return img

    def get_state(self):
        if not hasattr(self.env, 'get_state'):
            raise AttributeError('Underlying env does not implement get_state.')
        return self.env.get_state()

    def close(self):
        if hasattr(self.env, 'close'):
            return self.env.close()
        return None

class RobomimicVisualModWrapper(RobomimicImageWrapper):
    def __init__(
        self,
        env: EnvRobosuite,
        shape_meta: dict,
        init_state: Optional[np.ndarray] = None,
        render_obs_key='agentview_image',
        task_name: Optional[str] = None,
        texture_map: Optional[Dict[str, str]] = None,
        material_map: Optional[Dict[str, Dict[str, Any]]] = None,
        skybox_config: Optional[Dict[str, Any]] = None,
        light_config: Optional[Dict[str, Any]] = None,
        texture_seed: Optional[int] = None,
        visual_mode: Optional[str] = None,
        scene_id: Optional[int] = None,
        randomize_on_reset: bool = False,
    ):
        super().__init__(
            env=env,
            shape_meta=shape_meta,
            init_state=init_state,
            render_obs_key=render_obs_key,
        )
        self.task_name = task_name or get_robomimic_task_name(env)
        self.texture_map = texture_map
        self.material_map = material_map
        self.skybox_config = skybox_config
        self.light_config = light_config
        self.texture_seed = texture_seed
        self.visual_mode = visual_mode
        self.scene_id = scene_id
        self.visual_config_override = None
        self.randomize_on_reset = randomize_on_reset
        self._base_xml_string = get_robomimic_env_xml(env)
        self.applied_texture_map = None
        self.applied_material_map = None
        self.applied_skybox_config = None
        self.applied_light_config = None
        self.last_applied_xml = None
        self._has_applied_visual_mod = False

    def _reset_with_current_mode(self):
        if self.init_state is not None:
            raw_obs = self.env.reset_to({'states': self.init_state})
            return raw_obs

        if self._seed is not None:
            seed = self._seed
            if seed in self.seed_state_map:
                raw_obs = self.env.reset_to({'states': self.seed_state_map[seed]})
            else:
                np.random.seed(seed=seed)
                raw_obs = self.env.reset()
                state = self.env.get_state()['states']
                self.seed_state_map[seed] = state
            self._seed = None
            return raw_obs

        return self.env.reset()

    def _get_current_visual_config(self) -> Dict[str, Any]:
        if self.visual_config_override is not None:
            override = self.visual_config_override
            return {
                'texture_map': dict(override['texture_map']),
                'material_map': None if override.get('material_map') is None else dict(override['material_map']),
                'skybox_config': None if override.get('skybox_config') is None else dict(override['skybox_config']),
                'light_config': None if override.get('light_config') is None else dict(override['light_config']),
            }

        sampled = sample_task_visual_config(
            task_name=self.task_name,
            seed=self.texture_seed,
            visual_mode=self.visual_mode,
            scene_id=self.scene_id,
        )
        return {
            'texture_map': dict(self.texture_map) if self.texture_map is not None else sampled['texture_map'],
            'material_map': dict(self.material_map) if self.material_map is not None else sampled['material_map'],
            'skybox_config': dict(self.skybox_config) if self.skybox_config is not None else sampled['skybox_config'],
            'light_config': dict(self.light_config) if self.light_config is not None else sampled['light_config'],
        }

    def apply_visual_mod(self) -> Dict[str, Any]:
        visual_config = self._get_current_visual_config()
        self.last_applied_xml = build_visual_mod_xml(
            xml_string=self._base_xml_string,
            texture_map=visual_config['texture_map'],
            task_name=self.task_name,
            material_map=visual_config['material_map'],
            skybox_config=visual_config['skybox_config'],
            light_config=visual_config['light_config'],
        )
        base_env = unwrap_robosuite_env(self.env)
        if not hasattr(base_env, 'reset_from_xml_string'):
            raise RuntimeError('Underlying robosuite env does not support reset_from_xml_string.')
        base_env.reset_from_xml_string(self.last_applied_xml)
        self.applied_texture_map = visual_config['texture_map']
        self.applied_material_map = visual_config['material_map']
        self.applied_skybox_config = visual_config['skybox_config']
        self.applied_light_config = visual_config['light_config']
        self._has_applied_visual_mod = True
        # reset_from_xml_string already performs an internal env reset.
        self.has_reset_before = True
        return visual_config

    def restore_visual_mod(self):
        base_env = unwrap_robosuite_env(self.env)
        if not hasattr(base_env, 'reset_from_xml_string'):
            raise RuntimeError('Underlying robosuite env does not support reset_from_xml_string.')
        base_env.reset_from_xml_string(self._base_xml_string)
        self.applied_texture_map = None
        self.applied_material_map = None
        self.applied_skybox_config = None
        self.applied_light_config = None
        self.last_applied_xml = None
        self._has_applied_visual_mod = True
        self.has_reset_before = True

    def reset(self):
        should_apply = self.randomize_on_reset or (not self._has_applied_visual_mod)
        if should_apply:
            self.apply_visual_mod()
            raw_obs = self._reset_with_current_mode()
            self._reset_episode_success()
            return self.get_observation(raw_obs)
        return super().reset()
