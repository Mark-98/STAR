from star.utils.paths import resource
from typing import Dict
from pathlib import Path
from omegaconf import DictConfig
from hydra import compose, initialize_config_dir
import robosuite as suite
from robosuitevgb.vgb_wrapper import VGBWrapper
from robosuite.controllers import load_controller_config
import hydra
import yaml


ROBO_SETTING_PATH = resource('configs', 'environments', 'robosuite', 'setting', 'robo_setting.yaml')

def omegaconf_to_dict(d: DictConfig) -> Dict:
    """Converts an omegaconf DictConfig to a python Dict, respecting variable interpolation."""
    ret = {}
    for k, v in d.items():
        if isinstance(v, DictConfig):
            ret[k] = omegaconf_to_dict(v)
        else:
            ret[k] = v
    return ret


def make_env_train(
    task_name: str,
    seed: int,
    scene_id: int = 0,
    mode_override: str = None,
    camera_mode_override: str = None,
):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_train_config")
        cfg_dict = omegaconf_to_dict(cfg)
    if mode_override is not None:
        cfg_dict['mode'] = mode_override
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    camera_mode = camera_mode_override
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        camera_mode = cfg_dict['mode']
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    if camera_mode in {'cam-easy', 'cam-hard'}:
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        difficulty = 'easy' if camera_mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
        cfg_dict['camera']['position_perturbation_size'] = xml_dict['camera']['position'][difficulty]
        cfg_dict['camera']['rotation_perturbation_size'] = xml_dict['camera']['rotation'][difficulty]
        cfg_dict['camera']['fovy_perturbation_size'] = xml_dict['camera']['fov'][difficulty]
        randomize_camera = True
    print('Now the mode is {}'.format(cfg_dict['mode']))
    if camera_mode is not None:
        print('Now the camera mode is {}'.format(camera_mode))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_config")
        cfg_dict = omegaconf_to_dict(cfg)
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_easy(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_easy_config")
        cfg_dict = omegaconf_to_dict(cfg)
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_medium(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_medium_config")
        cfg_dict = omegaconf_to_dict(cfg)
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_hard(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_hard_config")
        cfg_dict = omegaconf_to_dict(cfg)
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env


def make_env_train_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_train_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_easy_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_easy_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_medium_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_medium_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_hard_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_hard_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    if cfg_dict['mode'] == 'train':
        randomize_color = randomize_lighting = False
        randomize_dynamics = False
        randomize_camera = False
        cfg_dict['moving_light'] = False
        cfg_dict['except_robot'] = True
    elif cfg_dict['mode'] == 'eval-easy':
        cfg_dict['except_robot'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-medium':
        cfg_dict['except_robot'] = False
        cfg_dict['moving_light'] = False
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'eval-hard':
        cfg_dict['except_robot'] = True
        cfg_dict['moving_light'] = False
        cfg_dict['video_background'] = True
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    elif cfg_dict['mode'] == 'cam-easy' or cfg_dict['mode'] == 'cam-hard':
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = True
        with open(ROBO_SETTING_PATH, 'r') as f:
            xml_content = f.read()
        xml_dict = yaml.safe_load(xml_content)
        mode = cfg_dict['mode']
        difficulty = 'easy' if mode == 'cam-easy' else 'hard'

        for key in ['position', 'rotation', 'fov']:
            cfg_dict['_camera_'][key] = xml_dict['camera'][key][difficulty]
    else:
        randomize_color = randomize_lighting = True
        randomize_dynamics = False
        randomize_camera = False
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env

def make_env_eval_video_hand(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_video_hand_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    randomize_color = randomize_lighting = False
    randomize_dynamics = False
    randomize_camera = False
    cfg_dict['moving_light'] = False
    cfg_dict['except_robot'] = True
    cfg_dict['video_background'] = True
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env


def make_env_eval_video(task_name: str, seed: int, scene_id: int = 0):
    """
    create the robo envs with different parameters.
    """
    hydra.core.global_hydra.GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=str(resource("configs", "environments", "robosuite"))):
        cfg = compose(config_name="robo_eval_video_config")
        cfg_dict = omegaconf_to_dict(cfg)
    print("camera_names: ", cfg_dict['task_def']['camera_names'])
    assert scene_id >= 0 and scene_id <= 9
    cfg_dict['scene_id'] = scene_id
    cfg_dict['task_def']['controller_configs'] = load_controller_config(
        default_controller=cfg_dict['task_def']['controller_configs'])
    cfg_dict['task_def']['env_name'] = task_name
    if type(cfg_dict['task_def']['robots']) != str:
        robots = []
        for robot in cfg_dict['task_def']['robots']:
            robots.append(robot)
        cfg_dict['task_def']['robots'] = robots
    if "TwoArm" in cfg_dict['task_def']['env_name'] and type(cfg_dict['task_def']['robots']) == str:
        cfg_dict['task_def']['robots'] = [cfg_dict['task_def']['robots']] * 2
    env = suite.make(
        **cfg_dict['task_def'],
        use_camera_obs=True,
        has_offscreen_renderer=True,
    )
    randomize_color = randomize_lighting = False
    randomize_dynamics = False
    randomize_camera = False
    cfg_dict['moving_light'] = False
    cfg_dict['except_robot'] = True
    cfg_dict['video_background'] = True
    print('Now the mode is {}'.format(cfg_dict['mode']))
    env = VGBWrapper(env=env,
                     color_randomization_args=cfg_dict['color'],
                     camera_randomization_args=cfg_dict['camera'],
                     lighting_randomization_args=cfg_dict['light'],
                     dynamics_randomization_args=cfg_dict['dynamic'],
                     seed=seed,
                     task=cfg_dict['task_def']['env_name'],
                     obs_modality=cfg_dict['obs_modality'],
                     channel_first=cfg_dict['channel_first'],
                     camera_depths=cfg_dict['task_def']['camera_depths'],
                     episode_length=cfg_dict['task_def']['horizon'],
                     hard_reset=cfg_dict['task_def']['hard_reset'],
                     mode=cfg_dict['mode'],
                     scene_id=cfg_dict['scene_id'],
                     moving_light=cfg_dict['moving_light'],
                     randomize_color=randomize_color,
                     randomize_camera=randomize_camera,
                     randomize_lighting=randomize_lighting,
                     randomize_dynamics=randomize_dynamics,
                     video_background=cfg_dict['video_background'],
                     image_height=cfg_dict['image_height'],
                     image_width=cfg_dict['image_width'],
                     except_robot=cfg_dict['except_robot'],
                     )
    return env
