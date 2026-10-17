"""Create an exact frozen scene without a trained policy or historical logs."""
import json
from pathlib import Path
from star.environments.scenes import get_scene, visual_config
from star.utils.paths import resource


def make(task, scene_id, dataset_path, demo_id=150):
    import h5py
    import numpy as np
    from omegaconf import OmegaConf
    from star.evaluation.eval_robomimic_texture_generalization import create_wrapped_eval_env

    record = get_scene(task, scene_id)
    with h5py.File(Path(dataset_path).expanduser(), 'r') as data:
        meta = json.loads(data['data'].attrs['env_args'])
        state = data[f'data/demo_{demo_id}/states'][0]
    expected = {'can': 'PickPlaceCan', 'square': 'NutAssemblySquare'}[task]
    if meta['env_name'] != expected:
        raise ValueError(f'Dataset task mismatch: expected {expected}, got {meta["env_name"]}.')
    meta['env_kwargs']['controller_configs']['control_delta'] = False
    task_cfg = OmegaConf.load(resource('configs', 'diffusion', 'task', task + '_image_abs.yaml'))
    cfg = OmegaConf.create({'task': task_cfg, 'n_obs_steps': 2, 'n_action_steps': 8,
                            'past_action_visible': False})
    # Only resolved runner fields are used; no dataset/cache/teacher is instantiated.
    cfg.task.env_runner.max_steps = 400
    env = create_wrapped_eval_env(cfg, meta, 42, apply_visual_mod=True,
                                 visual_config=visual_config(record), visual_mode='eval-easy',
                                 scene_id=scene_id, exact_visual_config=True)
    env.env.init_state = np.array(state, copy=True)
    env.env._seed = None
    env.reset()
    actual = env.env.env.get_state()['states']
    # Match the original evaluator's floating-point reset tolerance exactly.
    if not np.allclose(actual, state, rtol=0, atol=1e-10):
        env.close()
        raise RuntimeError(f'Frozen scene changed the demonstration state; max error={np.max(np.abs(actual-state))}.')
    return env
