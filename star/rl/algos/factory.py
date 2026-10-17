"""Resolve the selected original agent without importing unrelated algorithms."""
from collections.abc import MutableMapping
from importlib import import_module


_CLASS_PATHS = {
    'dad': ('star.rl.algos.stage1', 'STARStage1'),
    'dad2': ('star.rl.algos.stage2', 'STARStage2'),
    'star_stage1': ('star.rl.algos.stage1', 'STARStage1'),
    'star_stage2': ('star.rl.algos.stage2', 'STARStage2'),
}


class _AlgorithmRegistry(MutableMapping):
    def __init__(self):
        self._values = dict(_CLASS_PATHS)

    def __getitem__(self, key):
        value = self._values[key]
        if isinstance(value, tuple):
            import random
            import numpy as np
            import torch
            module, name = value
            # In legacy workspaces class resolution can happen after seeding.
            # Optional-module initialization must not consume that training RNG.
            py_state, np_state = random.getstate(), np.random.get_state()
            flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
            devices = None if torch.cuda.is_initialized() else []
            with torch.random.fork_rng(devices=devices):
                try:
                    value = getattr(import_module(module), name)
                finally:
                    random.setstate(py_state)
                    np.random.set_state(np_state)
                    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = flags
            self._values[key] = value
        return value

    def __setitem__(self, key, value):
        self._values[key] = value

    def __delitem__(self, key):
        del self._values[key]

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)

    def copy(self):
        return dict(self)


algorithm = _AlgorithmRegistry()
__all__ = ['algorithm', 'make_agent'] + [value[1] for value in _CLASS_PATHS.values()]


def __getattr__(name):
    # Existing from-imports of agent class names still resolve.
    for key, (_, exported) in _CLASS_PATHS.items():
        if name == exported:
            return algorithm[key]
    raise AttributeError(name)


def make_agent(obs_shape, action_shape, args):
    args.obs_shape = obs_shape
    args.action_shape = action_shape
    return algorithm[args.algo](args)
