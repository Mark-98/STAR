"""Relocate legacy RL pickle module names. Only load trusted checkpoints."""
import pickle

_LOCAL = {'utils', 'utils_vigen', 'augmentations', 'replay_buffer', 'logger', 'video',
          'wandblogger', 'train_teacher', 'train_stage2', 'arguments'}


class Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith(('algos.consistency_policy_algo.', 'consistency_policy.',
                              'star.rl.algos.consistency_policy_algo.')) or name == 'Diffusion_Recon':
            raise ValueError('Diffusion/consistency checkpoints are outside this STAR MLP release.')
        if module == 'algos' or module.startswith(('algos.', 'wrappers.')) or module in _LOCAL:
            module = 'star.rl.' + module
        return super().find_class(module, name)


Pickler = pickle.Pickler
load = pickle.load
loads = pickle.loads
dump = pickle.dump
dumps = pickle.dumps


def load_checkpoint(*args, **kwargs):
    import sys
    import torch
    kwargs.setdefault('pickle_module', sys.modules[__name__])
    return torch.load(*args, **kwargs)
