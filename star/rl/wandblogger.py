import torch
import logging
import os
import wandb
import json
from pathlib import Path
from contextlib import nullcontext


logger = logging.getLogger(__name__)


class WandbLogger(object):

    def __init__(self, config):
        """
        General logger.

        Args:
            log_dir (str): log directory
        """
        self.info = logger.info
        self.debug = logger.debug
        self.warning = logger.warning
        self.local_metrics_path = getattr(config, 'local_metrics_path', None)
        context = nullcontext()
        if getattr(config, 'diagnostic_hook', '') == 'experiments.diagnostics_1k.online:Measurements':
            raise NotImplementedError('Experimental diagnostic hooks are excluded from this release.')
        project = (f"{config.task_name}_new" if config.algo == 'dad2'
                   else f"{config.task_name}")
        project = getattr(config, 'wandb_project', None) or os.environ.get('WANDB_PROJECT') or project
        options = dict(project=project, name=config.exp_name, config=vars(config),
                       mode=getattr(config, 'wandb_mode', None) or os.environ.get('WANDB_MODE', 'offline'))
        for field in ('entity', 'group'):
            value = getattr(config, 'wandb_' + field, None) or os.environ.get('WANDB_' + field.upper())
            if value:
                options[field] = value
        with context:
            self.run = wandb.init(**options)

    def scalar_summary(self, tag, value, step):
        """
        Log scalar value to the disk.
        Args:
            tag (str): name of the value
            value (float): value
            step (int): update step
        """
        # self.writer.add_scalar(tag, value, step)
        data = {}
        data[tag]=value
        # wandb.log(data, step)
        try:
            # print(0)
            self._log(data, step)
        except Exception as e:
            print(e)

    def log_metrics(self, metrics, step, ty):
        """
        Logs a dictionary of metrics to Weights & Biases.

        Args:
            metrics (dict): A dictionary of metrics to log, where keys are the metric names and values are the metric values.
            step (int): The current step or epoch in training or evaluation.
            ty (str): The type of the metrics, typically 'train' or 'eval'.
        """
        # Preprocess the metrics dictionary to include the type (train/eval) in the keys
        wandb_metrics = {f'{ty}/{key}': value.item() if isinstance(value, torch.Tensor) else value for key, value in metrics.items()}

        # Log the metrics to wandb
        self._log(wandb_metrics, step)

    def use_environment_steps(self, action_repeat):
        """Opt-in for the 1K protocol; callers still supply native frame counts."""
        raise NotImplementedError('The 1K diagnostic protocol is excluded from this release.')

    def _log(self, metrics, frame):
        if self.local_metrics_path:
            with Path(self.local_metrics_path).open('a') as stream:
                stream.write(json.dumps({'step': int(frame), 'metrics': metrics}, default=str) + '\n')
        repeat = getattr(self, '_environment_step_repeat', None)
        if repeat is None:
            return wandb.log(metrics, step=frame)
        if int(frame) % repeat:
            raise ValueError('Native W&B frame is not divisible by action_repeat')
        step = int(frame) // repeat
        raise NotImplementedError('The 1K diagnostic protocol is excluded from this release.')


    def close(self):
        if self.run is not None:
            self.run.finish()
            self.run = None
