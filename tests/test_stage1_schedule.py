"""Exercise the native action/update block without a GPU or a simulator."""
import ast
import copy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
import torch
from torch import nn
import yaml

from star.rl import utils
from star.rl.algos.stage1 import STARStage1, _consume_legacy_decoder_rng
from star.rl.algos.utils.networks import ImageDecoder
from star.rl.arguments import parse_args


ROOT = Path(__file__).resolve().parents[1]


def native_action_update_block():
    """Compile the real workspace block, not a separate scheduling formula."""
    source = ROOT / 'star/rl/train_teacher.py'
    tree = ast.parse(source.read_text())
    workspace = next(node for node in tree.body if isinstance(node, ast.ClassDef)
                     and node.name == 'Workspace')
    train = next(node for node in workspace.body if isinstance(node, ast.FunctionDef)
                 and node.name == 'train')
    blocks = [node for node in ast.walk(train) if isinstance(node, ast.If)
              and isinstance(node.test, ast.UnaryOp)
              and isinstance(node.test.op, ast.Not)
              and isinstance(node.test.operand, ast.Call)
              and isinstance(node.test.operand.func, ast.Name)
              and node.test.operand.func.id == 'seed_until_step']
    assert len(blocks) == 1
    function = ast.parse('def run(self, time_step, seed_until_step): pass').body[0]
    function.body = [copy.deepcopy(blocks[0])]
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = dict(np=np, torch=torch, utils=utils)
    exec(compile(module, str(source), 'exec'), namespace)
    return namespace['run']


class CounterAgent:
    def __init__(self):
        self.training = True
        self.events = []
        for name in ('critic', 'dyna', 'actor', 'alpha'):
            module = nn.Linear(1, 1, bias=False)
            setattr(self, name, module)
            setattr(self, name + '_opt', torch.optim.Adam(module.parameters(), lr=5e-5))

    def train(self, training):
        self.training = training

    def act(self, observation, step, eval_mode=False):
        self.events.append(('act', step))
        return np.zeros(1, dtype=np.float32)

    def update(self, replay_iter, step):
        self.events.append(('update', step))
        names = ('critic', 'dyna', 'actor', 'alpha') if step % 2 == 0 else ('critic', 'dyna')
        for name in names:
            optimizer = getattr(self, name + '_opt')
            optimizer.zero_grad()
            getattr(self, name).weight.square().sum().backward()
            optimizer.step()
        return {'critic_loss': 0.}

    def counts(self):
        return {name: int(next(iter(getattr(self, name + '_opt').state.values()))['step'])
                for name in ('critic', 'dyna', 'actor', 'alpha')}


class Stage1ScheduleTests(unittest.TestCase):
    def setUp(self):
        self.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, self.previous_threads)

    def test_warmup_defaults_and_door_recipe(self):
        cfg = parse_args([])
        self.assertEqual((cfg.init_steps, cfg.num_seed_frames), (1000, 1000))
        for name in ('stage1.yaml', 'stage1_door.yaml'):
            values = yaml.safe_load((ROOT / 'configs/rl' / name).read_text())
            self.assertEqual((values['init_steps'], values['num_seed_frames']), (1000, 1000))
        door = yaml.safe_load((ROOT / 'configs/rl/stage1_door.yaml').read_text())
        self.assertEqual((door['batch_size'], door['lr'], door['replay_buffer_num_workers']),
                         (256, 5e-5, 2))
        self.assertTrue(door['skip_eval'])
        self.assertFalse(door['use_target_enc'])

    def test_actual_adam_counts_and_action_order_through_initial_100_updates(self):
        run = native_action_update_block()
        cfg = parse_args(['--init_steps', '1000', '--num_seed_frames', '1000'])
        agent = CounterAgent()
        action_spec = SimpleNamespace(minimum=-1., maximum=1., shape=(1,))
        workspace = SimpleNamespace(cfg=cfg, agent=agent, _mechanism=None,
                                    replay_iter=iter(()), wandblogger=Mock(),
                                    train_env=SimpleNamespace(action_spec=lambda: action_spec))
        time_step = SimpleNamespace(observation=np.zeros(1, dtype=np.float32))
        trace = {}
        seed_until = utils.Until(cfg.num_seed_frames, 1)
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=AssertionError('No clipping')), \
             patch('torch.nn.utils.clip_grad_value_', side_effect=AssertionError('No clipping')):
            for step in range(1100):
                workspace.global_step = workspace.global_frame = step
                run(workspace, time_step, seed_until)
                if step == 999:
                    self.assertEqual(agent.events, [])
                    self.assertEqual(len(agent.critic_opt.state), 0)
                if step in (1000, 1001, 1002, 1099):
                    trace[step] = agent.counts()
        self.assertEqual([trace[s]['critic'] for s in trace], [1, 2, 3, 100])
        self.assertEqual([trace[s]['dyna'] for s in trace], [1, 2, 3, 100])
        self.assertEqual([trace[s]['actor'] for s in trace], [1, 1, 2, 50])
        self.assertEqual([trace[s]['alpha'] for s in trace], [1, 1, 2, 50])
        self.assertEqual(agent.events, [event for step in range(1000, 1100)
                                      for event in (('act', step), ('update', step))])
        self.assertEqual(workspace.wandblogger.log_metrics.call_count, 100)
        self.assertTrue(agent.training)

    def test_init_steps_remains_an_update_gate_not_a_burst_size(self):
        run = native_action_update_block()
        agent = Mock()
        agent.training = True
        workspace = SimpleNamespace(cfg=SimpleNamespace(init_steps=1002), agent=agent,
                                    _mechanism=None, replay_iter=iter(()), wandblogger=Mock())
        for step in (1000, 1001, 1002, 1003):
            workspace.global_step = workspace.global_frame = step
            run(workspace, SimpleNamespace(observation=None), lambda _: False)
        self.assertEqual([call.args[1] for call in agent.update.call_args_list], [1002, 1003])

    def test_decoder_compatibility_consumes_only_the_expected_cpu_rng(self):
        for dimension in (100, 256):
            with self.subTest(repr_dim=dimension), torch.random.fork_rng(devices=[]):
                torch.manual_seed(731)
                ImageDecoder(dimension)
                expected = torch.get_rng_state().clone()
                torch.manual_seed(731)
                self.assertIsNone(_consume_legacy_decoder_rng(dimension))
                self.assertTrue(torch.equal(torch.get_rng_state(), expected))

    def test_native_critic_dynamics_gradients_are_not_clipped(self):
        class Critic(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.tensor(0.))
                self.encoder = SimpleNamespace(shared_cnn=nn.Identity())

            def forward(self, obs, action, encoded=False):
                q = self.weight.expand(obs.shape[0], 1)
                return q, q

        class Dynamics(nn.Linear):
            def forward(self, obs, action):
                return super().forward(obs)

            def loss(self, prediction, target):
                return nn.functional.mse_loss(prediction, target)

        agent = STARStage1.__new__(STARStage1)
        agent.args = SimpleNamespace(dyna_alpha=1., use_target_enc=False)
        agent.log_alpha = torch.tensor(0.)
        agent.critic = Critic()
        agent.critic_target = Critic()
        agent.dyna = Dynamics(1, 1, bias=False)
        agent.critic_opt = torch.optim.Adam(agent.critic.parameters(), lr=5e-5)
        agent.dyna_opt = torch.optim.Adam(agent.dyna.parameters(), lr=5e-5)
        agent.actor = lambda x: (None, torch.zeros_like(x), torch.zeros_like(x), None)
        batch = torch.ones(2, 1)
        with patch('torch.nn.utils.clip_grad_norm_', side_effect=AssertionError('No clipping')), \
             patch('torch.nn.utils.clip_grad_value_', side_effect=AssertionError('No clipping')):
            metrics = agent.update_critic_fw(batch, batch, 1000. * batch, batch, batch)
        self.assertGreater(abs(agent.critic.weight.grad.item()), 100.)
        self.assertTrue(np.isfinite(metrics['critic_loss']))
        self.assertEqual(int(next(iter(agent.critic_opt.state.values()))['step']), 1)
        self.assertEqual(int(next(iter(agent.dyna_opt.state.values()))['step']), 1)


if __name__ == '__main__':
    unittest.main()
