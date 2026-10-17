from robomimic.envs.env_robosuite import EnvRobosuite


class RobomimicRewardMixin:
    """
    Shared helpers for wrappers whose ``self.env`` is a robomimic ``EnvRobosuite``.

    Attribute chain (verified against robomimic 0.2.0 / robosuite 1.2.0):
        self.env            -> robomimic.envs.env_robosuite.EnvRobosuite
        self.env.env        -> robosuite env (e.g. PickPlaceCan, NutAssemblySquare)
        self.env.env.reward_shaping   -> bool read by robosuite ``reward()`` every step
        self.env.env._check_success() -> task success condition, exposed by EnvRobosuite.is_success()
    """

    def _robosuite_env(self):
        env = self.env
        assert isinstance(env, EnvRobosuite), type(env)
        assert hasattr(env.env, 'reward_shaping'), type(env.env)
        return env.env

    def set_reward_shaping(self, value: bool):
        # only this env instance is affected; env_meta / other instances are untouched
        self._robosuite_env().reward_shaping = bool(value)

    def get_reward_shaping(self) -> bool:
        return bool(self._robosuite_env().reward_shaping)

    def is_success(self):
        # robosuite task success condition via robomimic's official API
        return self.env.is_success()

    def _reset_episode_success(self):
        self.episode_success = False

    def _update_episode_success(self):
        # latch: an episode counts as successful once the task condition is met
        if not self.episode_success and bool(self.is_success()['task']):
            self.episode_success = True
