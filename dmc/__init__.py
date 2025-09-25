from collections import OrderedDict, deque
from typing import Any, NamedTuple

import dm_env
import numpy as np
from dm_control import suite
from dm_control.suite.wrappers import action_scale, pixels
from dm_env import StepType, specs

import dmc.custom_dmc_tasks as cdmc


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    physics: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


class FlattenJacoObservationWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        self._obs_spec = OrderedDict()
        wrapped_obs_spec = env.observation_spec().copy()
        if 'front_close' in wrapped_obs_spec:
            spec = wrapped_obs_spec['front_close']
            # drop batch dim
            self._obs_spec['pixels'] = specs.BoundedArray(
                shape=spec.shape[1:], dtype=spec.dtype, minimum=spec.minimum, maximum=spec.maximum, name='pixels'
            )
            wrapped_obs_spec.pop('front_close')

        for key, spec in wrapped_obs_spec.items():
            assert spec.dtype == np.float64
            assert type(spec) == specs.Array  # noqa
        dim = np.sum(np.fromiter((np.int(np.prod(spec.shape)) for spec in wrapped_obs_spec.values()), np.int32))

        self._obs_spec['observations'] = specs.Array(shape=(dim,), dtype=np.float32, name='observations')

    def _transform_observation(self, time_step):
        obs = OrderedDict()

        if 'front_close' in time_step.observation:
            pixels = time_step.observation['front_close']
            time_step.observation.pop('front_close')
            pixels = np.squeeze(pixels)
            obs['pixels'] = pixels

        features = []
        for feature in time_step.observation.values():
            features.append(feature.ravel())
        obs['observations'] = np.concatenate(features, axis=0)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionRepeatWrapper(dm_env.Environment):
    def __init__(self, env, num_repeats):
        self._env = env
        self._num_repeats = num_repeats

    def step(self, action):
        reward = 0.0
        discount = 1.0
        for i in range(self._num_repeats):
            time_step = self._env.step(action)
            reward += time_step.reward * discount
            discount *= time_step.discount
            if time_step.last():
                break

        return time_step._replace(reward=reward, discount=discount)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class FrameStackWrapper(dm_env.Environment):
    def __init__(self, env, num_frames, pixels_key='pixels'):
        self._env = env
        self._num_frames = num_frames
        self._frames = deque([], maxlen=num_frames)
        self._pixels_key = pixels_key

        wrapped_obs_spec = env.observation_spec()
        assert pixels_key in wrapped_obs_spec

        pixels_shape = wrapped_obs_spec[pixels_key].shape
        # remove batch dim
        if len(pixels_shape) == 4:
            pixels_shape = pixels_shape[1:]
        self._obs_spec = specs.BoundedArray(
            shape=np.concatenate([[pixels_shape[2] * num_frames], pixels_shape[:2]], axis=0),
            dtype=np.uint8,
            minimum=0,
            maximum=255,
            name='observation',
        )

    def _transform_observation(self, time_step):
        assert len(self._frames) == self._num_frames
        obs = np.concatenate(list(self._frames), axis=0)
        return time_step._replace(observation=obs)

    def _extract_pixels(self, time_step):
        pixels = time_step.observation[self._pixels_key]
        # remove batch dim
        if len(pixels.shape) == 4:
            pixels = pixels[0]
        return pixels.transpose(2, 0, 1).copy()

    def reset(self):
        time_step = self._env.reset()
        pixels = self._extract_pixels(time_step)
        for _ in range(self._num_frames):
            self._frames.append(pixels)
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        pixels = self._extract_pixels(time_step)
        self._frames.append(pixels)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ActionDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        wrapped_action_spec = env.action_spec()
        self._action_spec = specs.BoundedArray(
            wrapped_action_spec.shape, dtype, wrapped_action_spec.minimum, wrapped_action_spec.maximum, 'action'
        )

    def step(self, action):
        action = action.astype(self._env.action_spec().dtype)
        return self._env.step(action)

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._action_spec

    def reset(self):
        return self._env.reset()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ObservationDTypeWrapper(dm_env.Environment):
    def __init__(self, env, dtype):
        self._env = env
        self._dtype = dtype
        wrapped_obs_spec = env.observation_spec()['observations']
        self._obs_spec = specs.Array(wrapped_obs_spec.shape, dtype, 'observation')

    def _transform_observation(self, time_step):
        obs = time_step.observation['observations'].astype(self._dtype)
        return time_step._replace(observation=obs)

    def reset(self):
        time_step = self._env.reset()
        return self._transform_observation(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._transform_observation(time_step)

    def observation_spec(self):
        return self._obs_spec

    def action_spec(self):
        return self._env.action_spec()

    def __getattr__(self, name):
        return getattr(self._env, name)


class ExtendedTimeStepWrapper(dm_env.Environment):
    def __init__(self, env):
        self._env = env
        physics = env.physics.state()
        self._physics_spec = specs.Array(physics.shape, dtype=physics.dtype, name='physics')

    def reset(self):
        time_step = self._env.reset()
        return self._augment_time_step(time_step)

    def step(self, action):
        time_step = self._env.step(action)
        return self._augment_time_step(time_step, action)

    def _augment_time_step(self, time_step, action=None):
        if action is None:
            action_spec = self.action_spec()
            action = np.zeros(action_spec.shape, dtype=action_spec.dtype)

        def default_on_none(value, default):
            if value is None:
                return default
            return value

        return ExtendedTimeStep(
            observation=time_step.observation,
            step_type=time_step.step_type,
            action=action,
            reward=default_on_none(time_step.reward, 0.0),
            discount=default_on_none(time_step.discount, 1.0),
            physics=self._env.physics.state(),
        )

    def observation_spec(self):
        return self._env.observation_spec()

    def action_spec(self):
        return self._env.action_spec()

    def reward_spec(self):
        spec = self._env.reward_spec()
        if hasattr(self._task, 'get_reward_spec'):
            task_spec = self._task.get_reward_spec()
            if task_spec is not None:
                spec = task_spec
        if len(spec.shape) == 0:
            spec = spec.replace(shape=tuple((1,)), dtype=np.float32)
        return spec

    def physics_spec(self):
        return self._physics_spec

    def discount_spec(self):
        spec = self._env.discount_spec()
        if hasattr(self._task, 'get_discount_spec'):
            task_spec = self._task.get_discount_spec()
            if task_spec is not None:
                spec = task_spec
        if len(spec.shape) == 0:
            spec = spec.replace(shape=tuple((1,)), dtype=np.float32)
        return spec

    def __getattr__(self, name):
        return getattr(self._env, name)


def _make_jaco(obs_type, domain, task, frame_stack, action_repeat, seed, time_limit):
    env = cdmc.make_jaco(task, obs_type, seed, time_limit)
    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    env = FlattenJacoObservationWrapper(env)
    return env


def _make_dmc(obs_type, domain, task, frame_stack, action_repeat, seed, time_limit):
    visualize_reward = False
    if (domain, task) in suite.ALL_TASKS:
        env = suite.load(
            domain,
            task,
            task_kwargs=dict(random=seed, time_limit=time_limit),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )
    else:
        env = cdmc.make(
            domain,
            task,
            task_kwargs=dict(random=seed, time_limit=time_limit),
            environment_kwargs=dict(flat_observation=True),
            visualize_reward=visualize_reward,
        )

    env = ActionDTypeWrapper(env, np.float32)
    env = ActionRepeatWrapper(env, action_repeat)
    if obs_type == 'pixels':
        # zoom in camera for quadruped
        camera_id = dict(quadruped=2).get(domain, 0)
        render_kwargs = dict(height=84, width=84, camera_id=camera_id)
        env = pixels.Wrapper(env, pixels_only=True, render_kwargs=render_kwargs)
    return env


def _make(name, obs_type='states', frame_stack=1, action_repeat=1, seed=1, time_limit=float('inf')):
    assert obs_type in ['states', 'pixels']
    if name.startswith('point_mass_maze'):
        domain = 'point_mass_maze'
        _, _, _, task = name.split('_', 3)
    elif name.startswith('hybrid_'):
        _, domain, task = name.split('_', 2)
    else:
        domain, task = name.split('_', 1)
    domain = dict(cup='ball_in_cup').get(domain, domain)

    make_fn = _make_jaco if domain == 'jaco' else _make_dmc
    env = make_fn(obs_type, domain, task, frame_stack, action_repeat, seed, time_limit)

    if obs_type == 'pixels':
        env = FrameStackWrapper(env, frame_stack)
    else:
        env = ObservationDTypeWrapper(env, np.float32)

    env = action_scale.Wrapper(env, minimum=-1.0, maximum=+1.0)
    env = ExtendedTimeStepWrapper(env)
    return env


import os

import gymnasium as gym
import mujoco


# metamotivo:
# def reward_inference(self, task) -> torch.Tensor:
#     env = suite.load(
#         domain_name=self.cfg.domain_name,
#         task_name=task,
#         environment_kwargs={"flat_observation": True},
#     )
#     num_samples = self.cfg.num_inference_samples
#     batch = self.replay_buffer["train"].sample(num_samples)
#     rewards = []
#     for i in range(num_samples):
#         with env._physics.reset_context():
#             env._physics.set_state(batch["next"]["physics"][i].cpu().numpy())
#             env._physics.set_control(batch["action"][i].cpu().detach().numpy())
#         mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
#         mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
#         mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
#         mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
#         rewards.append(env._task.get_reward(env._physics))
#     rewards = np.array(rewards).reshape(-1, 1)
#     z = self.agent._model.reward_inference(
#         next_obs=batch["next"]["observation"],
#         reward=torch.tensor(rewards, dtype=torch.float32, device=self.agent.device),
#     )
#     return z
def infer_reward(env, next_physics, action):
    with env._physics.reset_context():
        env._physics.set_state(next_physics)
        env._physics.set_control(action)
    mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
    mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
    mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
    mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
    return env._task.get_reward(env._physics)


def label_exorl_dataset_episodes(
    domain_task: str, dataset, mode: str, n: int, rng: np.random.Generator | None = None
) -> tuple[np.ndarray, np.ndarray]:
    assert 'valids' in dataset, 'rest should not be supported'
    valids = dataset['valids']
    valid_ixs = valids.nonzero()[0]
    ep_starts = np.roll((valids == 0.0).nonzero()[0] + 1, 1)
    ep_starts[0] = 0

    if mode == 'early':
        ixs = valid_ixs[:n]
    elif mode == 'late':
        ixs = valid_ixs[-n:]
    elif mode == 'random':
        ep_ixs = rng.choice(ep_starts, (np.ceil(n / 1000).astype(np.int32),))
        ixs = (ep_ixs[:, None] + np.arange(1000)).reshape(-1)
        ixs = ixs[:n]
    else:
        assert False, 'invalid mode'

    assert np.all(valids[ixs]), 'not all ixs are valid!'

    next_physics = dataset['physics'][ixs + 1]  # +1 because see above
    actions = dataset['actions'][ixs]
    next_observations = dataset['observations'][ixs + 1]  # +1 because see above
    rewards = np.zeros((n,))

    env = _make(domain_task)
    rewards = np.asarray([infer_reward(env, nph, a) for nph, a in zip(next_physics, actions)])
    return next_observations, rewards


_experience_dset_src = 'datasets'


def get_experience_episodes(
    domain: str, task_name: str, agent: str, n: int, seed: int, rng: np.random.Generator | None = None
):
    p = _experience_dset_src
    if prefix := os.environ.get('SCRATCH', None):  # on the cluster
        p = os.path.join(prefix, 'zsmrl', p)
    p = os.path.join(p, agent, str(seed), domain, task_name)
    files = [x for x in os.listdir(p) if x.endswith('.npz')]
    f = rng.choice(files).item()
    x = np.load(os.path.join(p, f), allow_pickle=True)
    n = min(len(x['rewards']), n)
    o = x['next_observations'][:n]
    r = x['rewards'][:n]
    return o, r


class DMCAllTaskWrapper(gym.Env):
    _tasks = {
        'walker': ['stand', 'walk', 'run', 'flip'],
        'cheetah': ['walk', 'run', 'walk_backward', 'run_backward'],  # flip, flip_backward
        'quadruped': ['stand', 'jump', 'walk', 'run'],
        # 'point_mass_maze': [
        #     'reach_top_left',
        #     'reach_top_right',
        #     'reach_bottom_right',
        #     'reach_bottom_left',
        # ],  # 'loop', 'square', 'fast_slow',
        # 'jaco': [
        #     'reach_top_left',
        #     'jaco_reach_top_right',
        #     'jaco_reach_bottom_left',
        #     'jaco_reach_bottom_right',
        # ],
        # 'humanoid': ['stand', 'walk', 'run'],
        'hybrid_walker': ['stand2flip', 'run2flip', 'speedup', 'slowdown'],
    }

    def __init__(
        self,
        domain: str,
        dataset: dict,
        max_episode_steps: int = 1000,
        num_samples: int = 50_000,
        mode: str = 'default',
    ):
        self._dmc_env_name = domain

        tasks = list(self._tasks[domain])
        self._tasks = []
        ctrl_timestep = None
        for t in tasks:
            try:
                env = _make(domain + '_' + t)
                ctrl_timestep = env.control_timestep()
                self._tasks.append(t)
            except Exception as e:
                raise ValueError(f'missing task {t}') from e
        print(f'{domain} tasks', ', '.join(self._tasks))

        self._tl = ctrl_timestep * max_episode_steps

        self._dataset = dataset

        self._num_samples = num_samples
        self._mode = mode
        self._last_seed = np.random.randint(0, 2**32 - 1)
        self._rng = np.random.RandomState(self._last_seed)

        self._dmc_env = None
        self._task_id = None

        self._task_tasks = dict()

    @property
    def observation_space(self):
        raise NotImplementedError

    @property
    def action_space(self):
        raise NotImplementedError

    @property
    def task_infos(self):
        return [{'task_name': f'task{i + 1}_{t}', 'task_type': 'rl'} for i, t in enumerate(self._tasks)]

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._last_seed = seed
            self._rng.seed(seed)

        if options is None or 'task_id' not in options:
            # task_id = self._rng.randint(len(self._tasks)) + 1
            assert False, 'provide task id'
        else:
            task_id = options['task_id']
        task_name = self._tasks[task_id - 1]
        domain_task = self._dmc_env_name + '_' + task_name

        if self._task_id != task_id:
            seed = self._rng.randint(0, 2**32 - 1)
            self._dmc_env = _make(domain_task, seed=seed, time_limit=self._tl)
            self._task_id = task_id

        ts = self._dmc_env.reset()

        if self._mode == 'default':
            if not (user_task := self._task_tasks.get(task_id, None)):
                user_task = self.reward_inference(task_name)
                self._task_tasks[task_id] = user_task
        elif self._mode == 'default-no-cache':
            user_task = self.reward_inference(task_name)
        elif self._mode in ['early', 'late', 'random']:
            user_task = label_exorl_dataset_episodes(
                domain_task, self._dataset, self._mode, self._num_samples, self._rng
            )
        elif self._mode.startswith('agent-'):
            agent_name = self._mode[len('agent-') :]
            user_task = get_experience_episodes(
                self._dmc_env_name, task_name, agent_name, self._num_samples, self._last_seed, self._rng
            )
        else:
            assert False, f'invalid mode {self._mode}'

        return ts.observation, dict(physics=ts.physics, task=user_task)

    def step(self, action):
        ts = self._dmc_env.step(action)
        terminated = ts.last() and ts.discount == 0.0
        truncated = ts.last() and ts.discount > 0.0
        log = {}
        if hasattr(self._dmc_env._task, 'log_info'):
            log = self._dmc_env._task.log_info(self._dmc_env._physics)
        return ts.observation, ts.reward, terminated, truncated, dict(physics=ts.physics, **log)

    def reward_inference(self, task) -> tuple[np.ndarray, np.ndarray]:
        env = _make(self._dmc_env_name + '_' + task)
        num_samples = self._num_samples
        assert 'valids' in self._dataset, 'rest should not be supported'
        valids = self._dataset['valids']
        valid_ixs = valids.nonzero()[0]
        ixs = self._rng.choice(valid_ixs, size=num_samples, replace=True)
        physics = self._dataset['physics'][ixs + 1]  # +1 because see above
        actions = self._dataset['actions'][ixs]
        observations = self._dataset['observations'][ixs + 1]  # +1 because see above
        rewards = np.zeros(num_samples)
        for i in range(num_samples):
            with env._physics.reset_context():
                env._physics.set_state(physics[i])
                env._physics.set_control(actions[i])
            mujoco.mj_forward(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_fwdPosition(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_sensorVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            mujoco.mj_subtreeVel(env._physics.model.ptr, env._physics.data.ptr)  # pylint: disable=no-member
            rewards[i] = env._task.get_reward(env._physics)
        return observations, rewards

    def render(self, *args, **kwargs):
        kwargs.setdefault('camera_id', 1 if 'quadruped' in self._dmc_env_name else 0)
        return self._dmc_env.physics.render(*args, **kwargs)
