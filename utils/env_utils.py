import collections
import os
import platform
import time
import warnings

import gymnasium
import numpy as np
import ogbench
import ogbench.utils
from gymnasium.spaces import Box

from dmc import DMCAllTaskWrapper
from dmc.exorl import load_exorl_dataset
from utils.datasets import Dataset


class EpisodeMonitor(gymnasium.Wrapper):
    """Environment wrapper to monitor episode statistics."""

    def __init__(self, env):
        super().__init__(env)
        self._reset_stats()
        self.total_timesteps = 0

    def _reset_stats(self):
        self.reward_sum = 0.0
        self.episode_length = 0
        self.start_time = time.time()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.reward_sum += reward
        self.episode_length += 1
        self.total_timesteps += 1
        info['total'] = {'timesteps': self.total_timesteps}

        if terminated or truncated:
            info['episode'] = {}
            info['episode']['return'] = self.reward_sum
            info['episode']['length'] = self.episode_length
            info['episode']['duration'] = time.time() - self.start_time

            if hasattr(self.unwrapped, 'get_normalized_score'):
                info['episode']['normalized_return'] = (
                    self.unwrapped.get_normalized_score(info['episode']['return']) * 100.0
                )

        return observation, reward, terminated, truncated, info

    def reset(self, *args, **kwargs):
        self._reset_stats()
        return self.env.reset(*args, **kwargs)


class FrameStackWrapper(gymnasium.Wrapper):
    """Environment wrapper to stack observations."""

    def __init__(self, env, num_stack):
        super().__init__(env)

        self.num_stack = num_stack
        self.frames = collections.deque(maxlen=num_stack)

        low = np.concatenate([self.observation_space.low] * num_stack, axis=-1)
        high = np.concatenate([self.observation_space.high] * num_stack, axis=-1)
        self.observation_space = Box(low=low, high=high, dtype=self.observation_space.dtype)

    def get_observation(self):
        assert len(self.frames) == self.num_stack
        return np.concatenate(list(self.frames), axis=-1)

    def reset(self, **kwargs):
        ob, info = self.env.reset(**kwargs)
        for _ in range(self.num_stack):
            self.frames.append(ob)
        if 'goal' in info:
            info['goal'] = np.concatenate([info['goal']] * self.num_stack, axis=-1)
        return self.get_observation(), info

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        self.frames.append(observation)
        return self.get_observation(), reward, terminated, truncated, info


class NegativeRewardWrapper(gymnasium.Wrapper):
    """-1, 0 reward for gcrl"""

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return observation, reward - 1.0, terminated, truncated, info


def setup_egl():
    """Set up EGL for rendering."""
    if 'mac' in platform.platform():
        # macOS doesn't support EGL.
        pass
    else:
        os.environ['MUJOCO_GL'] = 'egl'
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
        if 'SLURM_STEP_GPUS' in os.environ:
            os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']


def make_env_and_datasets(
    dataset_name, frame_stack=None, ogbench_datadir=None, exorl_datadir=None, add_info=False, **env_kwargs
):
    """Make OGBench environment and datasets.

    Args:
        dataset_name: Name of the dataset.
        frame_stack: Number of frames to stack.

    Returns:
        A tuple of the environment, training dataset, and validation dataset.
    """

    if ogbench_datadir is not None:
        ogbench_datadir = os.path.expanduser(ogbench_datadir)
        ogbench.utils.DEFAULT_DATASET_DIR = ogbench_datadir

    if dataset_name.startswith('dmc_'):
        # expecting format dmc_<domain>-<expl_agent>-v0
        domain = dataset_name.split('-')[0][len('dmc_') :]
        dset_info = dataset_name.split('-')[1:-1]
        expl_agent = dset_info[0]
        num_samples = int(dset_info[1]) if len(dset_info) > 1 else 'all'
        if add_info:
            raise NotImplementedError('add_info not implemented for DMC')
        train_dataset = load_exorl_dataset(exorl_datadir, expl_agent, domain.replace('hybrid_', ''), num_samples)
        env = DMCAllTaskWrapper(domain, train_dataset, **env_kwargs)
        return env, train_dataset, None

    warnings.filterwarnings('ignore', category=UserWarning)
    env, train_dataset, val_dataset = ogbench.make_env_and_datasets(
        dataset_name, compact_dataset=True, add_info=add_info, **env_kwargs
    )
    warnings.filterwarnings('default', category=UserWarning)

    train_dataset = Dataset.create(**train_dataset)
    val_dataset = Dataset.create(**val_dataset) if val_dataset is not None else None

    if frame_stack is not None:
        env = FrameStackWrapper(env, frame_stack)

    env = NegativeRewardWrapper(env)

    env.reset()

    return env, train_dataset, val_dataset
