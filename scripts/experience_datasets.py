import os

import numpy as np
import zarr
from omegaconf import OmegaConf
from tqdm.contrib.concurrent import process_map

DEST = 'datasets'


def generate(src):
    try:
        config = OmegaConf.load(src.replace('eval.zarr', 'config.yaml'))
        cfg = OmegaConf.load(src.replace('eval.zarr', 'cfg.yaml'))
    except Exception:
        raise ValueError('invalid src')
    root = zarr.open(src, mode='r')
    for task_name, task in root.groups():
        for trial_name, trial in task.groups():
            next_observations = np.asarray(trial['next_observation'][:])
            rewards = np.asarray(trial['reward'][:])
            env = cfg.env_name.split('-')[0][4:]
            task_name_short = task_name[len('task?_') :]
            dest = os.path.join(DEST, config.agent, str(cfg.seed), env, task_name_short, trial_name + '.npz')
            os.makedirs(os.path.dirname(dest), exist_ok=True)
            np.savez(dest, rewards=rewards, next_observations=next_observations)


_files = [
]
if len(_files) == 0:
    raise ValueError('add the paths to the eval.zarr directories you want to convert here!')

process_map(generate, _files)
os.system(f'tree {DEST}')
