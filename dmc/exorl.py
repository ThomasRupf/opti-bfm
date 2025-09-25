import os
from pathlib import Path

import numpy as np
from tqdm import tqdm

_CACHE = True

DEFAULT_DATASET_PATH = os.path.expanduser('~/.exorl')


def load_exorl_dataset(dataset_path, expl_agent, domain_name, num_episodes='all') -> dict:
    dataset_path = dataset_path or DEFAULT_DATASET_PATH
    path = Path(dataset_path) / f'{domain_name}/{expl_agent}/buffer'
    path = path.expanduser()
    if _CACHE and num_episodes == 'all':
        cache_path = Path(dataset_path) / f'{domain_name}/{expl_agent}/dataset.npz'
        cache_path = cache_path.expanduser()
        if cache_path.exists():
            return np.load(cache_path)

    print(f'Data path: {path}')
    files = sorted(list(path.glob('*.npz')))
    if len(files) == 0:
        raise FileNotFoundError(f'No files found in {path}')

    if num_episodes == 'all':
        num_episodes = len(files)
    else:
        num_episodes = min(num_episodes, len(files))

    shapes = {}
    dtypes = {}
    data = np.load(str(files[0]))
    for k in ['observation', 'action', 'physics']:
        shapes[k] = data[k].shape[1:]
        dtypes[k] = data[k].dtype

    n = sum(int(str(files[i]).split('_')[-1].removesuffix('.npz')) + 1 for i in range(num_episodes))

    storage = {
        'observations': np.empty((n, *shapes['observation']), dtype=np.float32),
        'actions': np.empty((n, *shapes['action']), dtype=np.float32),
        'terminals': np.empty((n,), dtype=np.float32),
        'valids': np.empty((n,), dtype=np.float32),
        'physics': np.empty((n, *shapes['physics']), dtype=dtypes['physics']),
    }
    ptr = 0

    # Compact dataset: We need to invalidate the last state of each trajectory so that we can safely get
    # `next_observations[t]` by using `observations[t + 1]`.
    # Our goal is to have the following structure:
    #                  |<--- traj 1 --->|  |<--- traj 2 --->|  ...
    # -------------------------------------------------------------
    # 'observations': [s0, s1, s2, s3, s4, s0, s1, s2, s3, s4, ...]
    # 'actions'     : [a0, a1, a2, a3, a4, a0, a1, a2, a3, a4, ...]
    # 'terminals'   : [ 0,  0,  0,  1,  1,  0,  0,  0,  1,  1, ...]
    # 'valids'      : [ 1,  1,  1,  1,  0,  1,  1,  1,  1,  0, ...]

    for i in tqdm(range(num_episodes), desc='loading dataset'):
        f = files[i]
        data = np.load(str(f))
        n = data['observation'].shape[0]
        storage['observations'][ptr : ptr + n] = data['observation'].astype(np.float32)
        storage['actions'][ptr : ptr + n - 1] = data['action'][1:].astype(np.float32)
        storage['terminals'][ptr : ptr + n] = 0.0
        storage['terminals'][max(ptr, ptr + n - 2) : ptr + n] = 1.0  # last 2 are set (see above)
        storage['valids'][ptr : ptr + n] = 1.0
        storage['valids'][ptr + n - 1] = 0.0  # not valid since we do not know action here
        storage['physics'][ptr : ptr + n] = data['physics']
        ptr += n

    if _CACHE:
        np.savez(cache_path, **storage)

    return storage
