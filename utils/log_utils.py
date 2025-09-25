import dataclasses
import os
import sys
import tempfile
import typing as ty
from datetime import datetime

import hydra
import numpy as np
import wandb
import yaml
import zarr
import zarr.codecs
import zarr.codecs.blosc
import zarr.storage
from hydra.core.config_store import ConfigStore
from omegaconf import MISSING, OmegaConf


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class CsvLogger:
    """CSV logger for logging metrics to a CSV file."""

    def __init__(self, path):
        self.path = path
        self.header = None
        self.file = None
        self.disallowed_types = (wandb.Image, wandb.Video, wandb.Histogram)

    def log(self, row, step):
        row['step'] = step
        if self.file is None:
            self.file = open(self.path, 'w')
            if self.header is None:
                self.header = [k for k, v in row.items() if not isinstance(v, self.disallowed_types)]
                self.file.write(','.join(self.header) + '\n')
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        else:
            filtered_row = {k: v for k, v in row.items() if not isinstance(v, self.disallowed_types)}
            self.file.write(','.join([str(filtered_row.get(k, '')) for k in self.header]) + '\n')
        self.file.flush()

    def close(self):
        if self.file is not None:
            self.file.close()


def get_exp_name(run_group: str, id: int) -> str:
    """Return the experiment name."""
    exp_name = ''
    exp_name += f'{datetime.now().strftime("%Y%m%d_%H%M%S")}'
    exp_name += f'_{run_group}_{id:04d}'
    if 'SLURM_JOB_ID' in os.environ:
        exp_name += f'_{os.environ["SLURM_JOB_ID"]}'
    return exp_name


def setup_wandb(
    entity=None,
    project='project',
    group=None,
    name=None,
    mode='online',
    config=None,
):
    """Set up Weights & Biases for logging."""
    wandb_output_dir = tempfile.mkdtemp()
    tags = [group] if group is not None else None

    init_kwargs = dict(
        project=project,
        entity=entity,
        tags=tags,
        group=group,
        dir=wandb_output_dir,
        config=config,
        name=name,
        settings=wandb.Settings(
            start_method='thread',
            _disable_stats=False,
        ),
        mode=mode,
        save_code=True,
    )

    run = wandb.init(**init_kwargs)

    return run


def reshape_video(v, n_cols=None):
    """Helper function to reshape videos."""
    if v.ndim == 4:
        v = v[None,]

    _, t, h, w, c = v.shape

    if n_cols is None:
        # Set n_cols to the square root of the number of videos.
        n_cols = np.ceil(np.sqrt(v.shape[0])).astype(int)
    if v.shape[0] % n_cols != 0:
        len_addition = n_cols - v.shape[0] % n_cols
        v = np.concatenate((v, np.zeros(shape=(len_addition, t, h, w, c))), axis=0)
    n_rows = v.shape[0] // n_cols

    v = np.reshape(v, (n_rows, n_cols, t, h, w, c))
    v = np.transpose(v, axes=(2, 5, 0, 3, 1, 4))
    v = np.reshape(v, (t, c, n_rows * h, n_cols * w))

    return v


# def get_wandb_video(renders=None, n_cols=None, fps=15):
#     # Pad videos to the same length.
#     max_length = max([len(render) for render in renders])
#     for i, render in enumerate(renders):
#         assert render.dtype == np.uint8

#         # Decrease brightness of the padded frames.
#         final_frame = render[-1]
#         final_image = Image.fromarray(final_frame)
#         enhancer = ImageEnhance.Brightness(final_image)
#         final_image = enhancer.enhance(0.5)
#         final_frame = np.array(final_image)

#         pad = np.repeat(final_frame[np.newaxis, ...], max_length - len(render), axis=0)
#         renders[i] = np.concatenate([render, pad], axis=0)

#         # Add borders.
#         renders[i] = np.pad(renders[i], ((0, 0), (1, 1), (1, 1), (0, 0)), mode='constant', constant_values=0)
#     renders = np.array(renders)  # (n, t, h, w, c)

#     renders = reshape_video(renders, n_cols)  # (t, c, nr * h, nc * w)

#     return wandb.Video(renders, fps=fps, format='mp4')


def register_cfg(name: str, cfg: ty.Any, group: str | None = None, package: str | None = None) -> None:
    # print(f'hydra: registering `{name}` as `{group}` (pkg=`{package}`)')
    ConfigStore.instance().store(name=name, node=cfg, provider='user', group=group, package=package)


def parse_input_and_init_hydra(config_name: str = 'base') -> ty.Any:
    config_paths = [e for e in sys.argv[1:] if e.endswith('.yaml') and os.path.isfile(e)]
    if len(config_paths) > 1:
        print('Found too many .yaml configs: ', config_paths)
        exit(1)
    elif len(config_paths) == 1:
        path = config_paths[0]
        sys.argv.remove(path)
        print(f"Loading '{path}'.")
        with open(path, 'r') as file:
            config = yaml.safe_load(file)

        def flatten(x, prefix=''):
            if isinstance(x, dict):
                return {kk: vv for k, v in x.items() for kk, vv in flatten(v, f'{prefix}{k}.').items()}
            return {prefix[:-1]: x}

        config = flatten(config)

        sys.argv = (
            sys.argv[:1]
            + [f'{k}={(v if v is not None else "null")}' for k, v in config.items() if v != '???']
            + sys.argv[1:]
        )

    hydra.initialize(version_base='1.3.2')
    cfg = hydra.compose(config_name=config_name, overrides=sys.argv[1:])
    OmegaConf.resolve(cfg)
    OmegaConf.set_readonly(cfg, True)
    return cfg


@dataclasses.dataclass
class BaseModelConfig:
    name: str = MISSING

    discount: float = 0.99
    batch_size: int = 1024
    discrete: bool = False
    encoder: str | None = None

    dataset_class: str = 'GCDataset'  # Dataset class name.
    value_p_curgoal: float = 1.0  # Probability of using the current state as the value goal.
    value_p_trajgoal: float = 0.0  # Probability of using a future state in the same trajectory as the value goal.
    value_p_randomgoal: float = 0.0  # Probability of using a random state as the value goal.
    value_geom_sample: bool = True  # Whether to use geometric sampling for future value goals.
    actor_p_curgoal: float = 1.0  # Probability of using the current state as the actor goal.
    actor_p_trajgoal: float = 0.0  # Probability of using a future state in the same trajectory as the actor goal.
    actor_p_randomgoal: float = 0.0  # Probability of using a random state as the actor goal.
    actor_geom_sample: bool = False  # Whether to use geometric sampling for future actor goals.
    gc_negative: bool = True  # Whether to use '0 if s == g else -1' (True) or '1 if s == g else 0' as reward.

    p_aug: float = 0.0  # Probability of applying image augmentation.
    frame_stack: ty.Optional[int] = None  # Number of frames to stack.


memspec_t = str


def _parse_memspec(s: memspec_t) -> int:
    """Parse a memory specification string into bytes."""
    s = s.upper()
    s = s.removesuffix('B')
    if s[-1:] == 'K':
        return int(s[:-1]) * 1024
    if s[-1:] == 'M':
        return int(s[:-1]) * 1024**2
    if s[-1:] == 'G':
        return int(s[:-1]) * 1024**3
    if s[-1:] == 'T':
        return int(s[:-1]) * 1024**4
    if s[-1] == 'B':
        return int(s[:-1])
    try:
        return int(s)
    except Exception as e:
        raise ValueError(f'unknown memspec {s}') from e


class BufferedArray:
    def __init__(self, buflen: int, item_shape: tuple[int, ...], dtype, arr: zarr.Array):
        if buflen == 1:
            self._buf = None
        else:
            self._buf = np.empty((buflen, *item_shape), dtype=dtype)
        self._ptr = 0
        self._arr = arr

    def append(self, data: np.ndarray):
        if self._buf is None:
            self._arr.append(np.expand_dims(data, axis=0), axis=0)
        else:
            assert data.shape == self._buf.shape[1:]
            self._buf[self._ptr] = data
            self._ptr += 1
            if self._ptr >= self._buf.shape[0]:
                self.flush()

    def flush(self):
        if self._buf is None or self._ptr == 0:
            return
        data = self._buf[0 : self._ptr]
        self._arr.append(data, axis=0)
        self._ptr = 0


class ExpLog:
    def __init__(
        self,
        path: str,
        chunk_size: memspec_t = '128KB',
        shard_size: memspec_t = '1MB',
        compression='light',
        verbose=False,
    ):
        self._store = zarr.storage.LocalStore(path)
        self._chunk_size = _parse_memspec(chunk_size)
        self._shard_size = _parse_memspec(shard_size)
        if compression is None or compression == 'none':
            self._compression = None
        elif compression == 'light':
            self._compression = zarr.codecs.BloscCodec(cname=zarr.codecs.blosc.BloscCname.lz4)
        elif compression == 'strong':
            self._compression = zarr.codecs.BloscCodec(cname=zarr.codecs.blosc.BloscCname.zstd, clevel=7)
        else:
            self._compression = compression
        self._verbose = verbose

        self._arrays: dict[str, BufferedArray] = dict()
        self._prefix = []
        self._add_guard = False

    def _register_array(self, name, shape: tuple[int, ...], dtype, **kwargs):
        """`zarr.create_array` wrapper with potential buffering for chunks.
        NOTE: `add` lazy-initializes arrays, you do not need to call this
        """
        item_bytes = int(np.prod(shape) * np.dtype(dtype).itemsize)
        chunklen = max(1, self._chunk_size // item_bytes)
        shardlen = max(1, ((self._shard_size // item_bytes) // chunklen) * chunklen)
        if self._verbose:
            print(f'registered {name} with shape {shape} and chunklen={chunklen}, shardlen={shardlen}')
        kwargs['shape'] = (0, *shape)
        kwargs.setdefault('chunks', (chunklen, *shape))
        kwargs.setdefault('shards', (shardlen, *shape))
        kwargs.setdefault('dtype', dtype)
        kwargs.setdefault('compressors', self._compression)
        arr = zarr.create_array(self._store, name=name, **kwargs)
        return BufferedArray(buflen=chunklen, item_shape=shape, dtype=dtype, arr=arr)

    def add_attr(self, k: str, v: ty.Any):
        zarr.open(self._store).attrs[k] = v

    def add_attrs(self, attrs: dict[str, ty.Any]):
        root = zarr.open(self._store)
        for k, v in attrs.items():
            root.attrs[k] = v

    def add(self, data: ty.Mapping[str, ty.Any], prefix: str | None = None):
        name = '/'.join(self._prefix + ([str(prefix)] if prefix is not None else []))
        self._add_guard = True
        self._add(data, name=name)
        self._add_guard = False

    def _add(self, data: ty.Any | ty.Mapping[str, ty.Any], name: str | None = None):
        if not self._add_guard:
            raise ValueError('do not call `_add` directly, use `add` instead!')
        if isinstance(data, (ty.Mapping, dict)):
            for k, v in data.items():
                new_name = k if name is None else name + '/' + str(k)
                self._add(v, name=new_name)
            return
        if not isinstance(data, np.ndarray):
            try:
                data = np.asarray(data)
            except Exception as e:
                raise ValueError(f'data for name {name} cannot be viewed as `np.ndarray`') from e
        if name not in self._arrays:
            assert name not in self._arrays.keys(), 'store/array missmatch'
            self._arrays[name] = self._register_array(name, data.shape, data.dtype)
        self._arrays[name].append(data)

    def close(self):
        for k, v in self._arrays.items():
            v.flush()
        if self._store._is_open:
            self._store.close()
        if self._verbose:
            try:
                print(zarr.open(self._store).tree())
            except Exception:
                pass

    def __del__(self):
        self.close()

    def __enter__(self):
        return self

    def __call__(self, prefix):
        self._prefix.append(str(prefix))
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._prefix:
            self._prefix.pop()
        else:
            self.close()
