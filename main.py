import os

# mujoco rendering
os.environ['MUJOCO_GL'] = 'egl'
os.environ['PYOPENGL_PLATFORM'] = 'egl'
if 'SLURM_STEP_GPUS' in os.environ:
    os.environ['EGL_DEVICE_ID'] = os.environ['SLURM_STEP_GPUS']

# flags for determinism
os.environ['XLA_cfg'] = '--xla_gpu_deterministic_ops=true --xla_gpu_autotune_level=0'  # noqa
os.environ['TF_DETERMINISTIC_OPS'] = '1'  # noqa
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'  # noqa

import dataclasses
import random
import time
import warnings
from collections import defaultdict
from typing import Any, Optional

import hydra
import jax
import numpy as np
import wandb
from omegaconf import MISSING, OmegaConf, read_write
from tqdm.auto import tqdm

import agents  # noqa -> make sure agent configs are registered
import models  # noqa -> make sure model configs are registered
from utils.datasets import Dataset, GCDataset, HGCDataset
from utils.defs import ModelMixin
from utils.env_utils import make_env_and_datasets
from utils.evaluation import evaluate
from utils.flax_utils import restore_agent, save_agent
from utils.log_utils import (
    BaseModelConfig,
    CsvLogger,
    ExpLog,
    get_exp_name,
    parse_input_and_init_hydra,
    register_cfg,
    setup_wandb,
)

jax.config.update('jax_default_matmul_precision', 'float32')


warnings.filterwarnings('ignore', '.*in1d.*deprecated.*')


@dataclasses.dataclass
class BaseConfig:
    # Hydra
    defaults: list[Any] = dataclasses.field(default_factory=lambda: ['_self_', dict(model='fb'), dict(agent='oracle')])

    model: Any = MISSING
    agent: Any = MISSING

    # Run settings
    run_group: str = 'test'  # Run group.
    seed: int | None = 0  # Random seed.
    env_name: str | None = None  # Environment (dataset) name.
    env_num_samples: int = 50_000
    env_sample_mode: str = 'default'

    # Checkpoint / data paths
    restore_path: Optional[str] = None  # Restore path.
    restore_epoch: Optional[int] = None  # Restore epoch.
    exorl_datadir: Optional[str] = None  # Path to the EXORL dataset.

    # Training loop
    train_steps: int = 2_000_000  # Number of training steps.
    log_interval: int = 1_000  # Logging interval.
    eval_interval: Optional[int] = 100_000  # Evaluation interval.
    save_interval: Optional[int] = None  # Saving interval.

    # Evaluation
    eval_with_task: bool = True  # provide task (for logging extra metrics).
    eval_task: Optional[int] = None  # Task number (None for all tasks).
    num_eval: int = 10  # Number of evals for each task.
    num_eval_episodes: int = 20  # Number of episode steps
    num_eval_steps: int = 1000  # Number of environment steps per eval
    eval_temperature: float = 0.0  # policy temperature during eval
    log_to_zarr: bool = True  # Log eval to zarr
    log_dataset: bool = False  # log dataset
    full_log: bool = False  # log all physics, observations, and actions

    num_renders: int = 0  # Number of videos for each task.
    video_frame_skip: int = 2  # Frame skip for videos.

    # I/O & logging
    project_dir: str = '.'  # Path to the code.
    results_dir: Optional[str] = None  # Mother directory of working_dir.
    working_dir: Optional[str] = None  # Save directory.
    wandb_mode: str = 'offline'  # Wandb mode.
    id: int = 0  # Clus ID.


register_cfg('base', BaseConfig)


def main(cfg: BaseConfig):
    exp_name = get_exp_name(cfg.run_group, cfg.id)

    with read_write(cfg):
        if cfg.results_dir is None:
            if cfg.working_dir is None:
                cfg.results_dir = os.path.join(os.getcwd(), 'exp')
            else:
                cfg.results_dir = os.path.dirname(os.path.abspath(os.path.expanduser(cfg.working_dir)))
        if cfg.working_dir is None:
            cfg.working_dir = os.path.join(cfg.results_dir, exp_name)
        cfg.project_dir = os.path.abspath(os.path.expanduser(cfg.project_dir))
        cfg.working_dir = os.path.abspath(os.path.expanduser(cfg.working_dir))
        cfg.results_dir = os.path.abspath(os.path.expanduser(cfg.results_dir))
    os.makedirs(cfg.results_dir, exist_ok=True)
    os.makedirs(cfg.working_dir, exist_ok=True)
    print(f'[CLUS-CLIENT] Working directory: {cfg.working_dir}')

    if cfg.restore_path is not None:
        with read_write(cfg):
            cfg.restore_path = os.path.abspath(os.path.expanduser(cfg.restore_path))
            cfg_path = os.path.join(cfg.restore_path, 'cfg.yaml')
            if not os.path.exists(cfg_path):  # If cfg.yaml does not exist, try config.yaml (legacy).
                print(cfg_path, 'does not exist, trying config.yaml')
                cfg_path = os.path.join(cfg.restore_path, 'config.yaml')
            restore_cfg: BaseConfig = OmegaConf.load(cfg_path)
            cfg.model = restore_cfg.model
            if cfg.env_name is None:
                cfg.env_name = restore_cfg.env_name
                print('overwriting env_name from restore_path:', cfg.env_name)
            if cfg.seed is None:
                cfg.seed = restore_cfg.seed
                print('setting seed from restore_path:', cfg.seed)

    # Set up logger.
    setup_wandb(
        project='opti-bfm',
        group=cfg.run_group,
        name=exp_name,
        mode=cfg.wandb_mode,
        config=OmegaConf.to_container(cfg, resolve=True),
    )

    OmegaConf.save(cfg, os.path.join(cfg.working_dir, 'cfg.yaml'))

    # Set up environment and dataset.
    model_cfg: BaseModelConfig = cfg.model

    assert cfg.seed is not None, 'specify seed if not restoring from path'
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    env_kwargs = dict(max_episode_steps=cfg.num_eval_steps)
    if cfg.env_name.startswith('dmc_'):
        env_kwargs.update(
            num_samples=cfg.env_num_samples,
            mode=cfg.env_sample_mode,
        )
    else:
        env_kwargs.update(
            add_noise_to_goal=False,  # make sure that the task stays exactly the same in ogbench
            terminate_at_goal=False,  # task is "holding" the goal, not reaching it
        )

    env, train_dataset, val_dataset = make_env_and_datasets(
        cfg.env_name, frame_stack=model_cfg.frame_stack, exorl_datadir=cfg.exorl_datadir, **env_kwargs
    )

    dataset_class = {'GCDataset': GCDataset, 'HGCDataset': HGCDataset}[model_cfg.dataset_class]
    train_dataset = dataset_class(Dataset.create(**train_dataset), model_cfg)
    if val_dataset is not None:
        val_dataset = dataset_class(Dataset.create(**val_dataset), model_cfg)

    example_batch = train_dataset.sample(1)
    if model_cfg.discrete:
        # Fill with the maximum action to let the agent know the action space size.
        example_batch['actions'] = np.full_like(example_batch['actions'], env.action_space.n - 1)

    model: ModelMixin = hydra.utils.get_class('models.' + model_cfg.name).create(
        cfg.seed,
        example_batch['observations'],
        example_batch['value_goals'],  # NOTE: if using oracle_reps, this is different from observations
        example_batch['actions'],
        model_cfg,
    )

    print(f'JAX default backend is {jax.default_backend()}')

    # Restore model.
    if cfg.restore_path is not None:
        model = restore_agent(model, cfg.restore_path, cfg.restore_epoch)

    # Train model.
    train_logger = CsvLogger(os.path.join(cfg.working_dir, 'train.csv'))
    eval_logger = CsvLogger(os.path.join(cfg.working_dir, 'eval.csv'))
    first_time = time.time()
    last_time = time.time()
    for i in tqdm(range(1, max(2, cfg.train_steps + 1)), smoothing=0.1, dynamic_ncols=True, desc='Training'):
        if cfg.train_steps > 0:
            # Update model.
            batch = train_dataset.sample(model_cfg.batch_size)
            model, update_info = model.update(batch)

            # Log metrics.
            if i % cfg.log_interval == 0:
                train_metrics = {f'training/{k}': v for k, v in update_info.items()}
                if val_dataset is not None:
                    val_batch = val_dataset.sample(model_cfg.batch_size)
                    _, val_info = model.total_loss(val_batch, grad_params=None)
                    train_metrics.update({f'validation/{k}': v for k, v in val_info.items()})
                train_metrics['time/epoch_time'] = (time.time() - last_time) / cfg.log_interval
                train_metrics['time/total_time'] = time.time() - first_time
                last_time = time.time()
                train_logger.log(train_metrics, step=i)
                train_metrics = {
                    k: wandb.Histogram(np_histogram=v) if k.endswith('HIST') else v for k, v in train_metrics.items()
                }
                wandb.log(train_metrics, step=i)

        # Evaluate agent.
        if cfg.eval_interval is not None and (i == 1 or i % cfg.eval_interval == 0):
            eval_model = model.finalize(train_dataset.sample(50_000))
            eval_agent = hydra.utils.instantiate(cfg.agent, eval_model)

            eval_metrics = {}
            overall_metrics = defaultdict(list)

            task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
            tasks = [cfg.eval_task] if cfg.eval_task is not None else list(range(1, len(task_infos) + 1))

            use_zarr_logger = i >= cfg.train_steps and cfg.log_to_zarr
            eval_path = os.path.join(cfg.working_dir, 'eval.zarr')
            logger = (
                ExpLog(eval_path, compression='strong', chunk_size='1MB', shard_size='32MB')
                if use_zarr_logger
                else None
            )

            for task_id in tqdm(tasks, smoothing=0.1, dynamic_ncols=True, desc='Evaluating'):
                task_name = task_infos[task_id - 1]['task_name']
                logger = logger(task_name) if logger else logger
                eval_info = evaluate(
                    cfg,
                    env=env,
                    agent=eval_agent,
                    logger=logger,
                    task_id=task_id,
                    log_dataset=cfg.log_dataset,
                    full_log=cfg.full_log,
                )
                logger.__exit__(None, None, None) if logger else ()
                metric_names = ['return', 'avg_return', 'success']
                eval_metrics.update(
                    {f'evaluation/{task_name}_{k}': v for k, v in eval_info.items() if k in metric_names}
                )
                for k, v in eval_info.items():
                    if k in metric_names:
                        overall_metrics[k].append(v)

            if use_zarr_logger:
                logger.close()

            for k, v in overall_metrics.items():
                eval_metrics[f'evaluation/overall_{k}'] = np.mean(v).item()

            wandb.log(eval_metrics, step=i)
            eval_logger.log(eval_metrics, step=i)

            del eval_agent

        # Save agent.
        if cfg.save_interval is not None and i % cfg.save_interval == 0:
            save_agent(model, cfg.working_dir, i)

    train_logger.close()
    eval_logger.close()
    wandb.finish()
    print(f'[CLUS-CLIENT] Working directory: {cfg.working_dir}')


if __name__ == '__main__':
    cfg = parse_input_and_init_hydra('base')
    main(cfg)
    print('[CLUS-CLIENT] done')
