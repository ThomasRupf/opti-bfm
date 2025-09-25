import os
import time
import typing as ty
from collections import defaultdict
from functools import partial

import imageio
import jax
import numpy as np
from tqdm.auto import trange

from utils.defs import AgentMixin
from utils.log_utils import ExpLog, flatten


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


_MEASURE_TIME = False


def evaluate(
    cfg,
    env,
    agent: AgentMixin[ty.Any],
    logger: ExpLog | None,
    task_id=None,
    log_dataset: bool = False,
    full_log: bool = False,
):
    # NOTE: for some reason gymnasium environments default randomness is independent of np.random.seed()
    # furthermore, when manually seeding them, the seed does not affect the action-space sampling at least
    # in the case of MujocoEnv this matters because ogbench uses random actions internally for some environments
    env.reset(seed=cfg.seed, options=dict(task_id=task_id))
    try:
        env.action_space.seed(cfg.seed)
    except Exception:
        pass

    task_infos = env.unwrapped.task_infos if hasattr(env.unwrapped, 'task_infos') else env.task_infos
    task_name = task_infos[task_id - 1]['task_name']

    s1, s2, s3 = jax.random.split(jax.random.key(cfg.seed), 3)

    init_fn = supply_rng(agent.init, rng=s1)
    act_fn = partial(supply_rng(agent.act, rng=s2), temperature=cfg.eval_temperature)
    update_fn = partial(supply_rng(agent.update, rng=s3))

    stats = defaultdict(list)

    act_times = []
    update_times = []

    for ii in trange(cfg.num_eval, unit='step', desc=f'eval {agent.__class__.__name__} on {task_name}'):
        should_render = ii < cfg.num_renders
        if should_render:
            path = os.path.join(cfg.working_dir, 'videos', task_name + '.mp4')
            os.makedirs(os.path.dirname(path), exist_ok=True)
            writer = imageio.get_writer(path, fps=20)

        mc_return = 0.0
        step = 0
        goal_frame = None

        for i in range(cfg.num_eval_episodes):
            observation, info = env.reset(options=dict(task_id=task_id, render_goal=should_render and i == 0))
            # print('\nobs_hash', ii, hashlib.md5(np.asarray(observation).tobytes()).hexdigest())

            if i == 0:
                # NOTE: this flag just controls whether task is provided at all
                # see zsrl flags for the individual agent classes to see whether they use it for more than logging
                task = info.get('task', info.get('goal')) if cfg.eval_with_task else None
                state = init_fn(task=task, observation=observation)

                goal_frame = info.get('goal_rendered')

            done = False
            while not done:
                action_time_start = time.time()
                action, alog = act_fn(state=state, observation=observation)
                action = np.clip(np.asarray(action), -1, 1)
                assert np.isfinite(action).all(), f'got {action}'
                if _MEASURE_TIME and i > 0:
                    act_times.append(time.time() - action_time_start)

                next_observation, reward, terminated, truncated, info = env.step(action)

                done = truncated or terminated

                if full_log:
                    step_log = dict(
                        observation=observation,
                        next_observation=next_observation,
                        action=action,
                        truncated=truncated,
                        terminated=terminated,
                    )
                else:
                    del info['physics']
                    step_log = dict()

                update_start_time = time.time()
                update_out = update_fn(
                    state=state,
                    observation=observation,
                    next_observation=next_observation,
                    action=action,
                    reward=reward,
                    truncated=truncated,
                    terminated=terminated,
                )
                if _MEASURE_TIME and i > 0:
                    # explicitly wait for end
                    update_out = jax.tree.map(
                        lambda x: x.block_until_ready() if isinstance(x, jax.Array) else x, update_out
                    )
                    update_times.append(time.time() - update_start_time)

                if len(update_out) == 2:
                    state, ulog = update_out
                else:
                    state, ulog, (add_to_dset, entry) = update_out
                    if log_dataset and add_to_dset:
                        logger.add(dict(dset=entry), prefix=ii)

                step += 1
                mc_return += reward

                info['return'] = mc_return
                info['avg_return'] = mc_return / step
                info['step'] = step

                if should_render and (step % cfg.video_frame_skip == 0 or done):
                    frame = env.render().copy()
                    if goal_frame is not None:
                        frame = np.concatenate([goal_frame, frame], axis=0)
                    writer.append_data(frame)

                if logger:
                    logger.add(
                        dict(
                            **step_log,
                            reward=reward,
                            info=info,
                            log={**alog, **ulog},
                        ),
                        prefix=ii,
                    )

                observation = next_observation  # NOTE: do not forget!

            if should_render:
                writer.close()

        add_to(stats, flatten(info))

    if _MEASURE_TIME:
        act_time = sum(act_times) / len(act_times)
        update_time = sum(update_times) / len(update_times)
        print('\n\n')
        print(f'act_time:    {act_time}s')
        print(f'update_time: {update_time}s')

    for k, v in stats.items():
        stats[k] = np.mean(v)

    print('\n\n', 'Avg. Return', stats['avg_return'], '\n\n')

    return stats
