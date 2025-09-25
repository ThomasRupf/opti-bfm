import typing as ty

import flax
import jax
import jax.numpy as jnp

from utils.defs import AgentMixin, USFMixin
from utils.flax_utils import nonpytree_field
from utils.log_utils import register_cfg

register_cfg('oracle', dict(_target_='agents.basic.Oracle.create'), group='agent', package='agent')


class Oracle(flax.struct.PyTreeNode, AgentMixin[jax.Array]):
    bfm: USFMixin

    @classmethod
    def create(cls, bfm: USFMixin, *args, **kwargs):
        return cls(bfm, *args, **kwargs)

    def init(self, *, task, **kwargs) -> jax.Array:
        z = self.bfm.infer(task)
        assert z is not None, 'need task for Oracle'
        return z

    @jax.jit
    def act(self, *, state: jax.Array, observation, seed, temperature) -> tuple[jnp.ndarray, dict[str, ty.Any]]:
        z = state
        dist = self.bfm.pi(observation=observation, z=z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), {}


register_cfg('random', dict(_target_='agents.basic.Random.create', r=1), group='agent', package='agent')


class Random(flax.struct.PyTreeNode, AgentMixin[None]):
    bfm: USFMixin
    r: int = nonpytree_field()

    @classmethod
    def create(cls, bfm: USFMixin, *args, **kwargs):
        return cls(bfm, *args, **kwargs)

    def init(self, *, seed, **kwargs) -> tuple[jax.Array, jax.Array]:
        return jax.random.ball(seed, self.bfm.dim), 0

    @jax.jit
    def act(
        self, *, state: tuple[jax.Array, jax.Array], observation, seed, temperature
    ) -> tuple[jax.Array, dict[str, ty.Any]]:
        z, _ = state
        dist = self.bfm.pi(observation=observation, z=z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), {}

    @jax.jit
    def update(self, *, state: jax.Array, seed: jax.Array, **kwargs):
        z, i = state
        i = i + 1
        z_new = jax.random.ball(seed, self.bfm.dim)
        z = jax.lax.select(i % self.r == 0, z_new, z)
        return (z, i), {}
