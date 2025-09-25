import typing as ty

import distrax
import jax


class USFMixin:
    def pi(self, observation: jax.Array, z: jax.Array, temperature: float = 1.0) -> distrax.Distribution:
        """pi(a | s, z)"""
        raise NotImplementedError

    def phi(self, observation: jax.Array):
        """phi(s)"""
        raise NotImplementedError

    def psi(self, z: jax.Array, observation: jax.Array, action: jax.Array | None = None):
        """psi_z(s, [a,])"""
        raise NotImplementedError

    def infer(self, task: tuple[jax.Array] | jax.Array) -> jax.Array:
        if task is None:
            return None

        if isinstance(task, tuple):
            obs, r = task
            if len(obs) == 0:
                return None
            phi = jax.vmap(self.phi)(obs)
            x, _, _, _ = jax.numpy.linalg.lstsq(phi, r)
            return x
        else:
            zr = self.phi(task)
            return zr / jax.numpy.linalg.norm(zr)

    @property
    def dim(self) -> int:
        raise NotImplementedError

    @property
    def gamma(self) -> int:
        raise NotImplementedError


class ModelMixin:
    def total_loss(self, batch, grad_params=None, rng=None):
        return 0.0, {}

    def update(self, batch):
        return self

    def finalize(self, big_batch):
        return self


state_t = ty.TypeVar('state_t')


class AgentMixin(ty.Generic[state_t]):
    def init(self, *, task: tuple[jax.Array] | jax.Array | None, observation: jax.Array, seed: jax.Array) -> state_t:
        return {}

    def act(
        self, *, state: state_t, observation: jax.Array, seed: jax.Array, temperature: float = 1.0
    ) -> tuple[jax.Array, dict[str, ty.Any]]:
        raise NotImplementedError

    def update(
        self,
        *,
        state: state_t,
        observation: jax.Array,
        action: jax.Array,
        reward: jax.Array,
        next_observation: jax.Array,
        truncated: jax.Array,
        terminated: jax.Array,
        seed: jax.Array,
    ) -> tuple[state_t, dict[str, ty.Any]]:
        return state, {}
