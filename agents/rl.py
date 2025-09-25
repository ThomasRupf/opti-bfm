import typing as ty

import distrax
import flax
import jax
import jax.numpy as jnp
import optax

from utils.defs import AgentMixin, USFMixin
from utils.flax_utils import nonpytree_field
from utils.log_utils import register_cfg


class Buffer(ty.NamedTuple):
    """Jittable replay buffer"""

    dict: dict
    max_size: int
    pointer: int = 0

    @property
    def last(self):
        return self.pointer - 1

    @property
    def size(self):
        return self.pointer

    @classmethod
    def create(cls, transition, size):
        """Create a replay buffer from the example transition.

        Args:
            transition: Example transition (dict).
            size: Size of the replay buffer.
        """

        def create_buffer(example):
            example = jnp.array(example)
            return jnp.empty((size, *example.shape), dtype=example.dtype)

        buffer_dict = jax.tree.map(create_buffer, transition)
        return cls(buffer_dict, max_size=size)

    def add_transition(self, transition, pointer=None) -> 'Buffer':
        """Add a transition to the replay buffer."""
        pointer = pointer if pointer is not None else self.pointer
        return self._replace(
            dict={
                k: v.at[self.pointer].set(transition[k], mode='drop') if k in transition.keys() else v
                for k, v in self.dict.items()
            },
            pointer=self.pointer + 1,
        )

    def reset(self) -> 'Buffer':
        return self._replace(pointer=0)

    def __getitem__(self, key):
        return self.dict[key]


def normalize(vec: jax.Array):
    return vec * jax.lax.rsqrt(jnp.linalg.vecdot(vec, vec)[..., None])


def clip(vec: jax.Array):
    s = jax.lax.rsqrt(jnp.linalg.vecdot(vec, vec))
    return jax.lax.select(s < 1, vec * s, vec)


class PGState(ty.NamedTuple):
    n: int
    theta: jax.Array
    tx_state: optax.OptState
    buffer: Buffer
    z: jax.Array
    zgt: jax.Array | None


class PG_templ(flax.struct.PyTreeNode, AgentMixin[PGState]):
    bfm: USFMixin
    zsrl: bool = nonpytree_field()
    r: int = nonpytree_field()  # inference z update freq
    k: int = nonpytree_field()  # update freq & buffer size
    lr: float = nonpytree_field()  # learning rate
    std: float = nonpytree_field()  # std
    z_space: str = nonpytree_field()

    @classmethod
    def create(cls, fb_agent: USFMixin, *args, **kwargs):
        return cls(fb_agent, *args, **kwargs)

    @property
    def optim(self) -> optax.GradientTransformation:
        return optax.adam(self.lr)

    def init(self, *, task, observation, seed) -> PGState:
        zgt = self.bfm.infer(task)
        if self.zsrl:
            theta_init = normalize(zgt)
        elif self.z_space == 'sphere':
            theta_init = normalize(jax.random.normal(seed, (self.bfm.dim,)))
        else:
            theta_init = jnp.zeros((self.bfm.dim,))
        buffer = Buffer.create(
            dict(
                observation=observation,
                next_observation=observation,
                action=self.bfm.pi(observation, theta_init).mode(),  # dummy action
                reward=0.0,
                done=False,
                z=jnp.zeros_like(theta_init),
            ),
            size=self.k,
        )
        state = PGState(
            n=0,
            theta=theta_init,
            z=jnp.zeros_like(theta_init),
            tx_state=self.optim.init(theta_init),
            buffer=buffer,
            zgt=zgt,
        )
        state = self.sample_z(state, seed)  # set state.z
        return state

    def maybe_reproject(self, z):
        if self.z_space == 'Rd':
            pass
        elif self.z_space == 'ball':
            z = clip(z)
        elif self.z_space == 'sphere':
            z = normalize(z)
        else:
            assert False
        return z

    def compute_batch(self, buf: Buffer, param: jax.Array, zgt: jax.Array | None):
        raise NotImplementedError

    def loss(self, param, batch, key: jax.Array):
        raise NotImplementedError

    def optimize(self, buf: Buffer, params, tx_state, zgt, key):
        batch = self.compute_batch(buf, params, zgt)

        loss, grad = jax.value_and_grad(self.loss)(params, batch, key)

        updates, tx_state_new = self.optim.update(grad, tx_state, params=params)
        params_new = optax.apply_updates(params, updates)

        # reproject
        params_new = self.maybe_reproject(params_new)

        return (
            params_new,
            tx_state_new,
            dict(
                loss=loss,
                step=jnp.linalg.vector_norm(params - params_new),
                cos_step=optax.cosine_distance(params, params_new),
            ),
        )

    @jax.jit
    def act(self, *, state: PGState, observation, seed, temperature):
        log = {}
        if state.zgt is not None:
            log['cos'] = optax.cosine_similarity(state.z, state.zgt)
            log['mu_cos'] = optax.cosine_similarity(state.theta, state.zgt)
        dist = self.bfm.pi(observation=observation, z=state.z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), log

    @jax.jit
    def buffer_update(self, state: PGState, observation, action, reward, terminated, truncated, next_observation):
        return state._replace(
            buffer=state.buffer.add_transition(
                dict(
                    observation=observation,
                    action=action,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    next_observation=next_observation,
                    z=state.z,
                )
            )
        )

    @jax.jit
    def pg_step(self, state: PGState, seed):
        theta_, tx_state_, log = self.optimize(state.buffer, state.theta, state.tx_state, state.zgt, seed)
        return state._replace(theta=theta_, tx_state=tx_state_, buffer=state.buffer.reset()), log

    @jax.jit
    def sample_z(self, state: PGState, seed):
        z = state.theta + self.std * jax.random.normal(seed, state.theta.shape)
        z = self.maybe_reproject(z)
        return state._replace(z=z)

    def update(
        self, *, state: PGState, observation, action, reward, terminated, truncated, next_observation, seed, **kwargs
    ) -> tuple[dict[str, jax.Array], dict[str, ty.Any]]:
        s1, s2 = jax.random.split(seed)

        state = self.buffer_update(state, observation, action, reward, terminated, truncated, next_observation)

        # cannot jit whole update bc then this always runs pg_step
        log = dict(loss=jnp.nan, step=jnp.nan, cos_step=jnp.nan)
        if state.buffer.size == self.k:
            state, log = self.pg_step(state, s1)

        state = state._replace(n=state.n + 1)
        if state.n % self.r == 0:
            state = self.sample_z(state, s2)

        return state, dict(mu_norm=jnp.linalg.vector_norm(state.theta), max_cov_bound=self.std, **log)


register_cfg(
    'lola',
    dict(_target_='agents.rl.LoLA.create', zsrl=False, r=250, k=2000, lr=1.0, std=0.2, z_space='sphere', zr_proxy='mu'),
    group='agent',
    package='agent',
)


class LoLA(PG_templ):
    zr_proxy: str = nonpytree_field()

    @classmethod
    def create(cls, fb_agent: USFMixin, **kwargs):
        return cls(fb_agent, **kwargs)

    def compute_batch(self, buf: Buffer, param: jax.Array, zgt: jax.Array | None):
        B = self.k // self.r

        zs, rs, no = jax.tree.map(
            lambda x: jnp.reshape(x, (-1, self.r, *x.shape[1:])), (buf['z'], buf['reward'], buf['next_observation'])
        )  # (batch x time x *shape)

        z = zs[:, 0]
        assert z.shape == (B, self.bfm.dim)

        zr = z if self.zr_proxy == 'z' else param
        zr = normalize(zgt if self.zsrl else zr)  # LoLA takes zgt, we take best estimate if not available

        sfs = jax.vmap(self.bfm.psi)(z=z, observation=no[:, -1])
        assert sfs.shape == (B, 2, self.bfm.dim)
        vs = jnp.linalg.vecdot(sfs, zr[..., None, :])
        assert vs.shape == (B, 2)
        vs = vs.min(-1)  # pessimistic over psi ensemble

        gammas = self.bfm.gamma ** jnp.arange(self.r + 1)
        x = jnp.concat((rs, vs[:, None]), axis=1)
        assert x.shape == (B, self.r + 1)
        R = jnp.vecdot(gammas, x)

        # leave-one-out baseline
        adv = (R.shape[0] * R - jnp.sum(R)) / (R.shape[0] - 1)
        assert adv.shape == (B,)

        return (z, adv)

    def loss(self, param, batch, _key):
        z, adv = batch
        dist = distrax.MultivariateNormalDiag(param, jnp.full_like(param, self.std))
        return jnp.mean(adv * -dist.log_prob(z))
