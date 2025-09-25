import flax
import jax
import jax.numpy as jnp
import optax

from utils.blr import BLR
from utils.defs import AgentMixin, USFMixin
from utils.flax_utils import nonpytree_field
from utils.log_utils import register_cfg

register_cfg(
    'ucb',
    dict(
        _target_='agents.optimistic.LinUCB.create',
        zsrl=False,
        r=1,
        beta=1.0,
        n=128,
        lam=1.0,
        sample='C',
        decay=1.0,
        kappa=0.0,
    ),
    group='agent',
    package='agent',
)

state_t = tuple[BLR, jax.Array, jax.Array, jax.Array | None]


class LinUCB(flax.struct.PyTreeNode, AgentMixin[state_t]):
    bfm: USFMixin
    zsrl: bool = nonpytree_field()
    r: int = nonpytree_field()
    beta: float = nonpytree_field()
    n: int = nonpytree_field()
    lam: float = nonpytree_field()
    sample: str = nonpytree_field()
    decay: float = nonpytree_field()
    kappa: float = nonpytree_field()

    @classmethod
    def create(cls, *args, **kwargs) -> 'LinUCB':
        return cls(*args, **kwargs)

    def init(self, *, task, observation, seed) -> state_t:
        zgt = self.bfm.infer(task)
        C = BLR.create_LSQ(self.bfm.dim, self.lam)

        if self.zsrl:  # if zsrl, update LSQ with obs, reward pairs
            if isinstance(task, tuple):
                obs, rewards = task
            else:  # single goal
                obs, rewards = task[None, :], jnp.ones((1,))
            phis = jax.vmap(self.bfm.phi)(obs)
            C = C.update(X=phis, y=rewards)

        z = self.optimize_ucb(C, observation, seed)
        return (C, z, 0, zgt)

    @jax.jit
    def optimize_ucb(self, C: BLR, observation: jax.Array, seed):
        s1, s2 = jax.random.split(seed)
        if self.sample == 'Z':
            zs = jax.random.ball(s1, C.dim, 2, (self.n,))
        elif self.sample == 'C':
            zs = C.sample_ellipsoid(s1, (self.n,), 2.0 * self.beta)
        else:
            assert False
        sf = jax.vmap(self.bfm.psi, in_axes=(0, None))(zs, observation)
        assert sf.shape[0] == self.n and sf.shape[2] == self.bfm.dim
        vucb = C.ucb(sf, self.beta).min(1)  # pessimistic over sf ensemble
        ix = jnp.argmax(vucb + 1e-8 * jax.random.uniform(s2, vucb.shape))
        return zs[ix]

    @jax.jit
    def act(self, *, state: state_t, observation, seed, temperature):
        C, z, _, zgt = state
        mu = C.mean

        # d = z - mu
        log = dict(
            mu_norm=jnp.linalg.vector_norm(mu, axis=-1),
            max_cov_bound=C.cov_max_upper_bound,
            prec_log_det=C.prec_log_det,
            true_prec_log_det=C.true_prec_log_det,
            # max_prec_dist=C.prec_norm(d),
            # max_cov_dist=C.cov_norm(d),
            # max_l2_dist=jnp.linalg.vector_norm(d, axis=-1),
        )
        if zgt is not None:
            log['cos'] = optax.cosine_similarity(z, zgt)
            log['mu_cos'] = optax.cosine_similarity(mu, zgt)

        dist = self.bfm.pi(observation=observation, z=z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), log

    @jax.jit
    def update_C(self, C: BLR, next_observation, reward):
        phi = self.bfm.phi(next_observation)
        C_ = C.rank1_update(phi, reward, decay=self.decay)

        h = jnp.square(C.cov_norm(phi))
        aq = h >= jnp.exp(self.kappa) - 1
        C = jax.lax.cond(aq, lambda: C_, lambda: C)
        # NOTE that aq==True iff C = C_

        return C, dict(aq=aq, h=h), (aq, dict(next_observation=next_observation, reward=reward))

    def update(self, *, state: state_t, next_observation, reward, seed, **kwargs):
        C, z, i, zgt = state
        i = i + 1  # first action already happened at this point

        C, log, record = self.update_C(C, next_observation, reward)

        if i % self.r == 0:
            z = self.optimize_ucb(C, next_observation, seed)
        return (C, z, i, zgt), log, record


register_cfg(
    'ts',
    dict(_target_='agents.optimistic.TS.create', zsrl=False, r=1, sigma=0.01, decay=1.0, kappa=0.0),
    group='agent',
    package='agent',
)


class TS(flax.struct.PyTreeNode, AgentMixin[state_t]):
    bfm: USFMixin
    zsrl: bool = nonpytree_field()
    r: int = nonpytree_field()
    sigma: float = nonpytree_field()
    decay: float = nonpytree_field()
    kappa: float = nonpytree_field()

    @classmethod
    def create(cls, fb_agent: USFMixin, *args, **kwargs) -> 'LinUCB':
        return cls(fb_agent, *args, **kwargs)

    def init(self, *, task, observation, seed) -> state_t:
        zgt = self.bfm.infer(task)
        C = BLR.create(jnp.zeros((self.bfm.dim,)), jnp.diag(jnp.ones((self.bfm.dim,))))

        if self.zsrl:  # if zsrl, update BLR with obs, reward pairs
            if isinstance(task, tuple):
                obs, rewards = task
            else:  # single goal
                obs, rewards = task[None, :], jnp.ones((1,))
            phis = jax.vmap(self.bfm.phi)(obs)
            C = C.update(X=phis, y=rewards)

        state = (C, C.sample(seed), 0, zgt)
        return state

    @jax.jit
    def act(self, *, state: state_t, observation, seed, temperature):
        C, z, _, zgt = state
        mu = C.mean
        # d = z - mu
        log = dict(
            mu_norm=jnp.linalg.vector_norm(mu, axis=-1),
            max_cov_bound=C.cov_max_upper_bound,
            prec_log_det=C.prec_log_det,
            # max_prec_dist=C.prec_norm(d),
            # max_cov_dist=C.cov_norm(d),
            # max_l2_dist=jnp.linalg.vector_norm(d, axis=-1),
        )
        if zgt is not None:
            log['cos'] = optax.cosine_similarity(z, zgt)
            log['mu_cos'] = optax.cosine_similarity(mu, zgt)

        dist = self.bfm.pi(observation=observation, z=z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), log

    @jax.jit
    def update(self, *, state: state_t, next_observation, reward, seed, **kwargs):
        C, z, i, zgt = state
        i = i + 1  # first action already happened at this point

        phi = self.bfm.phi(next_observation)
        C_ = C.rank1_update(phi, reward, sigma=self.sigma, decay=self.decay)

        h = jnp.square(C.cov_norm(phi / self.sigma))
        aq = h >= jnp.exp(self.kappa) - 1
        C = jax.lax.cond(aq, lambda: C_, lambda: C)
        # NOTE that aq==True iff C = C_

        z_new = C.sample(seed)
        z = jax.lax.select(i % self.r == 0, z_new, z)

        return (C, z, i, zgt), dict(h=h, aq=aq), (aq, dict(next_observation=next_observation, reward=reward))
