import copy
import dataclasses
from functools import partial
from typing import Any

import distrax
import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

from utils.defs import AgentMixin, ModelMixin, USFMixin
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.log_utils import BaseModelConfig, register_cfg
from utils.networks import ensemblize


def norm_z(z, apply_norm=True):
    # this copies the torch.nn.functional.normalize used in the orignal FB exactly, including the clipping
    return (
        jnp.sqrt(z.shape[-1]) * (z / jnp.clip(jnp.linalg.norm(z, axis=-1, keepdims=True), min=1e-12))
        if apply_norm
        else z
    )


# def get_targets_uncertainty(
#     self, preds: torch.Tensor, pessimism_penalty: torch.Tensor | float
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     dim = 0
#     preds_mean = preds.mean(dim=dim)
#     preds_uns = preds.unsqueeze(dim=dim)  # 1 x n_parallel x ...
#     preds_uns2 = preds.unsqueeze(dim=dim + 1)  # n_parallel x 1 x ...
#     preds_diffs = torch.abs(preds_uns - preds_uns2)  # n_parallel x n_parallel x ...
#     num_parallel_scaling = preds.shape[dim] ** 2 - preds.shape[dim]
#     preds_unc = (
#         preds_diffs.sum(
#             dim=(dim, dim + 1),
#         )
#         / num_parallel_scaling
#     )
#     return preds_mean, preds_unc, preds_mean - pessimism_penalty * preds_unc
def compute_target_uncertainty(values, pessimism_penalty: float):
    n_ensembles, *rest = values.shape
    mean = jnp.mean(values, axis=0)
    diffs = jnp.abs(values[None, :, ...] - values[:, None, ...])
    unc = jnp.sum(diffs, axis=(0, 1)) / (n_ensembles**2 - n_ensembles)
    value = mean - pessimism_penalty * unc
    assert value.shape == tuple(rest), f'{value.shape} != {rest}'
    return value


# =============================================================================
# CUSTOM NETWORKS
# =============================================================================

# FB-CPR NETWORKS
# NOTE: since we can vmap networks in jax we can define all networks without the `num_parallel` parameter
# and just vmap the final network


# def weight_init(m):
#     if isinstance(m, nn.Linear):
#         nn.init.orthogonal_(m.weight.data)
#         if hasattr(m.bias, "data"):
#             m.bias.data.fill_(0.0)
#     elif isinstance(m, DenseParallel):
#         gain = nn.init.calculate_gain("relu")
#         parallel_orthogonal_(m.weight.data, gain)
#         if hasattr(m.bias, "data"):
#             m.bias.data.fill_(0.0)
#     elif hasattr(m, "reset_parameters"):
#         m.reset_parameters()
def linear(dim):  # input dim is inferred in jax
    return nn.Dense(dim, kernel_init=nn.initializers.orthogonal(jnp.sqrt(2.0)))


# def simple_embedding(input_dim, hidden_dim, hidden_layers, num_parallel=1):
#     assert hidden_layers >= 2, "must have at least 2 embedding layers"
#     seq = [linear(input_dim, hidden_dim, num_parallel), layernorm(hidden_dim, num_parallel), nn.Tanh()]
#     for _ in range(hidden_layers - 2):
#         seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
#     seq += [linear(hidden_dim, hidden_dim // 2, num_parallel), nn.ReLU()]
#     return nn.Sequential(*seq)
def simple_embedding(hidden_dim, hidden_layers):
    assert hidden_layers >= 2, 'must have at least 2 embedding layers'
    seq = [linear(hidden_dim), nn.LayerNorm(), nn.tanh]
    for _ in range(hidden_layers - 2):
        seq += [linear(hidden_dim), nn.relu]
    seq += [linear(hidden_dim // 2), nn.relu]
    return nn.Sequential(seq)


# class ForwardMap(nn.Module):
#     def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1,
#                  embedding_layers: int = 2, num_parallel: int = 2, output_dim=None) -> None:
#         super().__init__()
#         self.z_dim = z_dim
#         self.num_parallel = num_parallel
#         self.hidden_dim = hidden_dim
#
#         self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers, num_parallel)
#         self.embed_sa = simple_embedding(obs_dim + action_dim, hidden_dim, embedding_layers, num_parallel)
#
#         seq = []
#         for _ in range(hidden_layers):
#             seq += [linear(hidden_dim, hidden_dim, num_parallel), nn.ReLU()]
#         seq += [linear(hidden_dim, output_dim if output_dim else z_dim, num_parallel)]
#         self.Fs = nn.Sequential(*seq)
#
#     def forward(self, obs: torch.Tensor, z: torch.Tensor, action: torch.Tensor):
#         if self.num_parallel > 1:
#             obs = obs.expand(self.num_parallel, -1, -1)
#             z = z.expand(self.num_parallel, -1, -1)
#             action = action.expand(self.num_parallel, -1, -1)
#         z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # num_parallel x bs x h_dim // 2
#         sa_embedding = self.embed_sa(torch.cat([obs, action], dim=-1)) # num_parallel x bs x h_dim // 2
#         return self.Fs(torch.cat([sa_embedding, z_embedding], dim=-1))
class SingleForwardMap(nn.Module):
    hidden_dim: int
    hidden_layers: int
    embedding_layers: int
    z_dim: int

    @nn.compact
    def __call__(self, obs, z, action):
        z_embedding = simple_embedding(self.hidden_dim, self.embedding_layers)(jnp.concat([obs, z], axis=-1))
        sa_embedding = simple_embedding(self.hidden_dim, self.embedding_layers)(jnp.concat([obs, action], axis=-1))
        h = jnp.concat([sa_embedding, z_embedding], axis=-1)
        for _ in range(self.hidden_layers):
            h = linear(self.hidden_dim)(h)
            h = nn.relu(h)
        return linear(self.z_dim)(h)


class ForwardMap(nn.Module):
    hidden_dim: int
    hidden_layers: int
    embedding_layers: int
    z_dim: int
    num_parallel: int = 2

    @nn.compact
    def __call__(self, obs, z, action):
        return ensemblize(SingleForwardMap, self.num_parallel)(
            hidden_dim=self.hidden_dim,
            hidden_layers=self.hidden_layers,
            embedding_layers=self.embedding_layers,
            z_dim=self.z_dim,
        )(obs, z, action)


# class BackwardMap(nn.Module):
#     def __init__(self, goal_dim, z_dim, hidden_dim, hidden_layers: int = 2, norm=True) -> None:
#         super().__init__()
#         seq = [nn.Linear(goal_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Tanh()]
#         for _ in range(hidden_layers-1):
#             seq += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU()]
#         seq += [nn.Linear(hidden_dim, z_dim)]
#         if norm:
#             seq += [Norm()]
#         self.net = nn.Sequential(*seq)
#
#     def forward(self, x):
#         return self.net(x)
class BackwardMap(nn.Module):
    z_dim: int
    hidden_dim: int
    hidden_layers: int
    norm: bool

    @nn.compact
    def __call__(self, x):
        x = linear(self.hidden_dim)(x)
        x = nn.LayerNorm()(x)
        x = nn.tanh(x)
        for _ in range(self.hidden_layers - 1):
            x = linear(self.hidden_dim)(x)
            x = nn.relu(x)
        x = linear(self.z_dim)(x)
        x = norm_z(x, self.norm)
        return x


# Adopted from FB code
class TruncatedNormal(distrax.MultivariateNormalDiag):
    def __init__(self, loc, scale, clip) -> None:
        super().__init__(loc, jnp.full_like(loc, scale))
        self.clip = clip

    def _sample_n(self, key, n):  # overwrites distrax.MultivariateNormalDiag._sample_n
        eps = jax.random.normal(key, (n,) + self.loc.shape)
        eps *= self.scale_diag
        eps = jnp.clip(eps, -self.clip, self.clip)
        x = self.loc + eps
        return x


# class Actor(nn.Module):
#     def __init__(self, obs_dim, z_dim, action_dim, hidden_dim, hidden_layers: int = 1,
#                  embedding_layers: int = 2) -> None:
#         super().__init__()

#         self.embed_z = simple_embedding(obs_dim + z_dim, hidden_dim, embedding_layers)
#         self.embed_s = simple_embedding(obs_dim, hidden_dim, embedding_layers)


#         seq = []
#         for _ in range(hidden_layers):
#             seq += [linear(hidden_dim, hidden_dim), nn.ReLU()]
#         seq += [linear(hidden_dim, action_dim)]
#         self.policy = nn.Sequential(*seq)
#
#     def forward(self, obs, z, std):
#         z_embedding = self.embed_z(torch.cat([obs, z], dim=-1)) # bs x h_dim // 2
#         s_embedding = self.embed_s(obs) # bs x h_dim // 2
#         embedding = torch.cat([s_embedding, z_embedding], dim=-1)
#         mu = torch.tanh(self.policy(embedding))
#         std = torch.ones_like(mu) * std
#         dist = TruncatedNormal(mu, std)
#         return dist
class Actor(nn.Module):
    action_dim: int
    hidden_dim: int
    hidden_layers: int
    embedding_layers: int
    stddev: float
    stddev_clip: float

    @nn.compact
    def __call__(self, obs, z, temperature=1.0):
        z_embedding = simple_embedding(self.hidden_dim, self.embedding_layers)(jnp.concat([obs, z], axis=-1))
        s_embedding = simple_embedding(self.hidden_dim, self.embedding_layers)(obs)
        h = jnp.concat([s_embedding, z_embedding], axis=-1)
        for _ in range(self.hidden_layers):
            h = linear(self.hidden_dim)(h)
            h = nn.relu(h)
        h = linear(self.action_dim)(h)
        mu = nn.tanh(h)
        std = jnp.ones_like(mu) * self.stddev * temperature
        return TruncatedNormal(mu, std, self.stddev_clip)


# =============================================================================
# AGENT
# =============================================================================


@dataclasses.dataclass
class FBConfig(BaseModelConfig):
    """Configuration for the FB agent."""

    name: str = 'fb.FB'
    discount: float = 0.98
    batch_size: int = 1024

    z_dim: int = 50
    z_mix_ratio: float = 0.5
    norm_z: bool = True

    embedding_layers: int = 2
    forward_hidden_dim: int = 1024
    forward_hidden_layers: int = 1
    backward_hidden_dim: int = 256
    backward_hidden_layers: int = 2
    actor_hidden_dim: int = 1024
    actor_hidden_layers: int = 1

    stddev: float = 0.2
    stddev_clip: float = 0.3

    lr: float = 1e-4
    tau: float = 0.01

    bc_coef: float = 0.0  # for ogbench envs
    ortho_coef: float = 1.0
    m_penalty: float = 0.0
    q_penalty: float = 0.5


register_cfg('fb', FBConfig, group='model', package='model')


class FB(flax.struct.PyTreeNode, ModelMixin, USFMixin):
    rng: Any
    network: TrainState
    covb: jax.Array | None
    covb_inv: jax.Array | None
    config: FBConfig = nonpytree_field()

    def fb_loss(self, batch, grad_params, rng=None):
        """Compute forward-backward loss."""
        N = batch['z'].shape[0]
        I = jnp.eye(N)  # noqa
        Noff = N**2 - N
        Ioff = 1 - I

        # Compute target successor measure.
        tb = self.network.select('target_backward')(batch['next_observations'])
        dist = self.network.select('actor')(batch['next_observations'], batch['z'])
        action = jnp.clip(dist.sample(seed=rng), -1, 1)
        tf = self.network.select('target_forward')(batch['next_observations'], batch['z'], action)
        target_m = jnp.einsum('esd, td -> est', tf, tb)
        assert target_m.shape[0] == 2, 'dim 0 should be ensemble dim'
        target_m = compute_target_uncertainty(target_m, self.config.m_penalty)

        # Compute FB loss.
        f = self.network.select('forward')(batch['observations'], batch['z'], batch['actions'], params=grad_params)
        b = self.network.select('backward')(batch['next_observations'], params=grad_params)
        m = jnp.einsum('esd, td -> est', f, b)
        ne = m.shape[0]
        assert m.shape == (ne, self.config.batch_size, self.config.batch_size)

        diff = m - self.config['discount'] * target_m
        assert diff.shape == (ne, self.config.batch_size, self.config.batch_size)
        fb_offdiag = jnp.sum(jnp.square(diff * Ioff)) / (ne * Noff)
        fb_diag = -2 * jnp.sum(diff * I) / (ne * N)
        fb_loss = fb_offdiag + fb_diag

        # Orthonormality loss for backward embedding.
        cov = jnp.einsum('sd, td -> st', b, b)
        orth_loss_offdiag = jnp.sum(jnp.square(cov * Ioff)) / Noff
        orth_loss_diag = -2 * jnp.sum(cov * I) / N
        orth_loss = orth_loss_offdiag + orth_loss_diag

        total_loss = fb_loss + self.config['ortho_coef'] * orth_loss

        eye_diff = cov / b.shape[0] - jnp.eye(b.shape[0])
        return total_loss, {
            'target_m': target_m.mean(),
            'm_diag': jnp.sum(m * I) / (m.shape[0] * N),
            'm_offdiag': jnp.sum(m * Ioff) / (m.shape[0] * Noff),
            'f_norm': jnp.linalg.norm(f, axis=-1).mean(),
            'b_norm': jnp.linalg.norm(b, axis=-1).mean(),
            'fb_loss': fb_loss,
            'fb_loss_diag': fb_diag,
            'fb_loss_offdiag': fb_offdiag,
            'orth_loss': orth_loss,
            'orth_loss_diag': orth_loss_diag,
            'orth_loss_offdiag': orth_loss_offdiag,
            'orth_linf': jnp.max(jnp.abs(eye_diff)),
            'orth_l2': jnp.linalg.norm(eye_diff) / jnp.sqrt(self.config.z_dim),
        }

    def actor_loss(self, batch, grad_params, rng=None):
        assert not self.config.discrete
        dist: distrax.Distribution = self.network.select('actor')(batch['observations'], batch['z'], params=grad_params)
        q_actions = dist.sample(seed=rng)
        q_actions = q_actions - jax.lax.stop_gradient(q_actions - jnp.clip(q_actions, -1, 1))
        f = self.network.select('forward')(batch['observations'], batch['z'], q_actions)
        q = jnp.einsum('esd, sd -> es', f, batch['z'])
        assert q.shape == (q.shape[0], self.config.batch_size)
        q = compute_target_uncertainty(q, self.config.q_penalty)
        q_loss = jnp.mean(-q)
        if self.config.bc_coef > 0:
            actor_loss = q_loss / jnp.abs(q).mean() - self.config.bc_coef * dist.log_prob(batch['actions']).mean()
        else:
            actor_loss = q_loss
        return actor_loss, {'actor_loss': actor_loss, 'q_mean': q.mean(), 'q_abs_mean': jnp.abs(q).mean()}

    def sample_z(self, rng, batch_size=None):
        batch_size = batch_size if batch_size is not None else self.config.batch_size
        z = jax.random.normal(rng, (batch_size, self.config.z_dim))
        z = norm_z(z)
        if not self.config.norm_z:
            u = jax.random.uniform(rng, (self.config.batch_size, self.config.z_dim))
            z = u * z
        return z

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None):
        """Compute the total loss."""
        info = {}
        rng = rng if rng is not None else self.rng

        r1, r2, r3, r4, r5 = jax.random.split(rng, 5)
        z = self.sample_z(r1)
        goals_shuffled = jax.random.permutation(r2, batch['next_observations'])
        zb = self.network.select('backward')(goals_shuffled)
        mask = jax.random.bernoulli(r3, self.config['z_mix_ratio'], shape=(z.shape[0], 1))
        batch['z'] = jnp.where(mask, zb, z)

        fb_loss, fb_info = self.fb_loss(batch, grad_params, r4)
        info.update({f'fb/{k}': v for k, v in fb_info.items()})

        actor_loss, actor_info = self.actor_loss(batch, grad_params, r5)
        info.update({f'actor/{k}': v for k, v in actor_info.items()})

        loss = fb_loss + actor_loss
        return loss, info

    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config.tau + tp * (1 - self.config.tau),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch):
        """Update the agent and return a new agent with information dictionary."""
        train_rng, next_rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=train_rng)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'forward')
        self.target_update(new_network, 'backward')

        return self.replace(network=new_network, rng=next_rng), info

    @jax.jit
    def finalize(self, big_batch):
        obs = big_batch['observations']
        bs = self.network.select('backward')(obs)
        covb = jnp.einsum('bx, by -> xy', bs, bs) / bs.shape[0]
        covb_inv = jnp.linalg.inv(covb)
        return self.replace(covb=covb, covb_inv=covb_inv)

    # FB interface
    @partial(jax.jit, inline=True)
    def B(self, observation):
        return self.network.select('backward')(observation)

    @partial(jax.jit, inline=True)
    def F(self, z, observation, action=None):
        z = norm_z(z, self.config.norm_z)
        if action is None:
            dist = self.network.select('actor')(observation, z)
            action = jnp.clip(dist.mode(), -1, 1)
        return self.network.select('forward')(observation, z, action)

    @partial(jax.jit, inline=True)
    def pi_fb(self, observation, z, temperature=1.0):
        return self.network.select('actor')(observation, norm_z(z, self.config.norm_z), temperature=temperature)

    # USF Mixin Interface
    @partial(jax.jit, inline=True)
    def pi(self, observation, z, temperature=1.0):
        return self.pi_fb(observation=observation, z=z, temperature=temperature)

    @partial(jax.jit, inline=True)
    def phi(self, observation):
        return jnp.matvec(self.covb_inv, self.B(observation))

    @partial(jax.jit, inline=True)
    def psi(self, z, observation, action=None):
        return self.F(z=z, observation=observation, action=action)

    @property
    def dim(self):
        return self.config.z_dim

    @property
    def gamma(self):
        return self.config.discount

    @classmethod
    def create(cls, seed, ex_observations, ex_goals, ex_actions, config: FBConfig):
        """Create a new agent.

        Args:
            seed: Random seed.
            ex_observations: Example observations.
            ex_actions: Example batch of actions. In discrete-action MDPs, this should contain the maximum action value.
            config: Configuration dictionary.
        """

        rng = jax.random.key(seed)
        rng, init_rng = jax.random.split(rng, 2)

        ex_z = jnp.zeros((ex_goals.shape[0], config.z_dim))
        if config.discrete:
            action_dim = ex_actions.max() + 1
        else:
            action_dim = ex_actions.shape[-1]

        forward_def = ForwardMap(
            hidden_dim=config.forward_hidden_dim,
            hidden_layers=config.forward_hidden_layers,
            embedding_layers=config.embedding_layers,
            z_dim=config.z_dim,
        )
        backward_def = BackwardMap(
            z_dim=config.z_dim,
            hidden_dim=config.backward_hidden_dim,
            hidden_layers=config.backward_hidden_layers,
            norm=config.norm_z,
        )
        actor_def = Actor(
            action_dim=action_dim,
            hidden_dim=config.actor_hidden_dim,
            hidden_layers=config.actor_hidden_layers,
            embedding_layers=config.embedding_layers,
            stddev=config.stddev,
            stddev_clip=config.stddev_clip,
        )

        network_info = dict(
            forward=(forward_def, (ex_observations, ex_z, ex_actions)),
            backward=(backward_def, (ex_goals,)),
            target_forward=(copy.deepcopy(forward_def), (ex_observations, ex_z, ex_actions)),
            target_backward=(copy.deepcopy(backward_def), (ex_goals,)),
            actor=(actor_def, (ex_observations, ex_z)),
        )
        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.adam(learning_rate=config.lr)
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network_params
        params['modules_target_forward'] = params['modules_forward']
        params['modules_target_backward'] = params['modules_backward']

        return cls(rng, network=network, config=config, covb=None, covb_inv=None)


register_cfg('fb-oracle', dict(_target_='models.fb.FBOracle.create'), group='agent', package='agent')


class FBOracle(flax.struct.PyTreeNode, AgentMixin[jax.Array]):
    """as close as possible to normal FB inference, does not compute weight w of reward but FB z's directly"""

    fb: FB

    @classmethod
    def create(cls, bfm: FB):
        assert isinstance(bfm, FB)
        return cls(fb=bfm)

    @jax.jit
    def _zsrl_inference(self, obs, rs):
        bs = jax.vmap(self.fb.B)(obs)
        return jnp.einsum('bx, b -> x', bs, rs) / len(bs)

    def init(self, *, task, **kwargs) -> jax.Array:
        if isinstance(task, tuple):
            z = self._zsrl_inference(*task)
        else:  # single goal
            z = self.fb.B(task)
        return z

    @jax.jit
    def act(self, *, state: jax.Array, observation, seed, temperature) -> tuple[jnp.ndarray, dict[str, Any]]:
        log = {}
        z = state
        dist = self.fb.pi_fb(observation=observation, z=z, temperature=temperature)
        return jnp.clip(dist.sample(seed=seed), -1, 1), log
