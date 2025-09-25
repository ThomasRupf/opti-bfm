import functools
import glob
import os
import pickle
from typing import Any, Dict, Mapping, Sequence

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import optax

nonpytree_field = functools.partial(flax.struct.field, pytree_node=False)


class ModuleDict(nn.Module):
    """A dictionary of modules.

    This allows sharing parameters between modules and provides a convenient way to access them.

    Attributes:
        modules: Dictionary of modules.
    """

    modules: Dict[str, nn.Module]

    @nn.compact
    def __call__(self, *args, name=None, **kwargs):
        """Forward pass.

        For initialization, call with `name=None` and provide the arguments for each module in `kwargs`.
        Otherwise, call with `name=<module_name>` and provide the arguments for that module.
        """
        if name is None:
            if kwargs.keys() != self.modules.keys():
                raise ValueError(
                    f'When `name` is not specified, kwargs must contain the arguments for each module. '
                    f'Got kwargs keys {kwargs.keys()} but module keys {self.modules.keys()}'
                )
            out = {}
            for key, value in kwargs.items():
                if isinstance(value, Mapping):
                    out[key] = self.modules[key](**value)
                elif isinstance(value, Sequence):
                    out[key] = self.modules[key](*value)
                else:
                    out[key] = self.modules[key](value)
            return out

        return self.modules[name](*args, **kwargs)


class TrainState(flax.struct.PyTreeNode):
    """Custom train state for models.

    Attributes:
        step: Counter to keep track of the training steps. It is incremented by 1 after each `apply_gradients` call.
        apply_fn: Apply function of the model.
        model_def: Model definition.
        params: Parameters of the model.
        tx: optax optimizer.
        opt_state: Optimizer state.
    """

    step: int
    apply_fn: Any = nonpytree_field()
    model_def: Any = nonpytree_field()
    params: Any
    tx: Any = nonpytree_field()
    opt_state: Any

    @classmethod
    def create(cls, model_def, params, tx=None, **kwargs):
        """Create a new train state."""
        if tx is not None:
            opt_state = tx.init(params)
        else:
            opt_state = None

        return cls(
            step=1,
            apply_fn=model_def.apply,
            model_def=model_def,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )

    def __call__(self, *args, params=None, method=None, **kwargs):
        """Forward pass.

        When `params` is not provided, it uses the stored parameters.

        The typical use case is to set `params` to `None` when you want to *stop* the gradients, and to pass the current
        traced parameters when you want to flow the gradients. In other words, the default behavior is to stop the
        gradients, and you need to explicitly provide the parameters to flow the gradients.

        Args:
            *args: Arguments to pass to the model.
            params: Parameters to use for the forward pass. If `None`, it uses the stored parameters, without flowing
                the gradients.
            method: Method to call in the model. If `None`, it uses the default `apply` method.
            **kwargs: Keyword arguments to pass to the model.
        """
        if params is None:
            params = self.params
        variables = {'params': params}
        if method is not None:
            method_name = getattr(self.model_def, method)
        else:
            method_name = None

        return self.apply_fn(variables, *args, method=method_name, **kwargs)

    def select(self, name):
        """Helper function to select a module from a `ModuleDict`."""
        return functools.partial(self, name=name)

    def apply_gradients(self, grads, **kwargs):
        """Apply the gradients and return the updated state."""
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)

        info = {
            **{f'update/mean/{k}': jax.tree.reduce(jnp.add, jax.tree.map(jnp.mean, v)) for k, v in updates.items()},
            **{f'update/norm/{k}': optax.global_norm(v) for k, v in updates.items()},
            **{f'update/min/{k}': jax.tree.reduce(jnp.minimum, jax.tree.map(jnp.min, v)) for k, v in updates.items()},
            **{f'update/max/{k}': jax.tree.reduce(jnp.maximum, jax.tree.map(jnp.max, v)) for k, v in updates.items()},
        }

        new_params = optax.apply_updates(self.params, updates)

        return self.replace(
            step=self.step + 1,
            params=new_params,
            opt_state=new_opt_state,
            **kwargs,
        ), info

    def apply_loss_fn(self, loss_fn):
        """Apply the loss function and return the updated state and info.

        It additionally computes the gradient statistics and adds them to the dictionary.
        """
        grads, info = jax.grad(loss_fn, has_aux=True)(self.params)
        info = {f'info/{k}': v for k, v in info.items()}

        info.update(
            {
                **{f'grad/mean/{k}': jax.tree.reduce(jnp.add, jax.tree.map(jnp.mean, v)) for k, v in grads.items()},
                **{f'grad/max/{k}': jax.tree.reduce(jnp.maximum, jax.tree.map(jnp.max, v)) for k, v in grads.items()},
                **{f'grad/min/{k}': jax.tree.reduce(jnp.minimum, jax.tree.map(jnp.min, v)) for k, v in grads.items()},
                **{f'grad/norm/{k}': optax.global_norm(v) for k, v in grads.items()},
                **{
                    f'param/max/{k}': jax.tree.reduce(jnp.maximum, jax.tree.map(jnp.max, v))
                    for k, v in self.params.items()
                },
                **{
                    f'param/min/{k}': jax.tree.reduce(jnp.minimum, jax.tree.map(jnp.min, v))
                    for k, v in self.params.items()
                },
            }
        )
        if isinstance(self.opt_state[0], optax._src.transform.ScaleByAdamState):
            mu, nu = self.opt_state[0].mu, self.opt_state[0].nu
            info.update(
                {
                    f'grad/var/{k}': jax.tree.reduce(  # NOTE: this is only a variance proxy bc of different time scales
                        jnp.add, jax.tree.map(lambda o2, o1: jnp.sum(o2 - jnp.square(o1)), mu[k], nu[k])
                    )
                    for k in mu.keys()
                },
            )

        new_self, update_info = self.apply_gradients(grads=grads)
        info.update(update_info)

        return new_self, info


def save_agent(agent, save_dir, epoch):
    """Save the agent to a file.

    Args:
        agent: Agent.
        save_dir: Directory to save the agent.
        epoch: Epoch number.
    """

    save_dict = dict(
        agent=flax.serialization.to_state_dict(agent),
    )
    save_path = os.path.join(save_dir, f'params_{epoch}.pkl')
    with open(save_path, 'wb') as f:
        pickle.dump(save_dict, f)

    print(f'Saved to {save_path}')


def restore_agent(agent, restore_path, restore_epoch):
    """Restore the agent from a file.

    Args:
        agent: Agent.
        restore_path: Path to the directory containing the saved agent.
        restore_epoch: Epoch number.
    """
    candidates = glob.glob(restore_path)

    assert len(candidates) == 1, f'Found {len(candidates)} candidates: {candidates}'

    restore_path = candidates[0] + f'/params_{restore_epoch}.pkl'

    with open(restore_path, 'rb') as f:
        load_dict = pickle.load(f)
    sd_load = load_dict['agent']

    sd = flax.serialization.to_state_dict(agent)
    for k in sd.keys():
        if k not in sd_load:
            print(f"loaded state-dict does not have key `{k}` using the agent's `{k}`")
    sd_load = {k: sd_load.get(k, sd[k]) for k in sd.keys()}

    agent = flax.serialization.from_state_dict(agent, sd_load)

    print(f'Restored from {restore_path}')

    return agent
