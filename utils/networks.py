from typing import Any, Optional, Sequence

import distrax
import flax.linen as nn
import jax.numpy as jnp
import jax
from flax.linen.initializers import constant, orthogonal


def default_init(scale=1.0):
    return nn.initializers.variance_scaling(scale, 'fan_avg', 'uniform')


def ensemblize(cls, num_qs, in_axes=None, out_axes=0, **kwargs):
    return nn.vmap(
        cls,
        variable_axes={'params': 0, 'intermediates': 0},
        split_rngs={'params': True},
        in_axes=in_axes,
        out_axes=out_axes,
        axis_size=num_qs,
        **kwargs,
    )

class Identity(nn.Module):
    def __call__(self, x):
        return x

class LogParam(nn.Module):
    init_value: float = 1.0
    @nn.compact
    def __call__(self):
        log_value = self.param('log_value', init_fn=lambda key: jnp.full((), jnp.log(self.init_value)))
        return jnp.exp(log_value)


class TransformedWithMode(distrax.Transformed):
    def mode(self):
        return self.bijector.forward(self.distribution.mode())

class MLP(nn.Module):
    hidden_dims: Sequence[int]
    activations: Any = nn.gelu
    activate_final: bool = False
    kernel_init: Any = default_init()
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        for i, size in enumerate(self.hidden_dims):
            x = nn.Dense(size, kernel_init=self.kernel_init)(x)
            if i + 1 < len(self.hidden_dims) or self.activate_final:
                x = self.activations(x)
                if self.layer_norm:
                    x = nn.LayerNorm()(x)
            if i == len(self.hidden_dims) - 2:
                self.sow('intermediates', 'feature', x)
        return x

class Actor(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    log_std_min: Optional[float] = -5
    log_std_max: Optional[float] = 2
    tanh_squash: bool = False
    state_dependent_std: bool = False
    const_std: bool = True
    final_fc_init_scale: float = 1e-2
    encoder: nn.Module = None

    def setup(self):
        self.actor_net = MLP(self.hidden_dims, activate_final=True, layer_norm=self.layer_norm)
        self.mean_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        if self.state_dependent_std:
            self.log_std_net = nn.Dense(self.action_dim, kernel_init=default_init(self.final_fc_init_scale))
        else:
            if not self.const_std:
                self.log_stds = self.param('log_stds', nn.initializers.zeros, (self.action_dim,))

    def __call__(
        self,
        observations,
        temperature=1.0,
        info=False,
        dual_type=None
    ):
        if self.encoder is not None:
            inputs = self.encoder(observations)
        else:
            inputs = observations
        outputs = self.actor_net(inputs)

        means = self.mean_net(outputs)
        if self.state_dependent_std:
            log_stds = self.log_std_net(outputs)
        else:
            if self.const_std:
                log_stds = jnp.zeros_like(means)
            else:
                log_stds = self.log_stds

        log_stds = jnp.clip(log_stds, self.log_std_min, self.log_std_max)

        distribution = distrax.MultivariateNormalDiag(loc=means, scale_diag=jnp.exp(log_stds) * temperature)
        if self.tanh_squash:
            distribution = TransformedWithMode(distribution, distrax.Block(distrax.Tanh(), ndims=1))

        if info:
            if dual_type == 'scalar':
                return distribution, self.log_lam
            else:
                return distribution
        else:
            return distribution

class ActorVectorField(nn.Module):
    hidden_dims: Sequence[int]
    action_dim: int
    layer_norm: bool = False
    encoder: nn.Module = None

    def setup(self) -> None:
        self.mlp = MLP((*self.hidden_dims, self.action_dim), activate_final=False, layer_norm=self.layer_norm)

    @nn.compact
    def __call__(self, observations, actions, times=None, is_encoded=False):
        if not is_encoded and self.encoder is not None:
            observations = self.encoder(observations)
        if times is None:
            inputs = jnp.concatenate([observations, actions], axis=-1)
        else:
            inputs = jnp.concatenate([observations, actions, times], axis=-1)

        v = self.mlp(inputs)

        return v

class Value(nn.Module):
    hidden_dims: Sequence[int]
    layer_norm: bool = True
    num_ensembles: int = 2
    encoder: nn.Module = None

    def setup(self):
        mlp_class = MLP
        if self.num_ensembles > 1:
            mlp_class = ensemblize(mlp_class, self.num_ensembles)
        value_net = mlp_class((*self.hidden_dims, 1), activate_final=False, layer_norm=self.layer_norm)

        self.value_net = value_net

    def __call__(self, observations, actions=None):
        if self.encoder is not None:
            inputs = [self.encoder(observations)]
        else:
            inputs = [observations]
        if actions is not None:
            inputs.append(actions)
        inputs = jnp.concatenate(inputs, axis=-1)

        v = self.value_net(inputs).squeeze(-1)

        return v


class HyperNetwork(nn.Module):
    hidden_dim: int
    output_dim: int
    init_scale: float
    layer_norm: bool = False

    @nn.compact
    def __call__(self, x):
        x = nn.Dense(
            self.hidden_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        if self.layer_norm:
            x = nn.LayerNorm()(x)
        x = nn.relu(x)
        x = nn.Dense(
            self.output_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(x)
        return x

class Mixer(nn.Module):
    embedding_dim: int
    hypernet_hidden_dim: int
    init_scale: float = 1.0
    layer_norm: bool = False

    @nn.compact
    def __call__(self, q_vals, states):
        if q_vals.ndim != 3 or states.ndim != 3:
            raise ValueError(
                f"Mixer expects q_vals (n_agents, time_steps, batch_size) and states (time_steps, batch_size, state_dim), "
                f"got q_vals {q_vals.shape}, states {states.shape}"
            )

        time_steps, batch_size, n_agents = q_vals.shape

        # Hypernetworks to produce mixing weights and biases conditioned on state
        w_1 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim * n_agents,
            init_scale=self.init_scale,
            layer_norm=self.layer_norm,
        )(states)
        b_1 = nn.Dense(
            self.embedding_dim,
            kernel_init=orthogonal(self.init_scale),
            bias_init=constant(0.0),
        )(states)
        w_2 = HyperNetwork(
            hidden_dim=self.hypernet_hidden_dim,
            output_dim=self.embedding_dim,
            init_scale=self.init_scale,
            layer_norm=self.layer_norm,
        )(states)
        b_2 = HyperNetwork(
            hidden_dim=self.embedding_dim,
            output_dim=1,
            init_scale=self.init_scale,
            layer_norm=self.layer_norm,
        )(states)

        # Enforce monotonicity with non-negative weights and reshape
        w_1 = jnp.abs(w_1.reshape(time_steps, batch_size, n_agents, self.embedding_dim))
        b_1 = b_1.reshape(time_steps, batch_size, 1, self.embedding_dim)
        w_2 = jnp.abs(w_2.reshape(time_steps, batch_size, self.embedding_dim, 1))
        b_2 = b_2.reshape(time_steps, batch_size, 1, 1)

        hidden = nn.elu(jnp.matmul(q_vals[:, :, None, :], w_1) + b_1)
        q_tot = jnp.matmul(hidden, w_2) + b_2

        return q_tot.squeeze()


class QMixerCentralFF(nn.Module):
    embed_dim: int

    @nn.compact
    def __call__(self, agent_qs_flat, states_flat):
        x = jnp.concatenate([states_flat, agent_qs_flat], axis=-1)
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        advs = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)

        v = nn.Dense(self.embed_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(states_flat)
        v = nn.relu(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)

        return advs + v


class ShapelyMixer(nn.Module):
    n_agents: int
    n_actions: int
    sample_size: int
    embed_dim: int

    @nn.compact
    def __call__(self, states, agent_qs, rng):
        """Estimate Shapley values per agent.

        Expects time-major shapes:
        - states: (T, B, S)
        - agent_qs: (T, B, A, N) where A = n_actions, N = n_agents
        Returns shapley values per agent: (T, B, N)
        """
        if agent_qs.ndim != 4:
            raise ValueError("agent_qs must be (T, B, A, N)")
        if states.ndim < 3:
            raise ValueError("states must be (T, B, ...) — include T and B dims")

        time_steps, batch_size, n_actions, n_agents = agent_qs.shape
        if n_agents != self.n_agents or n_actions != self.n_actions:
            raise ValueError(
                f"ShapelyMixer configured for (n_agents={self.n_agents}, n_actions={self.n_actions}), "
                f"got (n_agents={n_agents}, n_actions={n_actions})"
            )

        B_tot = time_steps * batch_size
        state_flat = states.reshape(B_tot, -1)
        # (T, B, N, A)
        agent_qs_tbna = jnp.transpose(agent_qs, (0, 1, 3, 2))
        agent_qs_flat = agent_qs_tbna.reshape(B_tot, n_agents, n_actions)  # (B_tot, N, A)

        rand = jax.random.uniform(rng, shape=(B_tot, self.sample_size, n_agents))
        p2a = jnp.argsort(rand, axis=-1)  # position -> agent id
        a2p = jnp.argsort(p2a, axis=-1)   # agent -> position

        individual_map = jax.nn.one_hot(a2p, n_agents).astype(jnp.float32)  # (B_tot, S, N, N)
        seq_set = jnp.tril(jnp.ones((n_agents, n_agents), dtype=jnp.float32))
        subcoalition_map = jnp.matmul(individual_map, seq_set)
        subcoalition_map_no_i = subcoalition_map - individual_map

        def reorder_single(b_qs, perms):
            def reorder_one(p):
                return jnp.take(b_qs, p, axis=0)
            return jax.vmap(reorder_one)(perms)

        agent_qs_reordered = jax.vmap(reorder_single)(agent_qs_flat, p2a)  # (B_tot, S, N, A)

        aqr = agent_qs_reordered[:, :, None, :, :]  # (B_tot, S, 1, N, A)
        m_no_i = subcoalition_map_no_i[:, :, :, :, None]
        m_i = individual_map[:, :, :, :, None]
        agent_qs_coalition_no_i = aqr * m_no_i
        agent_qs_coalition_i = aqr * m_i
        agent_qs_coalition = jax.lax.stop_gradient(agent_qs_coalition_no_i) + agent_qs_coalition_i

        coalition_flat = agent_qs_coalition.reshape(B_tot * self.sample_size * n_agents, n_agents * n_actions)
        states_tiled = jnp.tile(state_flat[:, None, None, :], (1, self.sample_size, n_agents, 1))
        states_flat = states_tiled.reshape(B_tot * self.sample_size * n_agents, -1)

        mc = QMixerCentralFF(self.embed_dim)
        contribs = mc(coalition_flat, states_flat).reshape(B_tot, self.sample_size, n_agents)
        shapley_values = contribs.mean(axis=1)

        shapley_values = shapley_values.reshape(time_steps, batch_size, n_agents)
        return shapley_values

class SequenceLSTMEncoder(nn.Module):
    """Sequence encoder with optional per-step encoder and stacked LSTM.

    Inputs are expected to be time-major: (T, B, N, ...), but any leading batch
    shape is supported as long as time is the first dimension.

    - Optionally applies a `point_encoder` to each timestep independently
      (e.g., CNN/MLP) before the recurrent layers.
    - Stacked LSTM(s) then process features over time with reset masking.
    - Returns per-timestep hidden states of the top LSTM layer with the same
      leading shape as inputs, replacing the feature dimension.
    """

    hidden_dim: int
    num_layers: int = 1
    pre_mlp_dims: Sequence[int] = ()
    layer_norm: bool = False
    point_encoder: Optional[nn.Module] = None

    @nn.compact
    def __call__(self, observations, resets: Optional[jnp.ndarray] = None,
                 initial_carry: Optional[Sequence[Any]] = None,
                 return_carry: bool = False):
        # observations: (T, B, N, F...) -> flatten trailing feature dims if needed
        # Accept arbitrary leading dims after T; combine them to an effective batch M
        x = observations
        # Flatten any spatial dims before MLP/LSTM; apply point encoder first if given
        if self.point_encoder is not None:
            # point_encoder should broadcast over leading dims automatically
            x = self.point_encoder(x)
        # Ensure last dim is feature
        x = x.reshape(x.shape[: -1] + (-1,)) if x.ndim >= 2 else x

        T = x.shape[0]
        leading_shape = x.shape[1:-1]  # e.g., (B, N)
        # compute M using Python ints from shape, avoiding JAX tracers
        M = 1
        if len(leading_shape) > 0:
            for d in leading_shape:
                M *= d
        feat_dim = x.shape[-1]

        x_flat = x.reshape((T, M, feat_dim))

        if self.pre_mlp_dims:
            x_flat = MLP(self.pre_mlp_dims, activate_final=True, layer_norm=self.layer_norm)(x_flat)

        # Simple, version-agnostic zero carry with batch dim M
        def _zero_carry(batch_size: int, hidden: int, dtype):
            c = jnp.zeros((batch_size, hidden), dtype)
            h = jnp.zeros((batch_size, hidden), dtype)
            return (c, h)

        # Wrapper cell that applies reset mask before calling LSTMCell
        class _ResetLSTMCell(nn.Module):
            hidden_dim: int
            @nn.compact
            def __call__(self, carry, inputs):
                x_t, reset_t = inputs  # (M, D), (M,)
                lstm = nn.OptimizedLSTMCell(self.hidden_dim)

                # Support both tuple and object state
                def _get_c_h(s):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        return s.c, s.h
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return s[0], s[1]
                    raise TypeError(f"Unrecognized LSTM state structure: {type(s)}")

                def _make_state_like(s, c, h):
                    if hasattr(s, 'c') and hasattr(s, 'h'):
                        cls = type(s)
                        try:
                            return cls(c=c, h=h)
                        except TypeError:
                            try:
                                return cls(c, h)
                            except TypeError:
                                return (c, h)
                    if isinstance(s, (tuple, list)) and len(s) == 2:
                        return (c, h)
                    return (c, h)

                c, h = _get_c_h(carry)
                c = jnp.where(reset_t[:, None], jnp.zeros_like(c), c)
                h = jnp.where(reset_t[:, None], jnp.zeros_like(h), h)
                carry = _make_state_like(carry, c, h)

                new_carry, y = lstm(carry, x_t)
                return new_carry, y

        if resets is None:
            resets_flat = jnp.zeros((T, M), dtype=bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)
        else:
            assert resets.shape[: len(leading_shape) + 1] == (T, *leading_shape), (
                f"resets shape should be (T, *leading_shape), got {resets.shape}, expected {(T, *leading_shape)}"
            )
            resets_flat = resets.reshape((T, M)).astype(bool)
            if initial_carry is None:
                resets_flat = resets_flat.at[0, :].set(True)

        # Prepare initial carries
        if initial_carry is None:
            carries = tuple(_zero_carry(M, self.hidden_dim, x_flat.dtype) for _ in range(self.num_layers))
        else:
            carries = tuple(initial_carry)

        # Apply stacked LSTMs with nn.scan over time
        y = x_flat
        new_carries = []
        for i in range(self.num_layers):
            ScannedCell = nn.scan(
                _ResetLSTMCell,
                variable_broadcast=('params',),
                split_rngs={'params': False},
                in_axes=0,
                out_axes=0,
            )
            scan_cell = ScannedCell(self.hidden_dim, name=f'lstm_scan_{i}')
            carry_i, y = scan_cell(carries[i], (y, resets_flat))
            new_carries.append(carry_i)

        out = y.reshape((T, *leading_shape, self.hidden_dim))
        final_carry = tuple(new_carries)
        if return_carry:
            return out, final_carry
        return out
