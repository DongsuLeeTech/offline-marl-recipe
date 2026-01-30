import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import numpy as np
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, Mixer, ShapelyMixer
from util import *

class TD3BCAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    def _compute_future_discounted_returns(self, rewards_tm):
        discount = self.config['discount']
        T = rewards_tm.shape[0]
        def body_fn(carry, t):
            g_next = carry
            # Standard discounted return without termination gating (RES style)
            g_t = rewards_tm[t] + discount * g_next
            return g_t, g_t
        # scan backwards
        # Reverse inputs to accumulate from end to start
        g_last = jnp.zeros_like(rewards_tm[-1])
        _, g_rev = jax.lax.scan(lambda c, x: body_fn(c, x), g_last, jnp.arange(T-1, -1, -1))
        returns_tm = g_rev[::-1]
        return returns_tm

    def _huber(self, x, delta=2.0):
        abs_x = jnp.abs(x)
        quad = jnp.minimum(abs_x, delta)
        return 0.5 * quad * quad + delta * (abs_x - quad)

    def _cosine_similarity(self, a, b, eps=1e-6):
        a_flat = a.reshape(-1)
        b_flat = b.reshape(-1)
        denom = (jnp.linalg.norm(a_flat) * jnp.linalg.norm(b_flat)) + eps
        return jnp.sum(a_flat * b_flat) / denom

    def _warmup(self, step, target_lambda=0.05, warmup_steps=100000):
        ratio = jnp.clip(step / float(warmup_steps), 0.0, 1.0)
        return target_lambda * ratio

    def _is_discrete(self) -> bool:
        return bool(self.config.get('discrete', False))

    def _actions_to_one_hot(self, actions: jnp.ndarray) -> jnp.ndarray:
        if not self._is_discrete():
            return actions
        if actions.ndim == 3:
            return jax.nn.one_hot(actions.astype(jnp.int32), self.config['action_dim'])
        return actions

    def _discretize_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        return jnp.argmax(actions, axis=-1)

    def critic_loss(self, batch, grad_params, step=0):
        T, B, _, _ = batch['actions'].shape
        if self.config['decompose_q'] == 'central':
            qs = self.network.select("critic")(batch['observations'].reshape(T, B, -1)[:-1],
                                               actions=batch['actions'].reshape(T, B, -1)[:-1],
                                               params=grad_params)

            if self.config['q'] == 'sarsa':
                target_qs = self.network.select("target_critic")(batch['observations'].reshape(T, B, -1)[1:],
                                                                 actions=batch['actions'].reshape(T, B, -1)[1:])
            elif self.config['q'] == 'td':
                pi_dist = self.network.select("actor")(batch['observations'][1:])
                pi_actions = pi_dist.mode()
                target_qs = self.network.select("target_critic")(batch['observations'].reshape(T, B, -1)[1:],
                                                                 actions=pi_actions.reshape(T-1, B, -1))
        else:
            qs = self.network.select("critic")(batch['observations'][:-1], actions=batch['actions'][:-1],
                                               params=grad_params)

            if self.config['q'] == 'sarsa':
                target_qs = self.network.select("target_critic")(batch['observations'][1:],
                                                                 actions=batch['actions'][1:])
            elif self.config['q'] == 'td':
                pi_dist = self.network.select("actor")(batch['observations'][1:])
                pi_actions = pi_dist.mode()
                target_qs = self.network.select("target_critic")(batch['observations'][1:], actions=pi_actions)

        # Combine target ensemble Qs
        if self.config['decompose_q'] == 'mixer':
            target_qs_1 = self.network.select('target_q_mixer')(target_qs[0], batch['infos']['state'][1:])
            target_qs_2 = self.network.select('target_q_mixer')(target_qs[1], batch['infos']['state'][1:])
            target_qs_tot = jnp.stack([target_qs_1, target_qs_2], axis=0)
            target_q_combined = target_qs_tot.min(axis=0)
        else:
            target_q_combined = target_qs.min(axis=0)

        # Centralized critic uses team-level targets; aggregate rewards/terminations across agents
        if self.config['decompose_q'] == 'central':
            team_rewards = jnp.sum(batch['rewards'][:-1], axis=-1)
            team_terminals = jnp.max(batch['terminals'][1:], axis=-1)
            targets = team_rewards + (1.0 - team_terminals) * self.config['discount'] * target_q_combined
        elif self.config['decompose_q'] == 'mixer':
            team_rewards = jnp.sum(batch['rewards'][:-1], axis=-1)
            team_terminals = jnp.max(batch['terminals'][1:], axis=-1)
            targets = team_rewards + (1.0 - team_terminals) * self.config['discount'] * target_q_combined
        else:
            targets = batch['rewards'][:-1] + (1.0 - batch['terminals'][1:]) * self.config['discount'] * target_q_combined

        if self.config['decompose_q'] == 'mean':
            qs = jnp.mean(qs, axis=-1)
            targets = jnp.mean(targets, axis=-1)
        elif self.config['decompose_q'] == 'vdn':
            qs = jnp.sum(qs, axis=-1)
            targets = jnp.sum(targets, axis=-1)
        elif self.config['decompose_q'] == 'mixer':
            qs_1 = self.network.select('q_mixer')(qs[0], batch['infos']['state'][:-1])
            qs_2 = self.network.select('q_mixer')(qs[1], batch['infos']['state'][:-1])
            qs = jnp.stack([qs_1, qs_2], axis=0)
        elif self.config['decompose_q'] in ('individual', 'central'):
            pass

        # Scale-invariant TD loss
        if (self.config.get('mixer_q') == 'svn') and (self.config['decompose_q'] in ('mixer')):
            # Total Q from ensemble: min over critics
            q_tot_curr = jnp.minimum(qs[0], qs[1])  # (T-1, B)
            target_q = jax.lax.stop_gradient(targets)  # Fix TD target for Bellman invariance

            # Detached normalization statistics ONLY from Current Q
            mu_q = jax.lax.stop_gradient(jnp.mean(q_tot_curr))
            mad_q = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_tot_curr - mu_q))) + 1e-6

            # Normalize BOTH Q and Target using Q's statistics
            q_hat = (q_tot_curr - mu_q) / mad_q
            t_hat = (target_q - mu_q) / mad_q

            # Scale-invariant squared TD loss
            critic_loss = jnp.mean(0.5 * (q_hat - t_hat) ** 2)

            # Also compute standard diagnostics for visibility (not used for optimization)
            td_err_1 = qs[0] - targets
            td_err_2 = qs[1] - targets
            q_min = jnp.minimum(qs[0], qs[1])
            td_err_min = q_min - targets

            td_mse_1 = jnp.mean((td_err_1) ** 2)
            td_mse_2 = jnp.mean((td_err_2) ** 2)
            td_mse_min = jnp.mean((td_err_min) ** 2)
            td_abs_min = jnp.mean(jnp.abs(td_err_min))

            q1_mean = jnp.mean(qs[0])
            q2_mean = jnp.mean(qs[1])
            q_min_mean = jnp.mean(q_min)
            target_q_mean = jnp.mean(target_qs)
            targets_mean = jnp.mean(targets)

            # Per-critic MSE (diagnostic only)
            critic_loss_1 = jnp.mean(0.5 * (td_err_1) ** 2)
            critic_loss_2 = jnp.mean(0.5 * (td_err_2) ** 2)

            metrics = {
                # Optimization loss
                'critic/critic_loss': critic_loss,
                # Keep familiar diagnostics
                'critic/critic_loss_1': critic_loss_1,
                'critic/critic_loss_2': critic_loss_2,
                'critic/td_mse_1': td_mse_1,
                'critic/td_mse_2': td_mse_2,
                'critic/td_mse_min': td_mse_min,
                'critic/td_abs_min': td_abs_min,
                'critic/q1_mean': q1_mean,
                'critic/q2_mean': q2_mean,
                'critic/q_min_mean': q_min_mean,
                'critic/target_q_mean': target_q_mean,
                'critic/targets_mean': targets_mean,
                # Invariance stats
                'proposed_inv/enabled': 1.0,
                'proposed_inv/mu_q': mu_q,
                'proposed_inv/mad_q': mad_q,
            }
            return critic_loss, metrics

        # Per-ensemble TD errors
        td_err_1 = qs[0] - targets
        td_err_2 = qs[1] - targets
        critic_loss_1 = jnp.mean(0.5 * (td_err_1) ** 2)
        critic_loss_2 = jnp.mean(0.5 * (td_err_2) ** 2)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2

        # Additional debugging metrics
        q_min = jnp.minimum(qs[0], qs[1])
        td_err_min = q_min - targets
        td_mse_1 = jnp.mean((td_err_1) ** 2)
        td_mse_2 = jnp.mean((td_err_2) ** 2)
        td_mse_min = jnp.mean((td_err_min) ** 2)

        q1_mean = jnp.mean(qs[0])
        q2_mean = jnp.mean(qs[1])
        q_min_mean = jnp.mean(q_min)
        target_q_mean = jnp.mean(target_qs)
        targets_mean = jnp.mean(targets)
        td_abs_min = jnp.mean(jnp.abs(td_err_min))

        metrics = {
            'critic/critic_loss_1': critic_loss_1,
            'critic/critic_loss_2': critic_loss_2,
            'critic/critic_loss': critic_loss,
            # TD error diagnostics (no 0.5 factor)
            'critic/td_mse_1': td_mse_1,
            'critic/td_mse_2': td_mse_2,
            'critic/td_mse_min': td_mse_min,
            'critic/td_abs_min': td_abs_min,
            # Q statistics
            'critic/q1_mean': q1_mean,
            'critic/q2_mean': q2_mean,
            'critic/q_min_mean': q_min_mean,
            'critic/target_q_mean': target_q_mean,
            'critic/targets_mean': targets_mean,
        }

        return critic_loss, metrics

    def actor_loss(self,batch, grad_params):
        T, B, _, _ = batch['actions'].shape
        temp = self.config['alpha']
        use_mixer = bool(self.config.get('actor_use_mixer')) and self.config['decompose_q'] == 'mixer'

        def mix_ensemble_qs(qs, states):
            qs_1 = self.network.select('q_mixer')(qs[0], states)
            qs_2 = self.network.select('q_mixer')(qs[1], states)
            return jnp.stack([qs_1, qs_2], axis=0)

        # Consider proposed: turn Q normalization on for actor
        if self.config['use_log_q']:
            value_transform = lambda x: jnp.log(jnp.maximum(x, 1e-6))
        else:
            value_transform = lambda x: x

        if self.config['actor_loss'] == 'brac':
            pi_dist = self.network.select("actor")(batch['observations'][:-1], params=grad_params)
            pi_actions = pi_dist.mode()

            if self.config['decompose_q'] == 'central':
                pi_qs = value_transform(self.network.select("critic")(batch['observations'].reshape(T, B, -1)[:-1],
                                                                      actions=pi_actions.reshape(T-1, B, -1)))
            else:
                pi_qs = self.network.select("critic")(batch['observations'][:-1], actions=pi_actions)
                if use_mixer:
                    pi_qs = mix_ensemble_qs(pi_qs, batch['infos']['state'][:-1])
                pi_qs = value_transform(pi_qs)
            pi_q = pi_qs.min(axis=0)

            # For diagnostics, also evaluate critic on dataset actions
            if self.config['decompose_q'] == 'central':
                qs_dataset = value_transform(self.network.select('critic')(batch['observations'].reshape(T, B, -1)[:-1],
                                                                           actions=batch['actions'][:-1].reshape(T-1, B, -1)))
            else:
                qs_dataset = self.network.select('critic')(batch['observations'][:-1],
                                                           actions=batch['actions'][:-1])
                if use_mixer:
                    qs_dataset = mix_ensemble_qs(qs_dataset, batch['infos']['state'][:-1])
                qs_dataset = value_transform(qs_dataset)
            q_dataset = qs_dataset.min(axis=0)

            bc_loss = jnp.mean((pi_actions - batch['actions'][:-1]) ** 2)

            if self.config.get('mixer_q') == 'svn' or 'simple-remedy':
                mu = jax.lax.stop_gradient(jnp.mean(pi_q))
                scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(pi_q))) + 1e-6
                q_loss = - jnp.mean((pi_q - mu) / scale)
            else:
                q_loss = - jnp.mean(pi_q)

            actor_loss = temp * bc_loss + q_loss
            metrics = {
                'actor/actor_loss': actor_loss,
                'actor/bc_loss': bc_loss,
                'actor/q_loss': q_loss,
                'actor/mse': bc_loss,
                'actor/pi_q_mean': jnp.mean(pi_q),
                'actor/beh_q_mean': jnp.mean(q_dataset),
                'actor/q_gap_mean': jnp.mean(q_dataset - pi_q),
                'actor/std': jnp.mean(pi_dist.scale_diag),
            }
        elif self.config['actor_loss'] == 'awr':
            pi_dist = self.network.select("actor")(batch['observations'][:-1], params=grad_params)
            pi_actions = pi_dist.mode()

            if self.config['decompose_q'] == 'central':
                pi_qs = value_transform(self.network.select("critic")(batch['observations'].reshape(T, B, -1)[:-1],
                                                                      actions=pi_actions.reshape(T-1, B, -1)))
                pi_q = pi_qs.min(axis=0)

                qs = value_transform(
                    self.network.select('critic')(batch['observations'].reshape(T, B, -1)[:-1],
                                                  actions=batch['actions'].reshape(T, B, -1)[:-1]))
                q = qs.min(axis=0)
            else:
                pi_qs = self.network.select("critic")(batch['observations'][:-1], actions=pi_actions)
                if use_mixer:
                    pi_qs = mix_ensemble_qs(pi_qs, batch['infos']['state'][:-1])
                pi_qs = value_transform(pi_qs)
                pi_q = pi_qs.min(axis=0)

                qs = self.network.select('critic')(batch['observations'][:-1], actions=batch['actions'][:-1])
                if use_mixer:
                    qs = mix_ensemble_qs(qs, batch['infos']['state'][:-1])
                qs = value_transform(qs)
                q = qs.min(axis=0)

            adv = jax.lax.stop_gradient(q - pi_q)

            adv_mean = jnp.mean(adv)
            adv_std = jnp.std(adv) + 1e-6
            normalized_adv = (adv - adv_mean) / adv_std

            # Numerically stable exponentiation and explicit broadcasting
            logits = jnp.clip(normalized_adv * temp, a_min=-50.0, a_max=10.0)
            exp_a = jnp.exp(logits)
            exp_a = jnp.minimum(exp_a, 100.0)

            log_probs = pi_dist.log_prob(batch['actions'][:-1])

            # For centralized critic, adv has shape (T-1, B). Broadcast over agents explicitly.
            if self.config['decompose_q'] == 'central':
                weights = exp_a[..., None]
            else:
                weights = exp_a

            actor_loss = -(weights * log_probs).mean()

            metrics = {
                'actor/actor_loss': actor_loss,
                'actor/adv': adv.mean(),
                'actor/bc_log_probs': log_probs.mean(),
                'actor/mse': ((pi_actions - batch['actions'][:-1]) ** 2).mean(),
                'actor/std': jnp.mean(pi_dist.scale_diag),
                'actor/pi_q_mean': jnp.mean(pi_q),
                'actor/beh_q_mean': jnp.mean(q),
                'actor/q_gap_mean': jnp.mean(q - pi_q),
            }

        return actor_loss, metrics

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=0):
        info = {}
        rng = self.rng if rng is None else rng

        states = batch['infos']['state']
        observations = batch["observations"]  # (B,T,N,O)
        actions = self._actions_to_one_hot(batch["actions"])  # (B,T,N,A)
        rewards = batch["rewards"]  # (B,T,N)
        terminals = jnp.array(batch["terminals"], "float32")  # (B,T,N)
        truncations = jnp.array(batch["truncations"], "float32")  # (B,T,N)

        # Make time-major
        observations = batch_concat_agent_id_to_obs(observations)
        observations = switch_two_leading_dims(observations)

        states = switch_two_leading_dims(states)
        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)
        truncations = switch_two_leading_dims(truncations)

        batch["infos"]['state'] = states
        batch["observations"] = observations
        batch["actions"] = replay_actions
        batch["rewards"] = rewards
        batch["terminals"] = terminals
        batch["truncations"] = truncations

        c_loss, c_info = self.critic_loss(batch, grad_params, step=step)
        p_loss, p_info = self.actor_loss(batch, grad_params)

        for k, v in c_info.items():
            info[f'critic/{k}'] = v
        for k, v in p_info.items():
            info[f'policy/{k}'] = v

        loss = c_loss + p_loss
        info['loss'] = loss
        return loss, info

    # --------------------------------------------------------------------- #
    #  Update
    # --------------------------------------------------------------------- #
    def target_update(self, network, module_name):
        """Update the target network."""
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            network.params[f'modules_{module_name}'],
            network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step=0):
        """Update the agent and return a new agent with information dictionary."""
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        self.target_update(new_network, 'critic')
        if self.config['decompose_q'] == 'mixer':
            self.target_update(new_network, 'q_mixer')

        return self.replace(
            network=new_network,
            rng=new_rng,
        ), info

    # --------------------------------------------------------------------- #
    #  Acting
    # --------------------------------------------------------------------- #
    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed, actor_temperature=0.0):
        rngs = jax.random.split(seed, len(self.agent_names))
        acts = {}
        for r, i, agent in zip(rngs, range(self.config['num_agents']), self.agent_names):
            agent_observation = concat_agent_id_to_obs(observations[agent], i, self.config['num_agents'])
            dist = self.network.select('actor')(agent_observation, temperature=actor_temperature)
            action = dist.mode()
            if self._is_discrete():
                acts[agent] = self._discretize_actions(action)
            else:
                acts[agent] = action
        return acts

    # --------------------------------------------------------------------- #
    #  Factory
    # --------------------------------------------------------------------- #
    @classmethod
    def create(
        cls,
        seed: int,
        ex_states: jnp.ndarray,
        ex_observations: jnp.ndarray,
        ex_actions: jnp.ndarray,
        agent_names,
        config,
    ):
        """Instantiate new MABCAgent."""
        master_rng = jax.random.PRNGKey(seed)
        master_rng, big_init_rng = jax.random.split(master_rng, 2)

        if config.get('discrete', False) and ex_actions.ndim == 3:
            action_dim = int(config.get('discrete_action_dim', 0))
            if action_dim <= 0:
                action_dim = int(jnp.max(ex_actions)) + 1
            ex_actions = jax.nn.one_hot(ex_actions.astype(jnp.int32), action_dim)
        else:
            action_dim = ex_actions.shape[-1]

        # ------------------------------------------------- encoders (optional)
        encoders = {}
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['critic'] = encoder_module()
            encoders['policy'] = encoder_module()

        policy_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['layer_norm'],
            encoder=encoders.get('policy')
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        mixer_def = Mixer(
            embedding_dim=config['mixer_emb_dim'],
            hypernet_hidden_dim=config['mixer_hyper_dim'],
            layer_norm=config['mixer_layer_norm'],
        )
        if config['decompose_q'] == 'shapely':
            mixer_def = ShapelyMixer(
                n_agents=len(agent_names),
                n_actions=action_dim,
                sample_size=16,
                embed_dim=256,
            )

        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)
        B, T, N, O = ex_obs_with_id.shape
        if config['decompose_q'] == 'central':
            ex_states_tm = switch_two_leading_dims(ex_states)
            ex_agent_q = jnp.zeros((ex_states_tm.shape[0], ex_states_tm.shape[1], len(agent_names)), dtype=ex_states.dtype)
            network_info = dict(
                critic=(critic_def, (ex_obs_with_id.reshape(B, T, -1), ex_actions.reshape(B, T, -1))),
                target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id.reshape(B, T, -1), ex_actions.reshape(B, T, -1))),
                actor=(policy_def, (ex_obs_with_id, )),
                q_mixer=(mixer_def, (ex_agent_q, ex_states_tm)),
                target_q_mixer=(copy.deepcopy(mixer_def), (ex_agent_q, ex_states_tm)),
            )
        else:
            ex_states_tm = switch_two_leading_dims(ex_states)
            ex_agent_q = jnp.zeros((ex_states_tm.shape[0], ex_states_tm.shape[1], len(agent_names)), dtype=ex_states.dtype)
            network_info = dict(
                critic=(critic_def, (ex_obs_with_id, ex_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id, ex_actions)),
                actor=(policy_def, (ex_obs_with_id, )),
                q_mixer=(mixer_def, (ex_agent_q, ex_states_tm)),
                target_q_mixer=(copy.deepcopy(mixer_def), (ex_agent_q, ex_states_tm)),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        grad_clip = config.get('grad_clip', 0.0)
        tx_layers = [optax.zero_nans()]
        if grad_clip and grad_clip > 0.0:
            tx_layers.append(optax.clip_by_global_norm(grad_clip))
        tx_layers.append(optax.adam(learning_rate=config['lr']))
        network_tx = optax.chain(*tx_layers)
        network_params = network_def.init(big_init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_q_mixer'] = params['modules_q_mixer']

        config['ob_dims'] = ex_obs_with_id.shape[-1]
        config['action_dim'] = action_dim
        config['num_agents'] = len(agent_names)

        return cls(
            rng=master_rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='td3bc',
            lr=3e-4,
            grad_clip=0.0,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            mixer_layer_norm=True,
            discount=0.99,
            tau=0.005,
            actor_loss='brac',            # awr or brac
            actor_use_mixer=False,        # whether to use mixer Q for actor update
            decompose_q='mixer',          # individual, vdn, mixer, and central
            mixer_emb_dim=32,
            mixer_hyper_dim=128,
            mixer_q='svn',                # svn or simple-remedy
            q='td',                       # td or sarsa
            alpha=0.03,                   # policy extraction temperature
            const_std=True,
            use_log_q=False,
            dual_type='none',
            beta=1.0,
            discrete=False,               # whether the action space is discrete
            discrete_action_dim=ml_collections.config_dict.placeholder(int),
            encoder=ml_collections.config_dict.placeholder(str),
        )
    )
