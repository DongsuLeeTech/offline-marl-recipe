import copy
from typing import Any, Dict, Sequence

import flax
import jax
import jax.numpy as jnp
import ml_collections
import optax

from utils.encoders import encoder_modules
from utils.flax_utils import ModuleDict, TrainState, nonpytree_field
from utils.networks import Actor, Value, Mixer, ShapelyMixer
from util import concat_agent_id_to_obs, batch_concat_agent_id_to_obs, switch_two_leading_dims


class IQLAgent(flax.struct.PyTreeNode):
    rng: Any
    network: Any
    agent_names: Sequence[str] = nonpytree_field()
    config: Any = nonpytree_field()

    @staticmethod
    def expectile_loss(adv, diff, expectile):
        weight = jnp.where(adv >= 0, expectile, (1 - expectile))
        return weight * (diff**2)

    def _compute_future_discounted_returns(self, rewards_tm):
        discount = self.config['discount']
        T = rewards_tm.shape[0]

        def body_fn(carry, t):
            g_next = carry
            g_t = rewards_tm[t] + discount * g_next
            return g_t, g_t

        g_last = jnp.zeros_like(rewards_tm[-1])
        _, g_rev = jax.lax.scan(lambda c, x: body_fn(c, x), g_last, jnp.arange(T - 1, -1, -1))
        return g_rev[::-1]

    def _huber(self, x, delta=2.0):
        abs_x = jnp.abs(x)
        quad = jnp.minimum(abs_x, delta)
        return 0.5 * quad * quad + delta * (abs_x - quad)

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

    def value_loss(self, batch, grad_params):
        T, B, _, _ = batch['actions'].shape
        mixer_info = {}

        obs_tm = batch['observations']
        states_tm = batch['infos']['state']
        if self.config['decompose_q'] == 'central':
            obs_tm = obs_tm.reshape(T, B, -1)

        if self.config['value_only']:
            next_v1_t, next_v2_t = self.network.select('target_value')(obs_tm[1:])
            next_v_t = jnp.minimum(next_v1_t, next_v2_t)

            rewards_tm = batch['rewards'][:-1]
            terminals_tm = batch['terminals'][:-1]
            if self.config['decompose_q'] == 'central':
                rewards_tm = jnp.sum(rewards_tm, axis=-1)
                terminals_tm = jnp.max(terminals_tm, axis=-1)
            q = rewards_tm + self.config['discount'] * (1.0 - terminals_tm) * next_v_t

            v1_t, v2_t = self.network.select('target_value')(obs_tm[:-1], params=grad_params)
            v_t = (v1_t + v2_t) / 2

            if self.config['decompose_q'] == 'mean':
                q = jnp.mean(q, axis=-1)
                v_t = jnp.mean(v_t, axis=-1)
            elif self.config['decompose_q'] == 'vdn':
                q = jnp.sum(q, axis=-1)
                v_t = jnp.sum(v_t, axis=-1)
            elif self.config['decompose_q'] == 'mixer':
                q_agents_pre = q
                v_t_agents_pre = v_t
                q = self.network.select('target_q_mixer')(q, states_tm[:-1])
                v_t = self.network.select('target_q_mixer')(v_t, states_tm[:-1])
                mixer_info.update({
                    'mixer/q_agents_abs_mean': jnp.abs(q_agents_pre).mean(),
                    'mixer/vt_agents_abs_mean': jnp.abs(v_t_agents_pre).mean(),
                    'mixer/q_tot_abs_mean': jnp.abs(q).mean(),
                    'mixer/vt_tot_abs_mean': jnp.abs(v_t).mean(),
                })
            elif self.config['decompose_q'] in ('individual', 'central'):
                pass

            adv = q - v_t

            q1 = rewards_tm + self.config['discount'] * (1.0 - terminals_tm) * next_v1_t
            q2 = rewards_tm + self.config['discount'] * (1.0 - terminals_tm) * next_v2_t
            v1, v2 = self.network.select('value')(obs_tm[:-1], params=grad_params)
            v = (v1 + v2) / 2

            if self.config['decompose_q'] == 'mean':
                q1 = jnp.mean(q1, axis=-1)
                q2 = jnp.mean(q2, axis=-1)
                v1 = jnp.mean(v1, axis=-1)
                v2 = jnp.mean(v2, axis=-1)
                v = jnp.mean(v, axis=-1)
            elif self.config['decompose_q'] == 'vdn':
                q1 = jnp.sum(q1, axis=-1)
                q2 = jnp.sum(q2, axis=-1)
                v1 = jnp.sum(v1, axis=-1)
                v2 = jnp.sum(v2, axis=-1)
                v = jnp.sum(v, axis=-1)
            elif self.config['decompose_q'] == 'mixer':
                q1 = self.network.select('target_q_mixer')(q1, states_tm[:-1])
                q2 = self.network.select('target_q_mixer')(q2, states_tm[:-1])
                v1 = self.network.select('q_mixer')(v1, states_tm[:-1], params=grad_params)
                v2 = self.network.select('q_mixer')(v2, states_tm[:-1], params=grad_params)
                v = self.network.select('q_mixer')(v, states_tm[:-1], params=grad_params)
            elif self.config['decompose_q'] in ('individual', 'central'):
                pass

            value_loss1 = self.expectile_loss(adv, q1 - v1, self.config['expectile']).mean()
            value_loss2 = self.expectile_loss(adv, q2 - v2, self.config['expectile']).mean()
            value_loss = (value_loss1 + value_loss2) / 2
            v = v_t
        else:
            if self.config['decompose_q'] == 'central':
                obs_flat = obs_tm
                actions_flat = batch['actions'].reshape(T, B, -1)
                q1, q2 = self.network.select('target_critic')(obs_flat[:-1], actions=actions_flat[:-1])
                q = jnp.minimum(q1, q2)
                v = self.network.select('value')(obs_flat[:-1], params=grad_params)
            else:
                q1, q2 = self.network.select('target_critic')(obs_tm[:-1], actions=batch['actions'][:-1])
                q = jnp.minimum(q1, q2)
                v = self.network.select('value')(obs_tm[:-1], params=grad_params)

            if self.config['decompose_q'] == 'mean':
                q = jnp.mean(q, axis=-1)
                v = jnp.mean(v, axis=-1)
            elif self.config['decompose_q'] == 'vdn':
                q = jnp.sum(q, axis=-1)
                v = jnp.sum(v, axis=-1)
            elif self.config['decompose_q'] == 'mixer':
                q_agents_pre = q
                v_agents_pre = v
                q = self.network.select('target_q_mixer')(q, states_tm[:-1])
                v = self.network.select('q_mixer')(v, states_tm[:-1])
                mixer_info.update({
                    'mixer/q_agents_abs_mean': jnp.abs(q_agents_pre).mean(),
                    'mixer/v_agents_abs_mean': jnp.abs(v_agents_pre).mean(),
                    'mixer/q_tot_abs_mean': jnp.abs(q).mean(),
                    'mixer/v_tot_abs_mean': jnp.abs(v).mean(),
                })
            elif self.config['decompose_q'] in ('individual', 'central'):
                pass

            if self.config['ddqn_trick']:
                v_t = self.network.select('target_value')(obs_tm[:-1])
                if self.config['decompose_q'] == 'mean':
                    v_t = jnp.mean(v_t, axis=-1)
                elif self.config['decompose_q'] == 'vdn':
                    v_t = jnp.sum(v_t, axis=-1)
                elif self.config['decompose_q'] == 'mixer':
                    v_t = self.network.select('target_q_mixer')(v_t, states_tm[:-1])
                elif self.config['decompose_q'] in ('individual', 'central'):
                    pass
                adv = q - v_t
            else:
                adv = q - v

            diff = adv
            if (self.config.get('mixer_q') in ('svn')) and (self.config['decompose_q'] == 'mixer'):
                mu_q = jax.lax.stop_gradient(jnp.mean(q))
                mad_q = jax.lax.stop_gradient(jnp.mean(jnp.abs(q - mu_q))) + 1e-6
                diff = adv / mad_q  # (Q - V) / sigma_Q

                mixer_info['mixer/mad_q'] = mad_q
                mixer_info['mixer/diff_normalized_std'] = jnp.std(diff)

            value_loss = self.expectile_loss(adv, diff, self.config['expectile']).mean()

        return value_loss, {
            'value/value_loss': value_loss,
            'value/v max': v.max(),
            'value/v min': v.min(),
            'value/v mean': v.mean(),
            'value/abs adv mean': jnp.abs(adv).mean(),
            'value/adv mean': adv.mean(),
            'value/adv max': adv.max(),
            'value/adv min': adv.min(),
            'value/accept prob': (adv >= 0).mean(),
            **mixer_info,
        }

    def critic_loss(self, batch, grad_params, step=0):
        T, B, _, _ = batch['actions'].shape
        states_tm = batch['infos']['state']

        if self.config['decompose_q'] == 'central':
            obs_tm = batch['observations'].reshape(T, B, -1)
            act_tm = batch['actions'].reshape(T, B, -1)
            next_v = self.network.select('target_value')(obs_tm[1:]) if self.config['use_target_v'] else self.network.select('value')(obs_tm[1:])
            qs = self.network.select('critic')(obs_tm[:-1], actions=act_tm[:-1], params=grad_params)
            rewards_tm = jnp.sum(batch['rewards'][:-1], axis=-1)
            terminals_tm = jnp.max(batch['terminals'][:-1], axis=-1)
        else:
            obs_tm = batch['observations']
            next_v = self.network.select('target_value')(obs_tm[1:]) if self.config['use_target_v'] else self.network.select('value')(obs_tm[1:])
            qs = self.network.select('critic')(obs_tm[:-1], actions=batch['actions'][:-1], params=grad_params)
            rewards_tm = batch['rewards'][:-1]
            terminals_tm = batch['terminals'][:-1]

        targets = rewards_tm + self.config['discount'] * (1.0 - terminals_tm) * next_v

        if self.config['decompose_q'] == 'mean':
            qs = jnp.mean(qs, axis=-1)
            targets = jnp.mean(targets, axis=-1)
        elif self.config['decompose_q'] == 'vdn':
            qs = jnp.sum(qs, axis=-1)
            targets = jnp.sum(targets, axis=-1)
        elif self.config['decompose_q'] == 'mixer':
            qs_1 = self.network.select('q_mixer')(qs[0], states_tm[:-1])
            qs_2 = self.network.select('q_mixer')(qs[1], states_tm[:-1])
            qs = jnp.stack([qs_1, qs_2], axis=0)
            targets = self.network.select('target_q_mixer')(targets, states_tm[:-1])
        elif self.config['decompose_q'] in ('individual', 'central'):
            pass

        td_err_1 = qs[0] - targets
        td_err_2 = qs[1] - targets

        if (self.config.get('mixer_q') in ('svn')) and (self.config['decompose_q'] in ('mixer')):
            q_tot = jnp.minimum(qs[0], qs[1])  # joint Q
            target_q = jax.lax.stop_gradient(targets)  # r + γ V'
            td_err = q_tot - target_q

            mu = jax.lax.stop_gradient(jnp.mean(q_tot))
            mad = jax.lax.stop_gradient(jnp.mean(jnp.abs(q_tot - mu))) + 1e-6

            q_hat = (q_tot - mu) / mad
            t_hat = (target_q - mu) / mad
            critic_loss = jnp.mean(0.5 * (q_hat - t_hat) ** 2)

            td_err_min = q_tot - targets
            metrics = {
                'critic/critic_loss': critic_loss,
                'critic/td_mse_min': jnp.mean((td_err_min) ** 2),
                'critic/td_abs_min': jnp.mean(jnp.abs(td_err_min)),
                'critic/target_q_mean': jnp.mean(targets),
            }
            return critic_loss, metrics


        critic_loss_1 = jnp.mean(0.5 * (td_err_1) ** 2)
        critic_loss_2 = jnp.mean(0.5 * (td_err_2) ** 2)
        critic_loss = (critic_loss_1 + critic_loss_2) / 2

        q_min = jnp.minimum(qs[0], qs[1])
        td_err_min = q_min - targets
        td_mse_1 = jnp.mean((td_err_1) ** 2)
        td_mse_2 = jnp.mean((td_err_2) ** 2)
        td_mse_min = jnp.mean((td_err_min) ** 2)

        metrics = {
            'critic/critic_loss_1': critic_loss_1,
            'critic/critic_loss_2': critic_loss_2,
            'critic/critic_loss': critic_loss,
            'critic/td_mse_1': td_mse_1,
            'critic/td_mse_2': td_mse_2,
            'critic/td_mse_min': td_mse_min,
            'critic/td_abs_min': jnp.mean(jnp.abs(td_err_min)),
            'critic/q1_mean': jnp.mean(qs[0]),
            'critic/q2_mean': jnp.mean(qs[1]),
            'critic/q_min_mean': jnp.mean(q_min),
            'critic/target_q_mean': jnp.mean(targets),
            'critic/targets_mean': jnp.mean(targets),
        }
        return critic_loss, metrics

    def actor_loss(self, batch, grad_params, rng=None):
        T, B, _, _ = batch['actions'].shape
        temp = self.config['alpha']

        if self.config['use_log_q']:
            value_transform = lambda x: jnp.log(jnp.maximum(x, 1e-6))
        else:
            value_transform = lambda x: x

        if self.config['actor_loss'] == 'awr':
            if self.config['value_only']:
                obs_tm = batch['observations']
                if self.config['decompose_q'] == 'central':
                    obs_tm = obs_tm.reshape(T, B, -1)

                v1, v2 = value_transform(self.network.select('value')(obs_tm[:-1]))
                nv1, nv2 = value_transform(self.network.select('value')(obs_tm[1:]))
                v = (v1 + v2) / 2
                nv = (nv1 + nv2) / 2

                rewards_tm = batch['rewards'][:-1]
                terminals_tm = batch['terminals'][:-1]
                if self.config['decompose_q'] == 'central':
                    rewards_tm = jnp.sum(rewards_tm, axis=-1)
                    terminals_tm = jnp.max(terminals_tm, axis=-1)

                adv = rewards_tm + self.config['discount'] * (1.0 - terminals_tm) * nv - v
            else:
                if self.config['decompose_q'] == 'central':
                    obs_tm = batch['observations'].reshape(T, B, -1)
                    actions_tm = batch['actions'].reshape(T, B, -1)
                    v = value_transform(self.network.select('value')(obs_tm[:-1]))
                    q1, q2 = value_transform(self.network.select('critic')(obs_tm[:-1], actions=actions_tm[:-1]))
                else:
                    obs_tm = batch['observations']
                    v = value_transform(self.network.select('value')(obs_tm[:-1]))
                    q1, q2 = value_transform(self.network.select('critic')(obs_tm[:-1], actions=batch['actions'][:-1]))
                q = jnp.minimum(q1, q2)

                if self.config['decompose_q'] == 'mean':
                    q = jnp.mean(q, axis=-1)
                    v = jnp.mean(v, axis=-1)
                elif self.config['decompose_q'] == 'vdn':
                    q = jnp.sum(q, axis=-1)
                    v = jnp.sum(v, axis=-1)
                elif self.config['decompose_q'] == 'mixer':
                    q = self.network.select('q_mixer')(q, batch['infos']['state'][:-1])
                    v = self.network.select('q_mixer')(v, batch['infos']['state'][:-1])
                elif self.config['decompose_q'] in ('individual', 'central'):
                    pass
                adv = q - v

            adv = jax.lax.stop_gradient(adv)
            adv_mean = jnp.mean(adv)
            adv_std = jnp.std(adv) + 1e-6
            normalized_adv = (adv - adv_mean) / adv_std

            exp_a = jnp.exp(jnp.clip(normalized_adv * temp, a_min=-50.0, a_max=10.0))
            exp_a = jnp.minimum(exp_a, 100.0)

            dist = self.network.select('actor')(batch['observations'][:-1], params=grad_params)
            log_probs = dist.log_prob(batch['actions'][:-1])
            # Broadcast advantage weights to match per-agent log_probs when Q is aggregated over agents.
            while exp_a.ndim < log_probs.ndim:
                exp_a = exp_a[..., None]
            actor_loss = -(exp_a * log_probs).mean()

            metrics = {
                'actor/actor_loss': actor_loss,
                'actor/adv': adv.mean(),
                'actor/bc_log_probs': log_probs.mean(),
                'actor/mse': jnp.mean((dist.mode() - batch['actions'][:-1]) ** 2),
                'actor/std': jnp.mean(dist.scale_diag),
            }
            return actor_loss, metrics

        elif self.config['actor_loss'] == 'brac':
            assert not self.config['value_only']

            total_actor_loss = 0.0
            info = {}

            bc_dist = self.network.select('bc_actor')(batch['observations'][:-1], params=grad_params)
            bc_log_probs = bc_dist.log_prob(batch['actions'][:-1])
            bc_actor_loss = -bc_log_probs.mean()
            total_actor_loss = total_actor_loss + bc_actor_loss

            if self.config['dual_type'] in ['none', 'avg']:
                dist = self.network.select('actor')(batch['observations'][:-1], params=grad_params)
            else:
                dist, log_lam = self.network.select('actor')(batch['observations'][:-1],
                                                             dual_type='scala', info=True, params=grad_params)

            if self.config['const_std']:
                q_actions = jnp.clip(dist.mode(), -1, 1)
            else:
                q_actions = jnp.clip(dist.sample(seed=rng), -1, 1)

            if self.config['decompose_q'] == 'central':
                obs_tm = batch['observations'].reshape(T, B, -1)
                qs = value_transform(self.network.select('critic')(obs_tm[:-1], actions=q_actions.reshape(T - 1, B, -1)))
            else:
                qs = value_transform(self.network.select('critic')(batch['observations'][:-1], actions=q_actions))

            if self.config['decompose_q'] == 'mean':
                qs = jnp.mean(qs, axis=-1)
            elif self.config['decompose_q'] == 'vdn':
                qs = jnp.sum(qs, axis=-1)
            elif self.config['decompose_q'] == 'mixer':
                # Mix each critic ensemble separately, then stack back.
                qs_1 = self.network.select('q_mixer')(qs[0], batch['infos']['state'][:-1])
                qs_2 = self.network.select('q_mixer')(qs[1], batch['infos']['state'][:-1])
                qs = jnp.stack([qs_1, qs_2], axis=0)
            elif self.config['decompose_q'] in ('individual', 'central'):
                pass

            pi_q = qs.min(axis=0)
            log_probs = dist.log_prob(batch['actions'][:-1])

            if self.config.get('mixer_q') == 'svn' or 'simple-remedy':
                mu = jax.lax.stop_gradient(jnp.mean(pi_q))
                scale = jax.lax.stop_gradient(jnp.mean(jnp.abs(pi_q))) + 1e-6
                q_loss = - jnp.mean((pi_q - mu) / scale)
            else:
                q_loss = -pi_q.mean()

            if self.config['dual_type'] in ['avg', 'none']:
                if self.config['dual_type'] == 'none':
                    bc_loss = -(temp * log_probs).mean()
                else:
                    avg_q = jax.lax.stop_gradient(jnp.abs(q).mean())
                    bc_loss = -(temp * avg_q * log_probs).mean()
                dual_loss = 0
                kl = dist.kl_divergence(bc_dist).mean()
                info.update({'kl': kl})
            else:
                log_lam = jnp.minimum(log_lam, 10)
                lam = jnp.exp(log_lam)
                bc_loss = -(jax.lax.stop_gradient(lam) * log_probs).mean()
                kl = dist.kl_divergence(bc_dist).mean()
                dual_loss = -log_lam * (jax.lax.stop_gradient(kl) - temp).mean()
                info.update({'dual_loss': dual_loss, 'lam': lam, 'kl': kl})

            actor_loss = q_loss + bc_loss + dual_loss
            total_actor_loss = total_actor_loss + actor_loss
            info.update({
                'actor/actor_loss': actor_loss,
                'actor/bc_log_probs': log_probs.mean(),
                'actor/mse': jnp.mean((dist.mode() - batch['actions'][:-1]) ** 2),
                'actor/q_loss': q_loss,
                'actor/std': jnp.mean(dist.scale_diag),
            })
            return total_actor_loss, info

        else:
            raise ValueError(f'Unsupported actor loss: {self.config["actor_loss"]}')

    @jax.jit
    def total_loss(self, batch, grad_params, rng=None, step=0):
        info = {}
        rng = self.rng if rng is None else rng

        states = batch['infos']['state']
        observations = batch["observations"]
        actions = self._actions_to_one_hot(batch["actions"])
        rewards = batch["rewards"]
        terminals = jnp.array(batch["terminals"], "float32")

        observations = batch_concat_agent_id_to_obs(observations)
        observations = switch_two_leading_dims(observations)

        states = switch_two_leading_dims(states)
        replay_actions = switch_two_leading_dims(actions)
        rewards = switch_two_leading_dims(rewards)
        terminals = switch_two_leading_dims(terminals)

        batch["infos"]['state'] = states
        batch["observations"] = observations
        batch["actions"] = replay_actions
        batch["rewards"] = rewards
        batch["terminals"] = terminals

        value_loss, value_info = self.value_loss(batch, grad_params)
        info.update(value_info)

        critic_loss = 0
        if not self.config['value_only']:
            critic_loss, critic_info = self.critic_loss(batch, grad_params, step=step)
            info.update(critic_info)

        rng, actor_rng = jax.random.split(rng)
        actor_loss, actor_info = self.actor_loss(batch, grad_params, actor_rng)
        info.update(actor_info)

        loss = value_loss + critic_loss + actor_loss
        info['loss'] = loss
        return loss, info

    def target_update(self, network, module_name):
        new_target_params = jax.tree_util.tree_map(
            lambda p, tp: p * self.config['tau'] + tp * (1 - self.config['tau']),
            self.network.params[f'modules_{module_name}'],
            self.network.params[f'modules_target_{module_name}'],
        )
        network.params[f'modules_target_{module_name}'] = new_target_params

    @jax.jit
    def update(self, batch, step=0):
        new_rng, rng = jax.random.split(self.rng)

        def loss_fn(grad_params):
            return self.total_loss(batch, grad_params, rng=rng, step=step)

        new_network, info = self.network.apply_loss_fn(loss_fn=loss_fn)
        if not self.config['value_only']:
            self.target_update(new_network, 'critic')
        if self.config['use_target_v']:
            self.target_update(new_network, 'value')
        if self.config['decompose_q'] == 'mixer':
            self.target_update(new_network, 'q_mixer')

        return self.replace(network=new_network, rng=new_rng), info

    @jax.jit
    def sample_actions(self, observations: Dict[str, jnp.ndarray], seed=None, actor_temperature=1.0):
        rngs = jax.random.split(seed if seed is not None else self.rng, len(self.agent_names))
        acts = {}
        for r, i, agent in zip(rngs, range(self.config['num_agents']), self.agent_names):
            agent_obs = concat_agent_id_to_obs(observations[agent], i, self.config['num_agents'])
            dist = self.network.select('actor')(agent_obs, temperature=actor_temperature)
            action = dist.sample(seed=r)
            if self._is_discrete():
                acts[agent] = self._discretize_actions(action)
            else:
                acts[agent] = jnp.clip(action, -1, 1)
        return acts

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
        rng = jax.random.PRNGKey(seed)
        rng, init_rng = jax.random.split(rng, 2)

        if config.get('discrete', False) and ex_actions.ndim == 3:
            action_dim = int(config.get('discrete_action_dim', 0))
            if action_dim <= 0:
                action_dim = int(jnp.max(ex_actions)) + 1
            ex_actions = jax.nn.one_hot(ex_actions.astype(jnp.int32), action_dim)
        else:
            action_dim = ex_actions.shape[-1]

        encoders = dict()
        if config['encoder'] is not None:
            encoder_module = encoder_modules[config['encoder']]
            encoders['value'] = encoder_module()
            encoders['critic'] = encoder_module()
            encoders['actor'] = encoder_module()

        value_ensembles = 2 if config['value_only'] else 1
        value_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=value_ensembles,
            encoder=encoders.get('value'),
        )
        critic_def = Value(
            hidden_dims=config['value_hidden_dims'],
            layer_norm=config['layer_norm'],
            num_ensembles=2,
            encoder=encoders.get('critic'),
        )
        actor_def = Actor(
            hidden_dims=config['actor_hidden_dims'],
            action_dim=action_dim,
            layer_norm=config['actor_layer_norm'],
            state_dependent_std=False,
            const_std=config['const_std'],
            encoder=encoders.get('actor'),
        )

        mixer_def = Mixer(
            embedding_dim=128,
            hypernet_hidden_dim=512,
            layer_norm=config['mixer_layer_norm'],
            init_scale=config.get('mixer_init_scale', 1.0),
        )
        if config['decompose_q'] == 'shapely':
            mixer_def = ShapelyMixer(
                n_agents=len(agent_names),
                n_actions=action_dim,
                sample_size=16,
                embed_dim=256,
            )

        ex_obs_with_id = batch_concat_agent_id_to_obs(ex_observations)
        B, T, N, _ = ex_obs_with_id.shape

        if config['decompose_q'] == 'central':
            ex_states_tm = switch_two_leading_dims(ex_states)
            ex_agent_q = jnp.zeros((ex_states_tm.shape[0], ex_states_tm.shape[1], len(agent_names)), dtype=ex_states.dtype)
            network_info = dict(
                value=(value_def, (ex_obs_with_id.reshape(B, T, -1),)),
                target_value=(copy.deepcopy(value_def), (ex_obs_with_id.reshape(B, T, -1),)),
                critic=(critic_def, (ex_obs_with_id.reshape(B, T, -1), ex_actions.reshape(B, T, -1))),
                target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id.reshape(B, T, -1), ex_actions.reshape(B, T, -1))),
                bc_actor=(actor_def, (ex_obs_with_id,)),
                actor=(actor_def, (ex_obs_with_id,)),
                q_mixer=(mixer_def, (ex_agent_q, ex_states_tm)),
                target_q_mixer=(copy.deepcopy(mixer_def), (ex_agent_q, ex_states_tm)),
            )
        else:
            ex_states_tm = switch_two_leading_dims(ex_states)
            ex_agent_q = jnp.zeros((ex_states_tm.shape[0], ex_states_tm.shape[1], len(agent_names)), dtype=ex_states.dtype)
            network_info = dict(
                value=(value_def, (ex_obs_with_id,)),
                target_value=(copy.deepcopy(value_def), (ex_obs_with_id,)),
                critic=(critic_def, (ex_obs_with_id, ex_actions)),
                target_critic=(copy.deepcopy(critic_def), (ex_obs_with_id, ex_actions)),
                bc_actor=(actor_def, (ex_obs_with_id,)),
                actor=(actor_def, (ex_obs_with_id,)),
                q_mixer=(mixer_def, (ex_agent_q, ex_states_tm)),
                target_q_mixer=(copy.deepcopy(mixer_def), (ex_agent_q, ex_states_tm)),
            )

        networks = {k: v[0] for k, v in network_info.items()}
        network_args = {k: v[1] for k, v in network_info.items()}

        network_def = ModuleDict(networks)
        network_tx = optax.chain(optax.zero_nans(), optax.adam(learning_rate=config['lr']))
        network_params = network_def.init(init_rng, **network_args)['params']
        network = TrainState.create(network_def, network_params, tx=network_tx)

        params = network.params
        params['modules_target_value'] = params['modules_value']
        params['modules_target_critic'] = params['modules_critic']
        params['modules_target_q_mixer'] = params['modules_q_mixer']

        config['ob_dims'] = ex_obs_with_id.shape[-1]
        config['action_dim'] = action_dim
        config['num_agents'] = len(agent_names)

        return cls(
            rng=rng,
            network=network,
            agent_names=tuple(agent_names),
            config=flax.core.FrozenDict(**config),
        )


def get_config():
    return ml_collections.ConfigDict(
        dict(
            agent_name='iql',
            lr=3e-4,
            actor_hidden_dims=(512, 512, 512, 512),
            value_hidden_dims=(512, 512, 512, 512),
            layer_norm=True,
            actor_layer_norm=False,
            mixer_layer_norm=True,
            mixer_init_scale=0.1,
            discount=0.99,
            tau=0.005,
            expectile=0.9,
            actor_loss='awr',
            num_samples=None,
            decompose_q='mean',
            mixer_q='None',
            q='iql',
            alpha=10.0,
            const_std=True,
            use_log_q=False,
            use_target_v=False,
            ddqn_trick=False,
            value_only=False,
            dual_type='none',
            beta=1.0,
            encoder=ml_collections.config_dict.placeholder(str),
            discrete=False,
            discrete_action_dim=0,
        )
    )
