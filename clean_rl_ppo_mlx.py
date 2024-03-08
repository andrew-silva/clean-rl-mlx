# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppopy
import os
import random
import time
from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import tyro
from mlx.core.random import categorical
import wandb


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    wandb_project_name: str = "clean-rl-mlx"
    """the wandb's project name"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "CartPole-v1"
    """the id of the environment"""
    total_timesteps: int = 500000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 4
    """the number of parallel game environments"""
    num_steps: int = 128
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: Optional[float] = None
    """the target KL divergence threshold"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, r_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{r_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), mean: float = 0.0, bias_const: float = 0.0):
    weight_shape = layer.weight.shape
    a = np.random.normal(mean, std, weight_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == weight_shape else v  # Pick the one with the correct shape
    q = q.transpose()
    weights = q.reshape(weight_shape)
    layer.weight = mx.array(weights)
    layer.bias = mx.ones_like(layer.bias) * bias_const
    return layer


class Agent(nn.Module):
    def __init__(self, environments):
        super().__init__()
        hid_dim = 64
        self.critic_in = layer_init(nn.Linear(np.array(environments.single_observation_space.shape).prod(), hid_dim))
        self.critic_hid = layer_init(nn.Linear(hid_dim, hid_dim))
        self.critic_out = layer_init(nn.Linear(hid_dim, 1), std=1.0)

        self.actor_in = layer_init(nn.Linear(np.array(environments.single_observation_space.shape).prod(), hid_dim))
        self.actor_hid = layer_init(nn.Linear(hid_dim, hid_dim))
        self.actor_out = layer_init(nn.Linear(hid_dim, environments.single_action_space.n), std=0.01)

    def get_value(self, x):
        x_mid = nn.tanh(self.critic_in(x))
        x_mid = nn.tanh(self.critic_hid(x_mid))
        return self.critic_out(x_mid)

    def get_action_and_value(self, x, act=None):
        value_pred = self.get_value(x)
        x_mid = nn.tanh(self.actor_in(x))
        x_mid = nn.tanh(self.actor_hid(x_mid))
        logits = self.actor_out(x_mid)
        probs = categorical(logits=logits)
        logits = mx.softmax(logits, axis=1)
        entropy = -mx.sum(logits * mx.log(logits), axis=1, keepdims=True)
        if act is None:
            act = probs
        act = act.astype(mx.int32)
        act_log_prob = mx.log(logits[mx.arange(logits.shape[0]), act])
        return act, act_log_prob, entropy, value_pred


def loss(model, minibatch_observations, minibatch_actions,
         minibatch_log_probs, minibatch_advantages, minibatch_rewards, minibatch_values):
    s_t = time.time()
    _, newlogprob, entropy, newvalue = model.get_action_and_value(minibatch_observations,
                                                                  minibatch_actions)  # b_actions.long()[mb_inds])
    forward_pass_time = time.time()-s_t
    logratio = newlogprob - minibatch_log_probs
    ratio = logratio.exp()

    # with no grad
    # calculate approx_kl http://joschu.net/blog/kl-approx.html
    s_t = time.time()
    old_approx_kl = (-logratio).mean()
    approximate_kl = ((ratio - 1) - logratio).mean()
    # clipfracs = [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]
    kl_calc_time = time.time()-s_t
    s_t = time.time()
    # Policy loss
    pg_loss1 = -minibatch_advantages * ratio
    pg_loss2 = -minibatch_advantages * mx.clip(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
    pg_loss = mx.max(mx.stack([pg_loss1, pg_loss2]), axis=0).mean()
    pg_loss_time = time.time() - s_t

    s_t = time.time()
    # Value loss
    newvalue = newvalue.reshape(-1)
    if args.clip_vloss:
        v_loss_unclipped = (newvalue - minibatch_rewards) ** 2
        v_clipped = minibatch_values + mx.clip(
            newvalue - minibatch_values,
            -args.clip_coef,
            args.clip_coef,
        )
        v_loss_clipped = (v_clipped - minibatch_rewards) ** 2
        v_loss_max = mx.max(mx.stack([v_loss_unclipped, v_loss_clipped]), axis=0)
        v_loss = 0.5 * v_loss_max.mean()
    else:
        v_loss = 0.5 * ((newvalue - minibatch_rewards) ** 2).mean()
    value_loss_time = time.time()-s_t
    entropy_loss = entropy.mean()
    loss_v = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef
    approximate_kl = 0.0
    wandb.log({
        "losses/value_loss": v_loss.item(),
        "losses/policy_loss": pg_loss.item(),
        "losses/entropy": entropy_loss.item(),
        "losses/old_approx_kl": old_approx_kl.item(),
        "losses/approx_kl": approximate_kl,
        "times/update_forward_pass_time": forward_pass_time,
        "times/kl_time": kl_calc_time,
        "times/pg_loss_time": pg_loss_time,
        "times/value_loss_time": value_loss_time
    })
    return loss_v, approximate_kl


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.framework = 'MLX'
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    wandb.init(
        project=args.wandb_project_name,
        config=vars(args),
        name=run_name,
        monitor_gym=True,
        save_code=True,
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    mx.random.seed(args.seed)

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name) for i in range(args.num_envs)],
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "only discrete action space is supported"

    agent = Agent(envs)
    loss_value_and_grad = nn.value_and_grad(agent, loss)
    optimizer = optim.Adam(learning_rate=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = mx.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape)
    actions = mx.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape)
    logprobs = mx.zeros((args.num_steps, args.num_envs))
    rewards = mx.zeros((args.num_steps, args.num_envs))
    dones = mx.zeros((args.num_steps, args.num_envs))
    values = mx.zeros((args.num_steps, args.num_envs))

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = mx.array(next_obs)
    next_done = mx.zeros(args.num_envs)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.learning_rate = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            # with no grad:
            s_t = time.time()
            action, logprob, _, value = agent.get_action_and_value(next_obs)
            values[step] = value.flatten()
            action_time = time.time() - s_t
            wandb.log({"times/forward_pass_time": action_time})
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            s_t = time.time()
            next_obs, reward, terminations, truncations, infos = envs.step(np.array(action))
            step_time = time.time()-s_t
            wandb.log({"times/step_time": step_time})
            next_done = np.logical_or(terminations, truncations)
            rewards[step] = mx.array(reward).reshape(-1)
            next_obs, next_done = mx.array(next_obs), mx.array(next_done)
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        wandb.log({"charts/episodic_return": info['episode']['r'],
                                   "charts/episodic_length": info['episode']['l']})

        # bootstrap value if not done
        # with no grad:
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = mx.zeros_like(rewards)
        lastgaelam = 0
        for t in reversed(range(args.num_steps)):
            if t == args.num_steps - 1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t + 1]
                nextvalues = values[t + 1]
            delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
        returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        approx_kl = -999
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = mx.array(b_inds[start:end])

                mb_advantages = b_advantages[mb_inds]
                mb_obs = b_obs[mb_inds]
                mb_actions = b_actions[mb_inds]  # .astype(mx.int32)
                mb_rewards = b_returns[mb_inds]
                mb_log_probs = b_logprobs[mb_inds]
                mb_values = b_values[mb_inds]

                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.var().sqrt() + 1e-8)
                s_t = time.time()
                (lvalue, approx_kl), grad = loss_value_and_grad(agent, mb_obs, mb_actions,
                                                                mb_log_probs, mb_advantages,
                                                                mb_rewards, mb_values)
                loss_time = time.time()-s_t
                wandb.log({"times/loss_time": loss_time})
                s_t = time.time()
                optimizer.update(agent, grad)
                mx.eval(agent.parameters(), optimizer.state, lvalue)
                weight_time = time.time() - s_t
                wandb.log({"times/weight_update_time": weight_time})

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = np.array(b_values), np.array(b_returns)
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
        wandb.log({"losses/clipfrac": np.mean(clipfracs),
                   "charts/learning_rate": optimizer.learning_rate.item(),
                   "losses/explained_variance": explained_var,
                   "steps_per_sec": int(global_step / (time.time() - start_time))})

    envs.close()
