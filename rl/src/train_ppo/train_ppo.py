import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter

import time
import os
writer = SummaryWriter(log_dir=f"runs/train_base_ppo_{time.time()}")

import numpy as np

# Assume your gridworld env is imported properly:
from til_environment import gridworld

model_path="../models/base_ppo/"
custom_rewards = {
    # Strong terminal rewards to match evaluation
    gridworld.RewardNames.SCOUT_CAPTURED: -50,
    gridworld.RewardNames.GUARD_CAPTURES: 50,
    
    # Dense positive feedback for exploring + mission progress
    gridworld.RewardNames.SCOUT_RECON: 1,      # matches eval
    gridworld.RewardNames.SCOUT_MISSION: 5,    # matches eval
    
    # Mild penalty for being idle or inefficient
    gridworld.RewardNames.STATIONARY_PENALTY: -0.1,
    gridworld.RewardNames.SCOUT_STEP: -0.01,
    gridworld.RewardNames.GUARD_STEP: -0.01,
    
    # Minor punishment to discourage collisions
    gridworld.RewardNames.AGENT_COLLIDER: -0.2,
    gridworld.RewardNames.WALL_COLLISION: -0.2,
    
    # Truncation discourages stalling till timeout
    gridworld.RewardNames.SCOUT_TRUNCATION: -1,
    gridworld.RewardNames.GUARD_TRUNCATION: -1,
    
    # Winning as guard (if game has that notion)
    gridworld.RewardNames.GUARD_WINS: 2,
}


# Your flatten function for dict observations
def flatten_obs(obs):
    def one_hot(val, size):
        vec = np.zeros(size, dtype=np.float32)
        vec[val] = 1.0
        return vec

    direction_oh = one_hot(obs['direction'], 4)
    scout_oh = one_hot(obs['scout'], 2)
    step_oh = one_hot(obs['step'], 101)
    location = np.array(obs['location'], dtype=np.float32).flatten()
    viewcone = np.array(obs['viewcone'], dtype=np.float32).flatten()

    return np.concatenate([direction_oh, location, scout_oh, step_oh, viewcone])

# PPO Policy network
class PPOPolicy(nn.Module):
    def __init__(self, input_dim=144, action_dim=5):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.action_head = nn.Linear(128, action_dim)
        self.value_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_logits = self.action_head(x)
        state_value = self.value_head(x).squeeze(-1)
        return action_logits, state_value

lr = 1e-4              # Faster early learning; reduce if instability occurs 1e-4
gamma = 0.99           # Slightly shorter reward horizon to improve convergence 0.99
clip_eps = 0.2          # Allow more exploration in policy updates 0.2
epochs = 4000             # Shorter training for faster iteration/testing
ppo_epochs = 4            # Fewer PPO updates per cycle (safer for small batch learning) 4
batch_size = 64          # More stable updates, if memory allows 64

# Constants
initial_lr = 1e-4
final_lr = 5e-5
total_decay_steps = 1000
def linear_decay(initial_value, final_value, current_step, total_steps):
    """Linearly decay a value from initial to final over total_steps."""
    ratio = min(current_step / total_steps, 1.0)
    return initial_value + ratio * (final_value - initial_value)

#Trial 8 finished with value: -0.1650000000000842 and parameters: {'player_0_lr': 2.2846848480845284e-05, 'player_0_gamma': 0.9705600062723958, 'player_0_clip_eps': 0.1333830064313426, 'player_0_batch_size': 128, 'player_0_ppo_epochs': 10}. Best is trial 8 with value: -0.1650000000000842.

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#3e-4 2000 128
#1e-4 4000 128
# Main training loop
env = gridworld.env(env_wrappers=[], render_mode=None,rewards_dict=custom_rewards, debug=False, novice=True)
env.reset(seed=42)

# Get observation and action spaces info
agent_list = env.agents

# Build one policy per agent (independent PPO)
policies = {}
optimizers = {}
for agent in agent_list:
    action_dim = env.action_space(agent).n
    policies[agent] = PPOPolicy(input_dim=144, action_dim=action_dim).to(device)
    optimizers[agent] = optim.Adam(policies[agent].parameters(), lr=lr)

# Experience buffer per agent (simple lists)
buffer = {agent: {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []} for agent in agent_list}

def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    return advantages

reward_history = {agent: [] for agent in agent_list}

# Training loop
for episode in range(epochs):
    env.reset()
    done = False
    while not done:
        for agent in env.agent_iter():
            obs, reward, terminated, truncated, info = env.last()
            done = terminated or truncated

            flat_obs = flatten_obs(obs)
            flat_obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)

            policy = policies[agent]

            new_lr = linear_decay(initial_lr, final_lr, episode, total_decay_steps)
            for param_group in optimizers[agent].param_groups:
                param_group['lr'] = new_lr
            optimizer = optimizers[agent]

            if done:
                action = None  # Must pass None when agent is done
                log_prob = None
                value = None
            else:
                with torch.no_grad():
                    logits, value = policy(flat_obs_tensor)
                    probs = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)

            env.step(action if action is None else action.item())

            # Save experience only if action was taken (agent alive)
            if action is not None:
                buf = buffer[agent]
                buf["obs"].append(flat_obs)
                buf["actions"].append(action.item())
                buf["log_probs"].append(log_prob.item())
                buf["rewards"].append(reward)
                buf["dones"].append(done)
                buf["values"].append(value.item())

            if done:
                break

    for agent in agent_list:
        total_reward = sum(buffer[agent]["rewards"])
        reward_history[agent].append(total_reward)
        print(f"Agent {agent} - Total reward this episode: {total_reward}")
        writer.add_scalar(f"EpisodeReward/{agent}", reward, episode)
        

     # Save trained policies at the end of the episode
    for agent_id, policy in policies.items():
        torch.save(policy.state_dict(), f"{model_path}{agent_id}_ppo_policy.pt")


    # After episode ends, do PPO update for each agent
    for agent in agent_list:
        buf = buffer[agent]
        rewards = buf["rewards"]
        values = buf["values"]
        dones = buf["dones"]

        advantages = compute_gae(rewards, values, dones, gamma)
        returns = [adv + val for adv, val in zip(advantages, values)]

        #obs_batch = torch.tensor(buf["obs"], dtype=torch.float32).to(device)
        obs_batch = torch.tensor(np.array(buf["obs"]), dtype=torch.float32).to(device)

        actions_batch = torch.tensor(buf["actions"]).to(device)
        old_log_probs_batch = torch.tensor(buf["log_probs"]).to(device)
        returns_batch = torch.tensor(returns).to(device)
        advantages_batch = torch.tensor(advantages).to(device)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        policy = policies[agent]
        optimizer = optimizers[agent]

        for _ in range(ppo_epochs):
            logits, values = policy(obs_batch)
            probs = torch.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)

            entropy = dist.entropy().mean()
            new_log_probs = dist.log_prob(actions_batch)

            ratio = (new_log_probs - old_log_probs_batch).exp()
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_batch

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values, returns_batch)

            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            writer.add_scalar(f"Loss/policy_{agent}", policy_loss.item(), episode)
            writer.add_scalar(f"Loss/value_{agent}", value_loss.item(), episode)
            writer.add_scalar(f"Loss/entropy_{agent}", entropy.item(), episode)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Clear buffer
        buffer[agent] = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

    print(f"Episode {episode + 1} completed and policies saved.")

    if(episode%100 == 0):
        # Load trained policies and set to eval mode
        for agent in agent_list:
            policies[agent].load_state_dict(torch.load(f"{model_path}{agent}_ppo_policy.pt"))
            policies[agent].eval()

        num_eval_episodes = 8
        total_rewards = {agent: 0.0 for agent in agent_list}

        for episode in range(num_eval_episodes):
            env.reset()
            done = False

            while not done:
                for agent in env.agent_iter():
                    obs, reward, terminated, truncated, info = env.last()
                    done = terminated or truncated

                    if done:
                        action = None  # When done, action must be None
                    else:
                        flat_obs = flatten_obs(obs)
                        flat_obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0)

                        with torch.no_grad():
                            logits, _ = policies[agent](flat_obs_tensor)
                            probs = torch.softmax(logits, dim=-1)
                            dist = torch.distributions.Categorical(probs)
                            action = dist.sample().item()

                    env.step(action)

                    total_rewards[agent] += reward

                    if done:
                        break

            print(f"Episode {episode + 1} finished.")

        print("Average rewards per agent over evaluation:")
        for agent, total_reward in total_rewards.items():
            avg_reward = total_reward / num_eval_episodes
            print(f"Agent {agent}: {avg_reward}")

env.close()





