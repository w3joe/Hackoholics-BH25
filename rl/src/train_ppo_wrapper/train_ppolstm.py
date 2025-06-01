import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from til_environment import gridworldv2
import time
import os
import shape_wrapper
# docker build -t hackoholics-rl:latest .
# docker run -p 5004:5004 -d hackoholics-rl:latest
# docker run --network none hackoholics-rl:latest
# til submit hackoholics-rl:latest

# attempt 1: reward avg 2000 epoch scout reward 47, guard: 0, coverage 12%
# attempt 2 with visited map and revisit penalty: reward avg 2000 epoch scout_reward 61, guard: 0, coverage 25%

writer = SummaryWriter(log_dir=f"runs/gridworld_shared_ppo_{time.time()}")
gridworld = gridworldv2
class SlidingAverage:
    def __init__(self, name, steps=100):
        self.name = name
        self.steps = steps
        self.t = 0
        self.ns = []
        self.avgs = []
    
    def add(self, n):
        self.ns.append(n)
        if len(self.ns) > self.steps:
            self.ns.pop(0)
        self.t += 1
        if self.t % self.steps == 0:
            self.avgs.append(self.value)

    @property
    def value(self):
        if len(self.ns) == 0: return 0
        return sum(self.ns) / len(self.ns)

    def __str__(self):
        return "%s=%.4f" % (self.name, self.value)
    
    def __gt__(self, value): return self.value > value
    def __lt__(self, value): return self.value < value

# log every 1 episodes
log_every = 50
scout_reward_avg = SlidingAverage('scout reward avg', steps=log_every)
guard_reward_avg = SlidingAverage('guard reward avg', steps=log_every)
scout_coverage_avg = SlidingAverage('scout coverage avg', steps=log_every)


# --- Custom Rewards ---
custom_rewards = {
     gridworld.RewardNames.GUARD_WINS: 10.0,  # Try hinder other guards if negative.
    gridworld.RewardNames.GUARD_CAPTURES: 50.0,  # GUARD_WINS + 50.0 (actual reward)
    gridworld.RewardNames.SCOUT_CAPTURED: -30.0,  # -50.0,  # model should naturally learn to avoid this since this also terminates early.
    gridworld.RewardNames.SCOUT_RECON: 1.0,
    gridworld.RewardNames.SCOUT_MISSION: 5.0,
    gridworld.RewardNames.WALL_COLLISION: -5.0,  # Never do this, just be stationary if needed.
    gridworld.RewardNames.AGENT_COLLIDER: 0.0,  # Doesn't matter, should be 0 to allow sabotage to evolve.
    gridworld.RewardNames.AGENT_COLLIDEE: -1.0,  # Guard avoid getting hit/blocked by other guards.
    gridworld.RewardNames.STATIONARY_PENALTY: -0.2,  # Very minor penalty in case its necessary.
    gridworld.RewardNames.GUARD_TRUNCATION: -20.0,  # Guard should penalize if fail to capture.
    gridworld.RewardNames.SCOUT_TRUNCATION: 0.0,  # Scout should reward for staying alive... or maybe not, since termination should naturally scare scout.
    # TODO: GUARD_STEP should increase nearing end, SCOUT_STEP should decrease nearing end.
    gridworld.RewardNames.GUARD_STEP: -0.05,  # Guard should try and end game.
    gridworld.RewardNames.SCOUT_STEP: 0.0,  # Scout should try and stay alive.
    gridworld.RewardNames.GUARD_MOVE: 0.2,  # Small reward every step for the guard if it moves mostly in one direction.
    # gridworld.RewardNames.SCOUT_MOVE: 0.0,  # Scout has plenty of incentive to move alr.
}


# --- Observation Flattening ---
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
    visited   = obs["visited"].astype(np.float32).flatten()   # NEW
    return np.concatenate([direction_oh, location, scout_oh, step_oh, viewcone, visited])

# --- PPO Policy ---
INPUT_DIM = 144 + 16    # old + visited

class PPOLSTMPolicy(nn.Module):
    def __init__(self, input_dim=144, action_dim=5, hidden_dim=128):
        super().__init__()
        self.hidden_dim = hidden_dim 
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.value_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, lstm_state):
        """
        x: [B, T, input_dim] if sequence input
        lstm_state: (h_0, c_0) — each of shape [1, B, hidden_dim]
        """
        B, T, _ = x.shape

        x = x.view(B * T, -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(B, T, -1)  # Restore time dimension

        # Pass through LSTM
        if lstm_state is None:
            lstm_state = self.init_hidden(B)
        

        h_before, c_before = lstm_state
        # print("Before LSTM step - hidden mean:", h_before.mean().item(), 
        #         "cell mean:", c_before.mean().item())

        lstm_out, new_lstm_state = self.lstm(x, lstm_state)

        h_after, c_after = new_lstm_state
        # print("After LSTM step  - hidden mean:", h_after.mean().item(), 
            # "cell mean:", c_after.mean().item())

        # Optional: check difference
        diff_h = (h_after - h_before).abs().mean().item()
        diff_c = (c_after - c_before).abs().mean().item()
        # print(f"Mean abs difference - hidden: {diff_h:.6f}, cell: {diff_c:.6f}")

        action_logits = self.action_head(lstm_out)
        state_value = self.value_head(lstm_out).squeeze(-1)
        # print("Logits:", action_logits)
        # print("Action probs:", torch.softmax(action_logits.squeeze(1), dim=-1))

        return action_logits, state_value, new_lstm_state
    
    def init_hidden(self, batch_size):
        # Hidden and cell states initialized to zeros
        h0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        c0 = torch.zeros(1, batch_size, self.hidden_dim).to(device)
        return (h0, c0)

# --- GAE ---
def compute_gae(rewards, values, dones, gamma=0.99, lam=0.95):
    advantages = []
    gae = 0
    values = values + [0]
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        advantages.insert(0, gae)
    return advantages

# --- Hyperparameters ---
lr = 1e-4
gamma = 0.99
clip_eps = 0.21
ppo_epochs = 6
batch_size = 256

# --- Setup ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
env = gridworld.env(
        env_wrappers=[shape_wrapper.ExplorationRewardWrapper,shape_wrapper.VisitedChannelWrapper, shape_wrapper.RevisitPenaltyWrapper, shape_wrapper.TurnPenaltyWrapper],
        render_mode=None,
        rewards_dict=custom_rewards,
        debug=True,
        novice=True
    )
env.reset(seed=42)
agent_list = env.agents
frames = []
scout_agents = [agent for agent in agent_list if "scout" in agent]
guard_agents = [agent for agent in agent_list if "guard" in agent]

# Shared policies
shared_policies = {
    "scout": PPOLSTMPolicy(input_dim=INPUT_DIM, action_dim=5).to(device),
    "guard": PPOLSTMPolicy(input_dim=INPUT_DIM, action_dim=5).to(device),
}
shared_optimizers = {
    "scout": optim.Adam(shared_policies["scout"].parameters(), lr=lr),
    "guard": optim.Adam(shared_policies["guard"].parameters(), lr=lr),
}
buffer = {
    "scout": {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []},
    "guard": {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []},
}
lstm_states = {
"scout": shared_policies["scout"].init_hidden(batch_size=1),
"guard": shared_policies["guard"].init_hidden(batch_size=1),
}
reward_history = {agent: [] for agent in agent_list}

# for model saving and debug-logging
epochs = 10000

# create folder to save model weights labelled with epoch time of when code starts running
# start_epoch_time = time.time()
# save_dir = f"../models/lstm_ppo/lstm_ppo_{int(start_epoch_time)}"
# os.makedirs(save_dir, exist_ok=True)

render_episodes = {50, 100, 150, 200,250,300,350, 400, 600, 800, 1000, 1200}

# --- Training Loop ---
for episode in range(epochs):
    env = gridworld.env(
            env_wrappers=[shape_wrapper.VisitedChannelWrapper, shape_wrapper.RevisitPenaltyWrapper, shape_wrapper.TurnPenaltyWrapper],
            render_mode="human" if episode in render_episodes else None,
            rewards_dict=custom_rewards,
            debug=True,
            novice=True
        )
    # Reset LSTM states at episode start for each role
    for role in lstm_states:
        lstm_states[role] = shared_policies[role].init_hidden(batch_size=1)
    env.reset()
    done = False
    guard_reward = 0
    scout_reward = 0
    step_count = 0
    while not done:
        for agent in env.agent_iter():
            step_count += 1

            obs, reward, terminated, truncated, info = env.last()

            done = terminated or truncated

            if obs['scout'] == 1:
                role = "scout" 
                scout_reward += reward
            else:
                role = "guard" 
                guard_reward += reward             

            if obs is None or not all(k in obs for k in ("viewcone", "direction", "location", "scout", "step")):
                env.step(None)
                continue
            #print(info)

            flat_obs = flatten_obs(obs)
            flat_obs_tensor = torch.tensor(flat_obs, dtype=torch.float32).unsqueeze(0).to(device)
            
            policy = shared_policies[role]
            lstm_state = lstm_states[role]  # get current LSTM state
            optimizer = shared_optimizers[role]
            buf = buffer[role]

            with torch.no_grad():
                logits, value, new_lstm_state = policy(flat_obs_tensor.unsqueeze(0), lstm_state)
                lstm_states[role] = new_lstm_state  # update stored state

                probs = torch.softmax(logits, dim=-1)
                action_dist = torch.distributions.Categorical(logits=logits.squeeze(0))
                action = action_dist.sample()
                log_prob = action_dist.log_prob(action)

            buf["obs"].append(flat_obs)
            buf["actions"].append(action.item())
            buf["log_probs"].append(log_prob.item())
            buf["rewards"].append(reward)
            buf["dones"].append(done)
            buf["values"].append(value.item())
            #print(f"Appended to {role} buffer — new size: {len(buf['obs'])}")

            if truncated or terminated:
                print(f"Episode is ending. Agent: {agent}, Role: {role}, reward = {reward}, truncated={truncated}, terminated={terminated}")
                vis = obs["visited"]
                scout_coverage_avg.add(vis.mean()*100)
                scout_reward_avg.add(scout_reward)
                guard_reward_avg.add(guard_reward)
                print(guard_reward, scout_reward)
                print(f"[EPISODE END] at Step {step_count}")
                writer.add_scalar(f"Scout Reward", scout_reward, episode)
                writer.add_scalar(f"Guard Reward", guard_reward, episode)
                env.step(None)
                done = True  # flag to break outer loop
                # Reset LSTM states when episode ends
                for role in lstm_states:
                    lstm_states[role] = shared_policies[role].init_hidden(batch_size=1)
                #break
            else:
                if role == "scout":
                    env.step(4)
                else:
                    env.step(action.item())

        if done:
            break


    # save every 50 episodes
    # if (episode + 1) % 500 == 0:
    #     for role, policy in shared_policies.items():
    #         torch.save(policy.state_dict(), f"{save_dir}/{role}_ppo_policy_{episode+1}.pt")
    #         print(f"model saved to f {save_dir}/{role}_ppo_policy_{episode+1}.pt")

    for role in ["scout", "guard"]:
        buf = buffer[role]
        total_reward = sum(buf["rewards"])
        # print average every 50 episodes
        if episode % 50 == 0:
            if role == "scout":
                #vis = obs["visited"]
                #print("coverage:", vis.mean()*100, "%")
                print(f"[Episode {episode}] {role} - Average Coverage: {scout_coverage_avg}%")

                print(f"[Episode {episode}] {role} - Average Reward: {scout_reward_avg}")
            if role == "guard":
                #guard_reward_avg.add(guard_reward)
                print(f"[Episode {episode}] {role} - Average Reward: {guard_reward_avg}")
        rewards, values, dones = buf["rewards"], buf["values"], buf["dones"]
        advantages = compute_gae(rewards, values, dones, gamma)
        returns = [a + v for a, v in zip(advantages, values)]

        obs_batch = torch.tensor(np.array(buf["obs"]), dtype=torch.float32).to(device)  # shape (T, input_dim)
        obs_batch = obs_batch.unsqueeze(0)  # shape (1, T, input_dim), batch_size=1, seq_len=T

        actions_batch = torch.tensor(buf["actions"]).to(device)
        old_log_probs_batch = torch.tensor(buf["log_probs"]).to(device)
        returns_batch = torch.tensor(returns).to(device)
        advantages_batch = torch.tensor(advantages).to(device)
        advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

        policy = shared_policies[role]
        optimizer = shared_optimizers[role]
        # Initialize hidden state once per role before PPO epochs
        lstm_state = lstm_states[role]  # get current hidden state  

        for _ in range(ppo_epochs):
   
            # Forward pass with the current lstm_state
            logits, values, new_lstm_state = policy(obs_batch, lstm_state)

            # Detach hidden state to avoid backprop through time across epochs
            lstm_state = (new_lstm_state[0].detach(), new_lstm_state[1].detach())  

            probs = torch.softmax(logits, dim=-1)
            dist = Categorical(probs)

            entropy = dist.entropy().mean()
            #print(entropy)
            new_log_probs = dist.log_prob(actions_batch)

            ratio = (new_log_probs - old_log_probs_batch).exp()
            surr1 = ratio * advantages_batch
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages_batch

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = F.mse_loss(values.squeeze(0), returns_batch)
            loss = policy_loss + 0.5 * value_loss - 0.01 * entropy

            writer.add_scalar(f"Loss/policy_{role}", policy_loss.item(), episode)
            writer.add_scalar(f"Loss/value_{role}", value_loss.item(), episode)
            writer.add_scalar(f"Loss/entropy_{role}", entropy.item(), episode)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Save updated hidden state back to lstm_states dict for next training iteration
        lstm_states[role] = lstm_state
        buffer[role] = {"obs": [], "actions": [], "log_probs": [], "rewards": [], "dones": [], "values": []}

    print(f"Episode {episode + 1} completed.")

env.close()

