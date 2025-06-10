from torch.utils.data import DataLoader, TensorDataset
from til_environment import gridworld
import pickle
import torch

import torch.nn as nn
import torch.nn.functional as F
import numpy as np



# Your flatten function for dict observations
import torch

def flatten_obs(obs):
    def one_hot(val, size):
        vec = torch.zeros(size, dtype=torch.float32)
        vec[val] = 1.0
        return vec

    direction_oh = one_hot(obs['direction'], 4)
    scout_oh = one_hot(obs['scout'], 2)
    step_oh = one_hot(obs['step'], 101)
    location = torch.tensor(obs['location'], dtype=torch.float32).flatten()
    viewcone = torch.tensor(obs['viewcone'], dtype=torch.float32).flatten()

    return torch.cat([direction_oh, location, scout_oh, step_oh, viewcone])


# Supervised Policy network
class SupervisedPolicy(nn.Module):
    def __init__(self, input_dim=144, action_dim=5): #144
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.output = nn.Linear(128, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.output(x)  # Raw logits for classification

env = gridworld.env(
    env_wrappers=[],   # No default wrappers
    render_mode="human",
    debug=True,
    novice=True,
)
env.reset(seed=42)

with open("guard_demos.pkl", "rb") as f:
    data = pickle.load(f)

obs_data = torch.stack([flatten_obs(o) for o, _ in data])
act_data = torch.tensor([a for _, a in data])

dataset = TensorDataset(obs_data, act_data)
loader = DataLoader(dataset, batch_size=64, shuffle=True)

model = SupervisedPolicy(input_dim=obs_data.shape[1], action_dim=env.action_space("player_1").n)
optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
loss_fn = nn.CrossEntropyLoss()

for epoch in range(300):
    for obs_batch, act_batch in loader:
        logits = model(obs_batch)
        loss = F.cross_entropy(logits, act_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # Save the trained model
        torch.save(model.state_dict(), "guard_policy.pt")
        # print("Model saved to guard_policy.pt")

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
