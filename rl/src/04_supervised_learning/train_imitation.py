from til_environment import gridworld
import pickle
import os

# === Config ===
EPISODES = 5   # Set number of episodes you want to collect
SAVE_PATH = "guard1_demos.pkl"

# === Load previous data if exists ===
if os.path.exists(SAVE_PATH):
    with open(SAVE_PATH, "rb") as f:
        data = pickle.load(f)
else:
    data = []

env = gridworld.env(
    env_wrappers=[],   # No default wrappers
    render_mode="human",
    debug=True,
    novice=True,
)
for ep in range(EPISODES):
    print(f"\n=== Episode {ep + 1} ===")
    if ep == 0:
        env.reset(seed=42)
    env.reset(seed=42)
    env.reset(seed=42)
    env.reset(seed=42)
    env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
        else:
            if agent == "player_1":
                keyboardInput = input("WASD? ")
                def switch(keyboardInput):
                    if keyboardInput == "w":
                        return 0
                    elif keyboardInput == "s":
                        return 1
                    elif keyboardInput == "a":
                        return 2
                    elif keyboardInput == "d":
                        return 3
                    else:
                        return 4  # NOOP or default
                action = switch(keyboardInput)
            else:
                # Other agents act randomly (or you could insert policies here)
                action = env.action_space(agent).sample()
            obs = env.observe(agent)
            env.step(action)
            if action is not None:
                data.append((obs, action))
                  
with open("guard_demos.pkl", "wb") as f:
    pickle.dump(data, f)  

env.close()
print(f"\n[✔] Saved {len(data)} samples to '{SAVE_PATH}'")