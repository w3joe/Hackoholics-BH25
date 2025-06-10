from til_environment import gridworld
import pickle
import os

# === Config ===
EPISODES = 10   # Set number of episodes you want to train

TRAINING_AGENT = 2 #Ranges from 0 to 3, where 0 is scout, 1 is bottom left guard, 2 is top right guard, 3 is bottom right guard
SAVE_PATH = "human{TRAINING_AGENT}_demo.pkl"

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
    env.reset(seed=42)
    step = 0
    print(f"\n=== Episode {ep + 1} ===")
    # if ep == 0:
    #     env.reset(seed=42)
    # env.reset(seed=42)
    # env.reset(seed=42)
    # env.reset(seed=42)
    # env.reset(seed=42)
    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        if termination or truncation:
            env.step(None)
        else:
            if step % 4 == TRAINING_AGENT:
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
            step += 1
            if action is not None:
                data.append((obs, action))
                  
with open(SAVE_PATH, "wb") as f:
    pickle.dump(data, f)  

env.close()
print(f"\n[✔] Saved {len(data)} samples to '{SAVE_PATH}'")