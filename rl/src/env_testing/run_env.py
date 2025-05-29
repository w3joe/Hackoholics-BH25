from til_environment import gridworld

env = gridworld.env(
    env_wrappers=[],   # No default wrappers
    render_mode="human",
    debug=True,
    novice=True,
)
env.reset(seed=42)

for agent in env.agent_iter():
    observation, reward, termination, truncation, info = env.last()

    if termination or truncation:
        env.step(None)
    else:
        if agent == "player_0":
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

        env.step(action)

env.close()
