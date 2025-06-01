import functools
import numpy as np
from pettingzoo.utils.env import AECEnv, AgentID, ObsType, ActionType
from pettingzoo.utils.wrappers.base import BaseWrapper
from gymnasium.spaces import Box, Dict

# assuming scout always has the same start point
SCOUT_START_POS = np.array([1, 1])

class RevisitPenaltyWrapper(BaseWrapper[AgentID, ObsType, ActionType]):

    def __init__(self, env: AECEnv[AgentID, ObsType, ActionType], penalty: float = 0.3):
        super().__init__(env)
        self.penalty = penalty
        self.visited_positions: dict[AgentID, set[tuple[int, int]]] = {}

    # when episode reset
    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # all agents exist *after* super().reset()
        self.visited_positions = {agent: set() for agent in self.agents}

    # main logic
    def step(self, action: ActionType):
        agent = self.agent_selection

        super().step(action)                      

        # fetch the observation emitted *after* the step
        obs, reward, term, trunc, info = self.last()

        if obs is None or "location" not in obs:
            return                                # nothing to do

        pos = tuple(np.asarray(obs["location"]))  # (row, col)

        # apply penalty before adding pos to visited 
        if agent in self.rewards and pos in self.visited_positions[agent]:
            self.rewards[agent] -= self.penalty   # subtract, don’t overwrite!

        # mark tile as visited (only once per agent per episode) 
        self.visited_positions[agent].add(pos)

    
    def observe(self, agent: AgentID) -> ObsType | None:
        return super().observe(agent)

    @functools.lru_cache(None)
    def observation_space(self, agent: AgentID):
        return super().observation_space(agent)
    


class VisitedChannelWrapper(BaseWrapper):
    #Adds `visited` (uint8[H,W]) to every observation.
    def __init__(self, env):
        super().__init__(env)
        self.visited = None          # will hold a 2-D bool grid

    
    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        size = self.unwrapped.size            # 16 in the TIL gridworld
        self.visited = np.zeros((size, size), dtype=np.uint8)
        # mark spawn tiles as visited
        for loc in self.unwrapped.agent_locations.values():
            self.visited[tuple(loc)] = 1
        self._inject_channel()
        return self.observe(self.agent_selection), {}

    
    def step(self, action):
        super().step(action)
        # after env movement, mark the current agent’s new tile
        loc = self.unwrapped.agent_locations[self.agent_selection]
        self.visited[tuple(loc)] = 1
        self._inject_channel()

    
    def observe(self, agent):
        obs = super().observe(agent).copy()
        obs["visited"] = self.visited.copy()      # uint8 mask 0/1
        return obs

    # helper: push visited map into every agent’s observation dict
    def _inject_channel(self):
        for ag in self.agents:
            if ag in self.observations:
                self.observations[ag]["visited"] = self.visited.copy()

    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        space = super().observation_space(agent)
        size = self.unwrapped.size
        
        return Dict({**space.spaces,
                     "visited": Box(0, 1, shape=(size, size), dtype=np.uint8)})

class TurnPenaltyWrapper(BaseWrapper):
    TURN_PENALTY = 0.1      

    def __init__(self, env):
        super().__init__(env)
        self._last_dir = {}

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        # start with each agent’s current dir
        self._last_dir = {a: self.observe(a)["direction"]
                          for a in self.agents}

    def step(self, action):
        agent = self.agent_selection
        super().step(action)

        obs, reward, term, trunc, _ = self.last()
        if obs is None:
            return                       # dead agent

        cur_dir = obs["direction"]
        if cur_dir != self._last_dir[agent] and agent in self.rewards:
            self.rewards[agent] -= self.TURN_PENALTY
        self._last_dir[agent] = cur_dir