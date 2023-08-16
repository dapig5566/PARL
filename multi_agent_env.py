from gym import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv, ENV_STATE
import numpy as np

NUM_STEPS = 10
NUM_ACTIONS = 3
NUM_AGENTS = 2

OBSERVATION_SPACE = spaces.Box(low=0, high=1, shape=(NUM_AGENTS+1 + NUM_ACTIONS+1,))
ACTION_SPACE = spaces.Discrete(NUM_ACTIONS)

class MultiAgentMockEnv(MultiAgentEnv):
    def __init__(self, *args, n_steps=NUM_STEPS, n_actions=NUM_ACTIONS, n_agents=NUM_AGENTS, **kwargs):
        super(MultiAgentMockEnv, self).__init__()
        
        self.max_episode_steps = n_steps
        self.n_actions = n_actions + 1
        self.n_agents = n_agents
        self.n_signs = self.n_agents + 1
        self.state = None

        

        self.observation_space = OBSERVATION_SPACE
        self.action_space = ACTION_SPACE

    def reset(self, **kwargs):
        self.steps = 0
        self.state = None

        return self.get_observation()
    
    def get_observation(self):
        sign = np.random.choice(self.n_signs)
        action = np.random.choice(self.n_actions)
        self.state = [sign, action]

        sign = np.eye(self.n_signs)[sign]
        action = np.eye(self.n_actions)[action]

        obs = np.concatenate([sign, action], axis=0)
        
        obs_dict = {}
        for agent_id in range(self.n_agents):
            obs_dict[agent_id] = obs

        return obs_dict
    
    def step(self, actions):
        init_r = 5
        rewards = {agent_id: init_r for agent_id in range(self.n_agents) }
        if self.state[0] == self.n_signs-1:
            for agent_id, agent_action in actions.items():
                # agent_action = np.random.choice(list(range(self.n_actions)), p=p)
                if agent_action != self.state[1]:
                    r = 0
                    rewards[agent_id] = r
                    
        else:
            for agent_id, agent_action in actions.items():
                # agent_action = np.random.choice(list(range(self.n_actions)), p=p)
                if (agent_id == self.state[0] and agent_action != self.state[1]) or\
                    (agent_id != self.state[0] and agent_action != self.n_actions-1):
                    r = 0
                    rewards[agent_id] = r
                    
           
        self.steps += 1

        # average_reward = r / self.n_agents
        
        # rewards = {agent_id: average_reward for agent_id in range(self.n_agents)}
        terminateds = {"__all__": self.steps >= self.max_episode_steps}
        
        return self.get_observation(), rewards, terminateds, {}
    

class MockEnvWithGroupedAgents(MultiAgentEnv):
    def __init__(self, *args, **kwargs):
        super().__init__()
        env = MultiAgentMockEnv()
        tuple_obs_space = spaces.Tuple([env.observation_space, env.observation_space])
        tuple_act_space = spaces.Tuple([env.action_space, env.action_space])

        self.env = env.with_agent_groups(
            groups={"agents": [0, 1]},
            obs_space=tuple_obs_space,
            act_space=tuple_act_space,
        )
        self.observation_space = self.env.observation_space
        self.action_space = self.env.action_space
        self._agent_ids = {"agents"}

    def reset(self, *, seed=None, options=None):
        return self.env.reset()

    def step(self, actions):
        return self.env.step(actions)