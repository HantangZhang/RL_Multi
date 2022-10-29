from envs.test_env.multiagentenv import MultiAgentEnv
import numpy as np
from gym.spaces import Discrete


class TestEnv(MultiAgentEnv):

    def __init__(self,
                 args):

        self.n_agents = args.n_agents

        # obsevations and state

        # actions
        self.n_action_move = 3
        self.n_action = 6

        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        for i in range(self.n_agents):
            self.action_space.append(Discrete(self.n_action))
            self.observation_space.append(self.get_obs_size())
            self.share_observation_space.append(self.get_obs_size())


    def _launch(self):
        print('开始启动testenv')

    def _restart(self):
        print('重启env')

    def reset(self):
        self._episode_steps = 0
        # if self._episode_count == 0:
        #     # Launch StarCraft II
        #     self._launch()
        # else:
        #     self._restart()

        local = self.get_obs()
        global_state = [self.get_state(agent_id) for agent_id in range(self.n_agents)]
        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(i))

        return local, global_state, available_actions

    def step(self, actions):
        local_obs = self.get_obs()
        global_state = [self.get_state(agent_id) for agent_id in range(self.n_agents)]

        reward = self.reward_battle()
        rewards = [[reward]] * self.n_agents

        dones = np.zeros((self.n_agents), dtype=bool)
        infos = [{} for i in range(self.n_agents)]

        available_actions = []
        for i in range(self.n_agents):
            available_actions.append(self.get_avail_agent_actions(i))

        return local_obs, global_state, rewards, dones, infos, available_actions

    def get_obs_agent(self, agent_id):

        move_obs_dim = self.get_obs_move_obs_size()
        enemy_obs_dim = self.get_obs_enemy_obs_size()
        owe_obs_dim = self.get_obs_own_obs_size()

        move_obs = np.zeros(move_obs_dim, dtype=np.float32)
        enemy_obs = np.zeros(enemy_obs_dim, dtype=np.float32)
        owe_obs = np.zeros(owe_obs_dim, dtype=np.float32)

        agent_obs = np.concatenate((move_obs.flatten(),
                                    enemy_obs.flatten(),
                                    owe_obs.flatten()))

        return agent_obs

    def get_obs(self):
        agents_obs = [self.get_obs_agent(i) for i in range(self.n_agents)]
        return agents_obs

    def get_state(self, agint_id=-1):

        obs_concat = np.concatenate(self.get_obs(), axis=0).astype(np.float32)
        # obs_concat = self.get_obs()
        return obs_concat

    def reward_battle(self):
        # reward function

        reward = np.random.randint(1, 18)

        return reward

    def get_avail_agent_actions(self, agent_id):
        avail_actions = [0] * self.n_action
        return avail_actions

    def seed(self, seed):
        self._seed = seed

    def get_obs_size(self):
        own_obs = self.get_obs_own_obs_size()
        move_obs = self.get_obs_move_obs_size()
        enemy_obs = self.get_obs_enemy_obs_size()

        return [own_obs + enemy_obs + move_obs]



    def get_obs_move_obs_size(self):

        return 5

    def get_state_move_obs_size(self):
        return 5

    def get_obs_own_obs_size(self):
        return 7

    def get_state_own_obs_size(self):
        return 7

    def get_obs_enemy_obs_size(self):
        return 4

    def get_state_enemy_obs_size(self):
        return 4