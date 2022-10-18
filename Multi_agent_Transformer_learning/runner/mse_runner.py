import time
import torch
import numpy as np
from runner.base_runner import Runner


class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        # 总共训练6000步，每次100步就结束，一共起了n个env，那么一共要跑的局数等于
        # 6000 / n / 100
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                #



    def warmup(self):
        # 重置env
        obs = self.envs.reset()

        # replay buffer
        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def