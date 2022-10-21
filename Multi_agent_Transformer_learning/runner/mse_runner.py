import time
import torch
import numpy as np
from runner.base_runner import Runner
from algorithm.mat_trainer import MATTrainer


def _t2n(x):
    return x.detach().cpu().numpy()

class MPERunner(Runner):
    def __init__(self, config):
        super(MPERunner, self).__init__(config)

    def run(self):
        self.warmup()

        start = time.time()
        # 总共训练6000步，每次100步是一个episode，一共起了n个env，那么一共要跑
        # 6000 / n / 100个episodes
        # 这么理解我总共要求环境推num_env_steps步，那么如果起了n个env，实际只需要每个env
        # 推num_env_steps /n 步 = single_env_step
        # 如果我设置episode_length = 100，即定义每推100步形成一个episode
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # 通过collect从buffer取当前step的value，actions等数据
                # 也就是32个env，5个智能体，就是(32, 5, 1)
                # 同时利用当前step的obs信息，算出下一步应该执行的动作
                values, actions, action_log_probs, actions_env = self.collect(step)
                # 利用这个动作，环境推理出下一步的obs，rewards等
                obs, rewards, dones, infos = self.envs.step(actions_env)
                # 将step+1的obs等信息，和当前step的value打包成data
                data = obs, rewards, dones, infos, values, actions, action_log_probs

                self.insert(data)

        '''
        采集好一个episode后，即有200个新的obs，value
        取最后一步的obs，计算value，叫做next_value
        更新buffer中value_preds中的最后一条数据为next_value
        然后倒着更新，从200，199一直到0，更新每一步的return和advantage
        return = gae + v_step (这里面v都是神经网络算出来的)
        A = GAE = Q - V
        Q = RETURN
        '''
        self.compute()





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
    def collect(self, step):
        self.trainer.prep_rollout()

        # self.buffer.share_obs[step]就是每一条数据的
        # 如share_obs是(201, 32, 5 , obs_dim)
        # share_obs[step]就是(32, 5, obs_dim)
        # 这里训练的时候相当于直接训练所有env在[step]的数据
        # np.concatenate如果只输入一个array，那么会合并前两维，也就是(32, 5, obs_dim)
        # 经过concatenate后就是(160, obs_dim)
        # 这里返回的value就是32个env一个step的每一条数据的value，也就是(160,1)
        value, action, action_log_prob = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.masks[step])
        )

        # 将value利用split函数，从(160,1)变为(32, 5, 1)
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
        action_log_probs = np.array(np.split(_t2n(action_log_prob), self.n_rollout_threads))

        # 将action转为one hot的形式，得倒actions_env
        # 本身action的形状是(32, 5, 1)，经过转换actions_env的形状是(32, 5, 5)
        if self.envs.action_space[0].__class__.__name__ == 'MultiDiscrete':
            for i in range(self.envs.action_space[0].shape):
                uc_actions_env = np.eye(self.envs.action_space[0].high[i] + 1)[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == 'Discrete':
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            raise NotImplementedError

        return values, actions, action_log_probs, actions_env

    def insert(self, data):
        # 这里obs就是一条数据，即(32, 5, obsdim), mpe就是(32, 5, 31)
        obs, rewards, dones, infos, values, actions, action_log_probs = data

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)


        if self.use_centralized_V:
            # 5 * obsdim，即把5个agent的obs融合到一起,(32, 155)
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            # 把155这个维度重复num_agent的次数，即变成（5，155），整个维度就变成（32， 5， 155）
            share_obs = np.expand_dims(share_obs, 1).repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(share_obs, obs, actions, action_log_probs, values, rewards, masks)

