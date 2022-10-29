import time
import torch
import numpy as np

from runners.base_runner import Runner

def _t2n(x):
    return x.detach().cpu().numpy()

class SMACRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for SMAC. See parent class for details."""
    def __init__(self, config):
        super(SMACRunner, self).__init__(config)


    def run(self):
        self.warmup()

        start =time.time()

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
                values, actions, action_log_probs, = self.collect(step)

                obs, share_obs, rewards, dones, infos, available_actions = self.envs.step(actions)

                data = obs, share_obs, rewards, dones, infos, available_actions, \
                       values, actions, action_log_probs

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
            # 整个计算流程是通过base_runner中的compute，里面再调用buffer当中的compute return
            self.compute()
            #
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads




    def warmup(self):
        # reset env
        obs, share_obs, available_actions = self.envs.reset()
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].copy()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].copy()
            self.buffer[agent_id].available_actions[0] = available_actions[:, agent_id].copy()

    @torch.no_grad()
    def collect(self, step):
        # 存着每个步长每个智能体的value, 如果是5个智能体，那么列表的长度就为5，然后每个位置存着子列表，列表的长度为10，对应着起的10个环境
        value_collector = []
        action_collector = []
        action_log_prob_collector = []

        for agent_id in range(self.num_agents):

            self.trainer[agent_id].prep_rollout()
            # 每个agent有自己的一个buffer，从每个agent的自己的buffer取出obs的数据，然后更新

            value, action, action_log_prob = self.trainer[agent_id].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].masks[step],
                self.buffer[agent_id].available_actions[step])

            value_collector.append(_t2n(value))
            action_collector.append(_t2n(action))
            action_log_prob_collector.append(_t2n(action_log_prob))

        # 将维度转换为(10, 5, 1)
        values = np.array(value_collector).transpose(1, 0, 2)
        actions = np.array(action_collector).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_prob_collector).transpose(1, 0, 2)

        return values, actions, action_log_probs

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, available_actions, \
        values, actions, action_log_probs = data

        # dones是（10，5）的向量, axiis= 1 检查每一行是否全部是一样的，如果不是就是false，都是0也是false，都是1就是true
        dones_env = np.all(dones, axis=1)

        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        # (dones == True).sum()会统计有几个环境已经结束了，如果是4个结束了那么就=4
        # 下面返回的就是(4, 5, 1)
        masks[dones_env == True] = np.zeros(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        active_masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        active_masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)
        active_masks[dones_env == True] = np.ones(((dones_env == True).sum(), self.num_agents, 1), dtype=np.float32)

        if not self.use_centralized_V:
            share_obs = obs
        for agent_id in range(self.num_agents):
            # 这里share_obs的维度是(10, 5, 16)，对应着10个env，5个智能体，obsdim是16
            # share_obs[:, agent_id]的维度是(10, 16)，不同的agint_id就代表取的是对应智能体的obs数据
            # share_obs[:,:, x]就代表以第三维取，维度将会是(10,5,1)
            # 所以这里放入buffer就是把每个智能体对应的obs信息放入指定位置
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id],
                                         actions[:, agent_id],
                                         action_log_probs[:, agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id],
                                         available_actions[:, agent_id])
