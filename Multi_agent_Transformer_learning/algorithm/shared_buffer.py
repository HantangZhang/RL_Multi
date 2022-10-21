import numpy as np
import torch

from algorithm.utils import get_shape_from_obs_space


def _shuffle_agent_grad(x, y):
    # 以x为6400， y=5举例
    # 经过indices形成的矩阵形状是(2, 6400, 5)
    # rows的形状就是(6400, 5)代表是每个元素行所代表的索引（再加上列就可以索引到原数组每个元素的位置了）
    rows = np.indices((x, y))[0]
    # 那么就是(6400, 5）的矩阵，每一行=[0, 1, 2, 3, 5]
    cols = np.stack([np.arange(y) for _ in range(x)])
    # cols = np.stack([np.random.permutation(y) for _ in range(x)])

class SharedReplayBuffer(object):

    def __init__(self, args, num_agents, obs_space, cent_obs_space, act_space, env_name):
        self.episode_length = args.episode_length
        self.n_rollout_threads = args.n_rollout_threads
        self.hidden_size = args.hidden_size

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(cent_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

        # 初始化一个share_obs，它是一个例如(201, 32, 5 , obs_dim)
        # 一共有32个env，5个智能体，采样200个step
        self.share_obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *share_obs_shape),
                                  dtype=np.float32)
        self.obs = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, *obs_shape), dtype=np.float32)

        self.masks = np.ones((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)

        self.value_preds = np.zeros((self.episode_length + 1, self.n_rollout_threads, num_agents, 1), dtype=np.float32)
        self.returns = np.zeros_like(self.value_preds)
        self.advantages = np.zeros(
            (self.episode_length, self.n_rollout_threads, num_agents, 1), dtype=np.float32
        )

        self.step = 0

    def insert(self, share_obs, obs, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None
               ):
        # 插入就是不断的更新share_obs这个array，每次更新的当前step的内容
        # 当前步长执行action后，新的obs信息
        self.share_obs[self.step + 1] = share_obs.copy()
        self.obs[self.step + 1] = obs.copy()

        # 当前step执行的动作
        self.actions[self.step] = actions.cppy()
        self.action_log_probs[self.step] = action_log_probs.copy()
        # 当前步长的value和执行动作得倒的奖励
        self.value_preds[self.step] = value_preds.copy()
        self.rewards[self.step] = rewards.copy()
        self.masks[self.step + 1] = masks.copy()

        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy()
        if active_masks is not None:
            self.active_masks[self.step + 1] = active_masks.copy()
        if available_actions is not None:
            self.available_actions[self.step + 1] = available_actions.copy()

        self.step = (self.step + 1) % self.episode_length

    def compute_returns(self, next_value, value_normalizer=None):

        # 这里value_preds是一个episode的即(200, 32, 5, 1)
        # 但为什么更新最后一个数据为next_value，即(32, 5, 1)？？
        self.value_preds[-1] = next_value
        gae = 0

        # 对于每一条数据，可能这也是为什么是201条，这样反过来就是从200，199一直到0
        for step in reversed(range(self.rewards.shape[0])):
            if self._use_popart or self._use_valuenorm:
                delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(
                    self.value_preds[step + 1]) * self.masks[step + 1] \
                        - value_normalizer.denormalize(self.value_preds[step])
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                # here is a patch for mpe, whose last step is timeout instead of terminate
                if self.env_name == "MPE" and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
            else:
                delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * \
                        self.masks[step + 1] - self.value_preds[step]
                gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae

                if self.env_name == 'MPE' and step == self.rewards.shape[0] - 1:
                    gae = 0

                self.advantages[step] = gae
                self.returns[step] = gae + self.value_preds[step]

    def feed_forward_generator_transformer(self, advantages, num_mini_batch=None, mini_batch_size=None):
        '''
        利用yield生成器给mlp策略生成训练数据
        生成的数据就是当前episode，根据num_mini_batch将当前的数据分成多个mini_batch_size

        '''
        episode_length, n_rollout_threads, num_agents = self.rewards.shape[0:3]
        # 干嘛用的
        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length,
                          n_rollout_threads * episode_length,
                          num_mini_batch))
            # 32个env，5个智能体，200的length，就是6400条数据
            mini_batch_size = batch_size // num_mini_batch

        # 生成从0到batchsize的一组随机数列
        rand = torch.randperm(batch_size).numpy()
        # i * mini_batch_size:(i + 1) * mini_batch_size 就是把batch_size分为num_mini_batch块，每一块有mini_batch_size个数据
        # 例如10:20， 20:30， 30: 40
        # rand[10:20]就是取出这一块的随机数列，最后sampler是一个列表，长度为num_mini_batch，每一个长度是一个小列表，里面是一块随机数列
        sampler = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]
        rows, cols = _shuffle_agent_grad(batch_size, num_agents)
        # 原先是(201, 32， 5, 155), 取：-1即消掉一行，变为200,32,5,155)
        # -1指的是合并所有维度，2:只要从第三个维度后面的维度，最终就是(6400, 5, 155)
        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        # 同样还是(6400, 5, 155)但是因为cols里面是打乱的，所以打乱了每个agent的顺序，但是上面并没有打乱
        # 下面的内容就是打乱agent的顺序
        share_obs = share_obs[rows, cols]
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        obs = obs[rows, cols]
        actions = self.actions.reshape(-1, *self.actions.shape[2:])
        actions = actions[rows, cols]
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, *self.available_actions.shape[2:])
            available_actions = available_actions[rows, cols]
        value_preds = self.value_preds[:-1].reshape(-1, *self.value_preds.shape[2:])
        value_preds = value_preds[rows, cols]
        returns = self.returns[:-1].reshape(-1, *self.returns.shape[2:])
        returns = returns[rows, cols]
        masks = self.masks[:-1].reshape(-1, *self.masks.shape[2:])
        masks = masks[rows, cols]
        active_masks = self.active_masks[:-1].reshape(-1, *self.active_masks.shape[2:])
        active_masks = active_masks[rows, cols]
        action_log_probs = self.action_log_probs.reshape(-1, *self.action_log_probs.shape[2:])
        action_log_probs = action_log_probs[rows, cols]
        advantages = advantages.reshape(-1, *advantages.shape[2:])
        advantages = advantages[rows, cols]

        for indices in sampler:
            # sampler存的是一个一个minibatch列表，每个列表里是随机的index，如[14,394,123...], 所以叫indices
            # 下面就是依据随机的indices取出对应的shareobs，obs等等数据
            share_obs_batch = share_obs[indices].reshape(-1, *share_obs.shape[2:])
            obs_batch = obs[indices].reshape(-1, *obs.shape[2:])
            actions_batch = actions[indices].reshape(-1, *actions.shape[2:])
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices].reshape(-1, *available_actions.shape[2:])
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices].reshape(-1, *value_preds.shape[2:])
            return_batch = returns[indices].reshape(-1, *returns.shape[2:])
            masks_batch = masks[indices].reshape(-1, *masks.shape[2:])
            active_masks_batch = active_masks[indices].reshape(-1, *active_masks.shape[2:])
            old_action_log_probs_batch = action_log_probs[indices].reshape(-1, *action_log_probs.shape[2:])
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices].reshape(-1, *advantages.shape[2:])

            # 返回一个迭代器，没一个iter就是一个mini_batch_size的数据
            yield share_obs_batch, obs_batch, actions_batch, \
                value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch,\
                adv_targ, available_actions_batch




