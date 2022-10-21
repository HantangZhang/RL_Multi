
import os
import torch
import numpy as np
from algorithm.mat_trainer import MATTrainer
from algorithm.transformer_policy import TransformerPolicy
from algorithm.shared_buffer import SharedReplayBuffer

def _t2n(x):
    """Convert torch tensor to a numpy array."""
    return x.detach().cpu().numpy()

class Runner(object):

    def __init__(self, config):

        # 参数
        self.all_args = config['all_args']
        # 调用所有env函数
        self.envs = config['envs']
        # 评估env
        self.eval_envs = config['eval_envs']

        self.device = config['device']
        self.num_agents = config['num_agents']

        # 是否使用cetd吧，还不确定
        self.use_centralized_V = self.all_args.use_centralized_V

        # 5v5需要关注一下这个参数
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state

        # env总推进步数
        self.num_env_steps = self.all_args.num_env_steps

        # episode的概念还未清楚
        self.episode_length = self.all_args.episode_length

        # 同时并行几个env
        self.n_rollout_threads = self.all_args.n_rollout_threads

        # 学习速率的更新策略
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay

        # 隐藏层神经元个数
        self.hidden_size = self.all_args.hidden_size

        # 为什么用到了recurrent layer
        self.recurrent_N = self.all_args.recurrent_N

        # interval相关的东西
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.log_interval = self.all_args.log_interval

        # dir，如果有pretrained model
        self.model_dir = self.all_args.model_dir

        # 效果展示和路径相关的东西，需要的时候再看
        # if self.use_wandb:
        #     self.save_dir = str(wandb.run.dir)
        #     self.run_dir = str(wandb.run.dir)
        # else:
        #     self.run_dir = config["run_dir"]
        #     self.log_dir = str(self.run_dir / 'logs')
        #     if not os.path.exists(self.log_dir):
        #         os.makedirs(self.log_dir)
        #     self.writter = SummaryWriter(self.log_dir)
        #     self.save_dir = str(self.run_dir / 'models')
        #     if not os.path.exists(self.save_dir):
        #         os.makedirs(self.save_dir)

        # 确定一下5v5这里有没有用
        share_observation_space = self.envs.share_observation_space[0] if self.use_centralized_V else self.envs.observation_space[0]

        print("obs_space: ", self.envs.observation_space)
        print("share_obs_space: ", self.envs.share_observation_space)
        print("act_space: ", self.envs.action_space)

        self.policy = TransformerPolicy(self.all_args, self.envs.observation_space,
                                        share_observation_space, self.envs.action_space[0],
                                        self.num_agents, device=self.device)

        if self.model_dir is not None:
            self.restore(self.model_dir)

        self.trainer = MATTrainer(self.all_args, self.policy, self.num_agents, device=self.device)

        self.buffer = SharedReplayBuffer(
            self.all_args,
            self.num_agents,
            self.envs.observation_space[0],
            share_observation_space,
            self.envs.action_space[0],
            self.all_args.env_name
        )

    def run(self):
        """Collect training data, perform training updates, and evaluate policy."""
        raise NotImplementedError

    def warmup(self):
        """Collect warmup pre-training data."""
        raise NotImplementedError

    def collect(self, step):
        """Collect rollouts for training."""
        raise NotImplementedError

    def insert(self, data):
        """
        Insert data into buffer.
        :param data: (Tuple) data to insert into training buffer.
        """
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        self.trainer.prep_rollout()
        if self.buffer.available_actions is None:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]))
        else:
            next_values = self.trainer.policy.get_values(np.concatenate(self.buffer.share_obs[-1]),
                                                         np.concatenate(self.buffer.obs[-1]),
                                                         np.concatenate(self.buffer.rnn_states_critic[-1]),
                                                         np.concatenate(self.buffer.masks[-1]),
                                                         np.concatenate(self.buffer.available_actions[-1]))
        next_values = np.array(np.split(_t2n(next_values), self.n_rollout_threads))
        self.buffer.compute_returns(next_values, self.trainer.value_normalizer)

    def train(self):
        self.trainer.prep_training()
