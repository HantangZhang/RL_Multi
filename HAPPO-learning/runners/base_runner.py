import torch
import numpy as np
from algorithms.happo_trainer import HAPPO as TrainAlgo
from algorithms.happo_policy import HAPPO_Policy
from utils.separated_buffer import SeparatedReplayBuffer

def _t2n(x):
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

        self.policy = []
        # 每个agent有自己单独的happo_policy策略
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

            # policy network
            po = HAPPO_Policy(self.all_args,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device=self.device)
            self.policy.append(po)

        # 还未研究
        # if self.model_dir is not None:
        #     self.restore()

        # 每个智能体都有自己的一个trainer
        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):

            tr = TrainAlgo(self.all_args, self.policy[agent_id], device = self.device)

            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]

            bu = SeparatedReplayBuffer(self.all_args,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)


    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()

            # 从buffer中拿出的每个agent的self.buffer[agent_id].share_obs数据维度是(201, 10, 16)
            # -1就是取出最后一条数据
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                  self.buffer[agent_id].masks[-1])
            next_value = _t2n(next_value)
            # 利用最后一条更新整个buffer里面return的数据
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        factor = np.ones((self.episode_length, self.n_rollout_threads, 1), dtype=np.float32)

        # 将agent打乱顺序，按次序每次更新每个agent
        for agent_id in torch.randperm(self.num_agents):
            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            # 这一维的具体含义还没研究，得接到5v5环境具体适配
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[
                                                                                   agent_id].available_actions.shape[
                                                                               2:])
            # 算法目前就写happo，不管hatrpo
            # 这里每个self.buffer[agent_id].obs的维度就是每个epsiode中对应agent的obs数据，维度是(201, 10, 16)
            # self.buffer[agent_id].obs[:-1]就是不要最后一条，也就是200条，然后整个200和10，保留obsdim
            # 最后 self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]的维度就是(2000,16)

            old_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:],
            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
            available_actions,
            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            )
        # 研究到train的作用，和old与new的区别，为啥是new
        # 每个agent单独训练,调用happo_trianer里面的train
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            new_actions_logprob, _ = self.trainer[agent_id].policy.actor.evaluate_actions(
                self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                available_actions,
                self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            # 由论文factor就是大M，实际就是重要性采样，torch.prod(torch.exp(new_actions_logprob-old_actions_logprob)就是计算重要性采样
            factor = factor*_t2n(torch.prod(torch.exp(new_actions_logprob-old_actions_logprob),dim=-1).reshape(self.episode_length,self.n_rollout_threads,1))
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos


