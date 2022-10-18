import torch
from algorithm.utils import get_shape_from_obs_space, update_linear_schedule
from ma_transformer import MultiAgentTransformer as MAT


'''
MultiAgentTransformer
1. 将数据输入到encoder当中：
输入数据包括：
state_dim, obs_dim, n_block, n_embd, n_head, n_agent, encode_state
state_dim : 现在认为整个mpe没有用到state_dim
obs_dim: 环境obs数据特征的纬度，例如mpe就是21
n_embd：对于obs的数据，先把它embd成多少维的数据

encoder结构包括：encoderblock，mlp
    其中encoderblock中包括：selfattention和mlp



'''




class TransformerPolicy:

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr
        self.weight_decay = args.weight_decay

        self.opti_eps = args.opti_eps

        if act_space.__class__.__name__ == 'Box':
            self.action_type = 'Continuous'
        else:
            self.action_type = 'Discrete'

        self.obs_dim = get_shape_from_obs_space(obs_space)[0]
        self.share_sob_dim = get_shape_from_obs_space(cent_obs_space)[0]

        if self.action_type == 'Discrete':
            self.act_dim = act_space.n
            self.act_num = 1
        else:
            print("act high: ", act_space.high)
            self.act_dim = act_space.shape[0]
            self.act_num = self.act_dim

        print("obs_dim: ", self.obs_dim)
        print("share_obs_dim: ", self.share_obs_dim)
        print("act_dim: ", self.act_dim)

        self.num_agents = num_agents

        self.transformer = MAT(self.obs_dim, self.act_dim, num_agents,
                               n_block=args.n_block, n_embd=args.n_embd, n_head=args.n_head,
                               encode_state=args.encode_state, device=device,
                               action_type=self.action_type, dec_actor=args.dec_actor,
                               share_actor=args.share_actor)

        self.optimizer = torch.optim.Adam(self.transformer.parameters(),
                                          lr=self.lr, eps=self.opti_eps,
                                          weight_decay=self.weight_decay)

    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.optimizer, episode, episodes, self.lr)

    def get_actions(self, obs, available_actions=None, deterministic=False):
        obs = obs.reshape(-1, self.num_agents, self.obs_dim)
        if available_actions is not None:
            available_actions = available_actions.reshape(-1, self.num_agents, self.act_dim)

        actions, action_log_probs, values = self.transformer.get_actions(
                                                                         obs,
                                                                         available_actions,
                                                                        deterministic)

        # 这个把动作全部展开，batch=32，5个智能体，一人一个动作（act_num=1)，就是（160，1）
        actions = actions.view(-1, self.act_num)
        action_log_probs = action_log_probs.view(-1, self.act_num)
        values = values.view(-1, 1)

        return values, actions, action_log_probs



