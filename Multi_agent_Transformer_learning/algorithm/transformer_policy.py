import torch
from algorithm.utils import get_shape_from_obs_space
from mat.algorithms.mat.algorithm.ma_transformer import MultiAgentTransformer as MAT

class TransformerPolicy:

    def __init__(self, args, obs_space, cent_obs_space, act_space, num_agents, device=torch.device("cpu")):

        self.device = device
        self.lr = args.lr

        self.opti_eps = args.o
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

        self.transformer =


