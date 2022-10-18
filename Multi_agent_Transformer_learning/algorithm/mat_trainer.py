import numpy as np
import torch
import torch.nn as nn

class MATTrainer:

    def __init__(self,
                 args,
                 policy,
                 num_agents,
                 device=torch.device('cpu')):

        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.policy = policy
        self.num_agents = num_agents

        self.clip_param = args.clip_param
        self.ppo_epoch = args.ppo_epoch
        self.num_mini_batch = args.num_mini_batch
        # self.data_chunk_length = args.data_chunk_length 应该rnn采用得到
        self.value_loss_coef = args.value_loss_coef
        self.entropy_coef = args.entropy_coef
        self.max_grad_norm = args.max_grad_norm
        self.huber_delta = args.huber_delta

