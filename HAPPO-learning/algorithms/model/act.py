import torch
import torch.nn as nn
from algorithms.model.distributions import Categorical


class ACTLayer(nn.Module):

    def __init__(self, action_space, inputs_dim, use_orthogonal, gain, args=None):
        super(ACTLayer, self).__init__()

        self.mixed_action = False

        self.action_type = action_space.__class__.__name__

        # 暂时只把distrere放在这里，连续和multidiscrete后续再梳理
        if action_space.__class__.__name__ == "Discrete":
            action_dim = action_space.n
            # 初始化了一个model，它可以调用forward
            self.action_out = Categorical(inputs_dim, action_dim, use_orthogonal, gain)

    def forward(self, x, available_actions=None, deterministic=False):
        if self.mixed_action :
            actions = []
            action_log_probs = []
            for action_out in self.action_outs:
                action_logit = action_out(x)
                action = action_logit.mode() if deterministic else action_logit.sample()
                action_log_prob = action_logit.log_probs(action)
                actions.append(action.float())
                action_log_probs.append(action_log_prob)

            actions = torch.cat(actions, -1)
            action_log_probs = torch.sum(torch.cat(action_log_probs, -1), -1, keepdim=True)

        else:
            action_logits = self.action_out(x, available_actions)
            actions = action_logits.mode() if deterministic else action_logits.sample()
            action_log_probs = action_logits.log_probs(actions)
        return actions, action_log_probs

    def evaluate_actions(self, x, action, available_actions=None, active_masks=None):
        # dist_entropy 指的是给定输入得到的动作分布的熵

        action_logits = self.action_out(x, available_actions)
        action_log_probs = action_logits.log_probs(action)

        if active_masks is not None:
            if self.action_type == 'Discrete':
                dist_entropy = (action_logits.entropy() * active_masks.squeeze(-1)).sum() / active_masks.sum()

        else:
            dist_entropy = action_logits.entropy().mean()

        # 如果epsiode长度是200，有10个env，一共就是2000条数据，2000个步长
        # action_log_probs就是每个步长执行的动作的概率，维度为(2000, 1)
        # dist_entroy 就是一个数字

        return action_log_probs, dist_entropy
