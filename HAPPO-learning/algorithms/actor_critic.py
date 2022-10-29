import torch
import torch.nn as nn
from utils.util import get_shape_from_obs_space, check, init
from algorithms.model.mlp import MLPBase
from algorithms.model.act import ACTLayer



class Actor(nn.Module):

    def __init__(self, args, obs_space, action_space, device=torch.device("cpu")):
        super(Actor, self).__init__()

        self.hidden_size = args.hidden_size
        self.args = args
        self.tpdv = dict(dtype=torch.float32, device=device)

        # 5v5这里检查是否还有用
        obs_shape = get_shape_from_obs_space(obs_space)


        # 在这里构建神经网络，alpha可能会放在这里
        base = MLPBase
        self.base = base(args, obs_shape)

        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, args)

        self.to(device)

    def forward(self, obs, masks, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)

        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs

    def evaluate_actions(self, obs, action, masks, available_actions=None, active_masks=None):
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                   action, available_actions,
                                                                   active_masks=
                                                                   active_masks if self._use_policy_active_masks
                                                                   else None)
        return action_log_probs, dist_entropy


class Critic(nn.Module):

    def __init__(self, args, cent_obs_space, device=torch.device("cpu")):
        super(Critic, self).__init__()
        self.hidden_size = args.hidden_size
        self._use_orthogonal = args.use_orthogonal
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        base = MLPBase
        # critic和acotr用一样的简单的mlp
        self.base = base(args, cent_obs_space)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0))

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, masks):
        cent_obs = check(cent_obs).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values