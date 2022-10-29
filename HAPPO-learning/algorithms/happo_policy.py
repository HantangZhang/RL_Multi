
import torch
from utils.util import update_linear_schedule
from algorithms.actor_critic import Actor, Critic

class HAPPO_Policy:

    def __init__(self, args, obs_space, cent_obs_space, act_space, device=torch.device("cpu")):
        self.args=args
        self.device = device
        self.lr = args.lr
        self.critic_lr = args.critic_lr

        # 这两个都是optim有关的参数
        self.opti_eps = args.opti_eps
        self.weight_decay = args.weight_decay


        self.obs_space = obs_space
        self.share_obs_space = cent_obs_space
        self.act_space = act_space

        # 注意虽然每个agent都有一个critic，但是他们训练用的同样的data，和更新方法，因为他们有着同样的参数，所以可以认为每个agent的critic都一样
        #
        self.actor = Actor(args, self.obs_space, self.act_space, self.device)
        self.critic = Critic(args, self.share_obs_space, self.device)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(),
                                                lr=self.lr, eps=self.opti_eps,
                                                weight_decay=self.weight_decay)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(),
                                                 lr=self.critic_lr,
                                                 eps=self.opti_eps,
                                                 weight_decay=self.weight_decay)


    def lr_decay(self, episode, episodes):
        """
        Decay the actor and critic learning rates.
        :param episode: (int) current training episode.
        :param episodes: (int) total number of training episodes.
        """
        update_linear_schedule(self.actor_optimizer, episode, episodes, self.lr)
        update_linear_schedule(self.critic_optimizer, episode, episodes, self.critic_lr)

    def get_actions(self, cent_obs, obs, masks, available_actions=None,
                    deterministic=False):

        actions, action_log_probs = self.actor(obs,
                                               masks,
                                               available_actions,
                                               deterministic)
        values = self.critic(cent_obs, masks)

        return values, actions, action_log_probs

    def get_values(self, cent_obs, masks):

        values, _ = self.critic(cent_obs, masks)
        return values

    def evaluate_actions(self, cent_obs, obs, action, masks,
                         available_actions=None, active_masks=None):
        action_log_probs, dist_entropy = self.actor.evaluate_actions(obs,
                                                                     action,
                                                                     masks,
                                                                     available_actions,
                                                                     active_masks)
        values, _ = self.critic(cent_obs, masks)
        return values, action_log_probs, dist_entropy

    def act(self, obs, masks, available_actions=None, deterministic=False):
        # act只获得动作
        actions, _ = self.actor(obs, masks, available_actions, deterministic)

        return actions
