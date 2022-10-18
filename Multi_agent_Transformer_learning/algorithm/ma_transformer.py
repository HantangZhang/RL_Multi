import torch
import torch.nn as nn
import numpy as np
from torch.nn import functional as F
import math
from algorithm.utils import init, check
from transformer_act import discrete_parallel_act, continuous_parallel_act, continuous_autoregreesive_act, discrete_autoregreesive_act


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)



class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head, n_agent, masked=False):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0

        self.masked = masked
        self.n_head = n_head

        self.key = init_(nn.Linear(n_embd, n_embd))
        self.query = init_(nn.Linear(n_embd, n_embd))
        self.value = init_(nn.Linear(n_embd, n_embd))

        # output projection
        self.proj = init_(nn.Linear(n_embd, n_embd))



    def forward(self, key, value, query):
        # 可以用这样的方法提取中tensor的三个维度的坐标
        # B 就是batchsize，D是feature的大小,这里把21维的obs利用embed处理成64维, L是智能体的数量
        B, L, D = query.size()

        # 对于5个智能体，n_embd如果为64，batch是32
        # 输入的数据key维度是(32, 5, 64)，经过key层处理后维度不变，然后为了提取多个特征，利用了head
        # 通过view把数据分成n_head的数据，每个head的特征就是n_embd / n_head
        # 最后通过transpose把第一维和第二维调换一下
        k = self.key(key).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        q = self.query(query).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)
        v = self.value(value).view(B, L, self.n_head, D // self.n_head).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))

        if self.masked:
            att = att.masked_fill(self.mask[:, :, :L, :L] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v  # (B, nh, L, L) x (B, nh, L, hs) -> (B, nh, L, hs)
        # 返回一个制定的连续的内存张量，应该和算法训练没有关系
        y = y.transpose(1, 2).contiguous().view(B, L, D)  # re-assemble all head outputs side by side
        y = self.proj(y)
        return y

class EncodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(EncodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

        self.attn = SelfAttention(n_embd, n_head, n_agent, masked=False)
        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )

    def forward(self, x):
        x = self.ln1(x + self.attn(x, x, x))
        x = self.ln2(x + self.mlp(x))
        return x

class Encoder(nn.Module):

    def __init__(self, obs_dim, n_block, n_embd, n_head, n_agent):
        super(Encoder, self).__init__()

        self.obs_dim = obs_dim
        self.n_embd = n_embd
        self.n_agent = n_agent

        self.obs_encoder =  nn.Sequential(
                                    nn.LayerNorm(obs_dim),
                                    init_(nn.Linear(obs_dim, n_embd), activate=True),
                                    nn.GELU()
                                    )

        self.ln = nn.LayerNorm(n_embd)
        self.blocks = nn.Sequential(*[EncodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
        self.head = nn.Sequential(
            init_(nn.Linear(n_embd, n_embd), activate=True),
            nn.GELU(),
            nn.LayerNorm(n_embd),
            init_(nn.Linear(n_embd, 1))
        )

    def forward(self, obs):

        obs_embeddings = self.obs_encoder(obs)
        x = obs_embeddings

        # 编码后的obs，输入到decoder当中
        rep = self.blocks(self.ln(x))

        # encoder输出的价值，目前还不确定作用
        v_loc = self.head(rep)

        return v_loc, rep

class DecodeBlock(nn.Module):
    """ an unassuming Transformer block """

    def __init__(self, n_embd, n_head, n_agent):
        super(DecodeBlock, self).__init__()

        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ln3 = nn.LayerNorm(n_embd)

        self.attn1 = SelfAttention(n_embd, n_head, n_agent, masked=True)
        self.attn2 = SelfAttention(n_embd, n_head, n_agent, masked=True)

        self.mlp = nn.Sequential(
            init_(nn.Linear(n_embd, 1 * n_embd), activate=True),
            nn.GELU(),
            init_(nn.Linear(1 * n_embd, n_embd))
        )


    def forward(self, x, rep_enc):
        x = self.ln1(x + self.attn1(x, x, x))
        x = self.ln2(rep_enc + self.attn2(key=x, value=x, query=rep_enc))
        x = self.ln3(x + self.mlp(x))
        return x



class Decoder(nn.Module):

    def __init__(self, obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                 action_type='Discrete', dec_actor=False, share_actor=False):
        super(Decoder, self).__init__()

        self.action_dim = action_dim
        self.n_embd = n_embd
        self.action_type = action_type

        self.dec_actor = dec_actor
        self.share_actor = share_actor

        if self.dec_actor:
            if self.share_actor:
                self.mlp = nn.Sequential(nn.LayerNorm(obs_dim),
                                         init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                         nn.LayerNorm(n_embd),
                                         init_(nn.Linear(n_embd, action_dim)))
            else:
                self.mlp = nn.ModuleList()
                for n in range(n_agent):
                    actor = nn.Sequential(nn.LayerNorm(obs_dim),
                                          init_(nn.Linear(obs_dim, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(),
                                          nn.LayerNorm(n_embd),
                                          init_(nn.Linear(n_embd, action_dim)))
                    self.mlp.append(actor)
        else:
            # 对输入的动作进行编码，编码成n_emcd的维度
            if action_type == 'Discrete':
                self.action_encoder = nn.Sequential(
                    init_(nn.Linear(action_dim + 1, n_embd, bias=False), activate=True),
                    nn.GELU()
                )
            else:
                self.action_encoder = nn.Sequential(init_(nn.Linear(action_dim, n_embd), activate=True), nn.GELU())

            self.obs
            self.ln = nn.LayerNorm(n_embd)
            self.blocks = nn.Sequential(*[DecodeBlock(n_embd, n_head, n_agent) for _ in range(n_block)])
            self.head = nn.Sequential(init_(nn.Linear(n_embd, n_embd), activate=True), nn.GELU(), nn.LayerNorm(n_embd),
                                      init_(nn.Linear(n_embd, action_dim)))




    def forward(self, action, obs_rep, obs):
        if self.dec_actor:
            if self.share_actor:
                logit = self.mlp(obs)
            else:
                logit = []
                for n in range(len(self.mlp)):
                    logit_n = self.mlp[n](obs[:, n, :])
                    logit.append(logit_n)
                logit = torch.stack(logit, dim=1)
        else:
            action_embeddings = self.action_encoder(action)
            x = self.ln(action_embeddings)
            for block in self.blocks:
                x = block(x, obs_rep)
            logit = self.head(x)

        return logit


class MultiAgentTransformer(nn.Module):

    def __init__(self, obs_dim, action_dim, n_agent, n_block, n_embd, n_head, action_type='Discrete', device=torch.device("cpu"),
                 dec_actor=False, share_actor=False):
        super(MultiAgentTransformer, self).__init__()

        self.n_agent = n_agent
        self.action_dim = action_dim
        self.action_type = action_type
        self.device = device
        self.tpdv = dict(dtype=torch.float32, device=device)

        self.encoder = Encoder(obs_dim, n_block, n_embd, n_head, n_agent)
        self.decoder = Decoder(obs_dim, action_dim, n_block, n_embd, n_head, n_agent,
                               self.action_type, dec_actor=dec_actor, share_actor=share_actor)
        self.to(device)

    def forward(self, obs, action, available_actions=None):
        # forward用于训练，inference用get_action

        # **是对dict进行接包，让dict中每个都引用到obs中
        obs = check(obs).to(**self.tpdv)
        action = check(action).to(**self.tpdv)

        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)

        if self.action_type == 'Discrete':
            action = action.long()
            action_log, entropy = discrete_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                        self.n_agent, self.action_dim, self.tpdv, available_actions)
        else:
            action_log, entropy = continuous_parallel_act(self.decoder, obs_rep, obs, action, batch_size,
                                                          self.n_agent, self.action_dim, self.tpdv)

        return action_log, v_loc, entropy

    def get_actions(self, obs, available_actions=None, deterministic=False):
        obs = check(obs).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        batch_size = np.shape(obs)[0]
        v_loc, obs_rep = self.encoder(obs)
        if self.action_type == "Discrete":
            output_action, output_action_log = discrete_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                           self.n_agent, self.action_dim, self.tpdv,
                                                                           available_actions, deterministic)
        else:
            output_action, output_action_log = continuous_autoregreesive_act(self.decoder, obs_rep, obs, batch_size,
                                                                             self.n_agent, self.action_dim, self.tpdv,
                                                                             deterministic)

        return output_action, output_action_log, v_loc

    def get_values(self, obs):
        obs = check(obs).to(**self.tpdv)
        v_tot, obs_rep = self.encoder(obs)
        return v_tot