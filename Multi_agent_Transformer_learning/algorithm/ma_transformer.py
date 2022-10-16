import torch
import torch.nn as nn
from algorithm.utils import init


def init_(m, gain=0.01, activate=False):
    if activate:
        gain = nn.init.calculate_gain('relu')
    return init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), gain=gain)



class SelfAttention(nn.Module):

    def __init__(self, n_embd, n_head):
        super(SelfAttention, self).__init__()

        assert n_embd % n_head == 0

        self.key =

    def forward(self, key, value, query):
        # 可以用这样的方法提取中tensor的三个维度的坐标
        # B 就是batchsize，D是feature的大小, L的应该是数据长度的大小，暂时不清楚
        B, L, D = query.size()
        # 假如说数据的维度是512

