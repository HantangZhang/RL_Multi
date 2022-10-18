import torch.nn as nn
import torch
from torch.nn import functional as F
from torch.distributions import Categorical


# x = torch.zeros((3, 5, 1))
# x[0][0][0] = 3
# x[0][3][0] = 2
# x[1][2][0] = 4
# x[2][1][0] = 1
#
# x = x.long()
# one_hot = F.one_hot(x.squeeze(-1), num_classes=5)
# shift = torch.zeros((3, 5, 6))
# shift[:, 0, 0] = 1
# shift[:, 0:, 1:] = one_hot[:, :, :]
# print(one_hot)
# print(shift)
# print(one_hot[:, :-1, :].shape)

x = torch.zeros((1, 3))
x[0][1] = 1
m = Categorical(x)
print(m.sample())