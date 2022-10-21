import torch.nn as nn
import torch
import numpy as np
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

# x = torch.zeros((1, 3))
# x[0][1] = 1
# m = Categorical(x)
# print(m.sample())

num_mini_batch = 10
batch_size = 6400
mini_batch_size = batch_size // num_mini_batch


rand = torch.randperm(batch_size)

y = rand[1 * mini_batch_size:(2) * mini_batch_size]
y = [rand[i * mini_batch_size:(i + 1) * mini_batch_size] for i in range(num_mini_batch)]

rows = np.indices((6400, 5))[0]
cols = np.stack([np.arange(5) for _ in range(6400)])
share_obs = np.zeros((201, 32, 5, 155))
share_obs = share_obs[:-1].reshape(-1, *share_obs.shape[2:])
share_obs = share_obs[rows, cols]

obs = np.zeros((32, 3))
def add(share_obs, obs, obs2, obs3):
    for i in range(5):
        yield share_obs, obs, obs2, obs3
# data = add(share_obs, obs)

a = [1, 2]
b = [3, 3]
data1 = add(a, b, a, a)
for i in data1:
    print(1111, i)

