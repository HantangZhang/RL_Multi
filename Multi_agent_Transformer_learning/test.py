import torch.nn as nn
import torch

layer = nn.Linear(256, 256)

x = torch.empty((2,3,4,5 ))
B, L, D = x.shape
print(B, L, D)
print(x.shape)