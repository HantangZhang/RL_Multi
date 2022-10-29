import torch
import torch.nn as nn
from utils.util import init

class FixedCategorical(torch.distributions.Categorical):
    # 让下面调用这个类的变量可以直接调用这三个方法，看下面写的例子就懂了
    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return (
            super()
            .log_prob(actions.squeeze(-1))
            .view(actions.size(0), -1)
            .sum(-1)
            .unsqueeze(-1)
        )

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)



class Categorical(nn.Module):
    def __init__(self, num_inputs, num_outputs, use_orthogonal=True, gain=0.01):
        super(Categorical, self).__init__()
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][use_orthogonal]

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain)

        self.linear = init_(nn.Linear(num_inputs, num_outputs))

    def forward(self, x, available_actions=None):
        x = self.linear(x)

        if available_actions is not None:
            x[available_actions == 0] = -1e10

        return FixedCategorical(logits=x)


if __name__ == '__main__':
    #
    x = torch.zeros((4, 5))

    model = Categorical(5, 3)

    y = model(x)
    actions = y.sample()
    print(y.log_probs(actions))

