from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

"""
Original author https://github.com/ikostrikov/pytorch-a3c
The forward function has been edited and an act_A3C function has been added.
This function takes a greedy action
"""

def normalized_columns_initializer(weights, std=1.0):
    out = torch.randn(weights.size())
    out *= std / torch.sqrt(out.pow(2).sum(1, keepdim=True))
    return out


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = np.prod(weight_shape[1:4])
        fan_out = np.prod(weight_shape[2:4]) * weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        weight_shape = list(m.weight.data.size())
        fan_in = weight_shape[1]
        fan_out = weight_shape[0]
        w_bound = np.sqrt(6. / (fan_in + fan_out))
        m.weight.data.uniform_(-w_bound, w_bound)
        m.bias.data.fill_(0)


class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()

        self.action_space = action_space.n

        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(num_inputs, 32, 3, stride=2, padding=1)),
            ('elu1', nn.ELU()),
            ('conv2', nn.Conv2d(32, 32, 3, stride=1, padding=1)), # note the stride has been edited
            ('elu2', nn.ELU()),
            ('conv3', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            ('elu3', nn.ELU()),
            ('conv4', nn.Conv2d(32, 32, 3, stride=2, padding=1)),
            ('elu4', nn.ELU())
        ]))


        self.lstm = nn.LSTMCell(3200, 256)
        self.critic_linear = nn.Linear(256, 1)
        self.actor_linear = nn.Linear(256, self.action_space)

        # Init weights
        self.apply(weights_init)
        self.actor_linear.weight.data = normalized_columns_initializer(
            self.actor_linear.weight.data, 0.01)
        self.actor_linear.bias.data.fill_(0)
        self.critic_linear.weight.data = normalized_columns_initializer(
            self.critic_linear.weight.data, 1.0)
        self.critic_linear.bias.data.fill_(0)

        self.lstm.bias_ih.data.fill_(0)
        self.lstm.bias_hh.data.fill_(0)
        self.train()


    def forward(self, x, hx, cx):
        x = self.features(x)
        x = x.view(x.size(0), -1)

        # this enables batch forwards
        hx = hx.expand(x.shape[0], -1)
        cx = cx.expand(x.shape[0], -1)

        hx, cx = self.lstm(x, (hx, cx))
        x = hx

        return self.critic_linear(x), self.actor_linear(x), (hx, cx)


    def act_A3C(self, state, hx, cx):
        """Take a greedy action, only to be used during play not during training."""
        state = Variable(state, volatile=False).unsqueeze(0)
        value, logit, (hx, cx) = self.forward(state, hx, cx)
        action = logit.max(1)[1].data[0]
        return action, hx, cx
