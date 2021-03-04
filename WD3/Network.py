import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, input_out, layer1_dims, layer2_dims, layer3_dims, action_space):
        super(ActorNetwork, self).__init__()
        norm_const = 1
        self.first = nn.Linear(input_dims, input_out)
        self.first.weight.data /= norm_const
        self.first.bias.data /= norm_const
        self.second = nn.Linear(input_out, layer1_dims)
        self.second.weight.data /= norm_const
        self.second.bias.data /= norm_const
        self.third = nn.Linear(layer1_dims, layer2_dims)
        self.third.weight.data /= norm_const
        self.third.bias.data /= norm_const

        self.last = nn.Linear(layer2_dims, action_space)
        torch.nn.init.uniform_(self.last.weight, -0.003, 0.003)
        self.last.bias.data /= norm_const

    def forward(self, inp):
        out = F.relu(self.first(inp.float()))
        out = F.relu(self.second(out))
        out = F.relu(self.third(out))
#        out = F.relu(self.fourth(out))
        out = torch.tanh(self.last(out))
        return out


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, input_out, layer1_dims, layer2_dims):
        """input1_dims pos
            input2_dims steer"""
        super(CriticNetwork, self).__init__()
        norm_const = 1e3
        self.first = nn.Linear(input_dims + action_dims + action_dims + action_dims, input_out)
        self.second = nn.Linear(input_out + action_dims + action_dims + action_dims, layer1_dims)
        self.third = nn.Linear(layer1_dims + action_dims + action_dims + action_dims, layer2_dims)
        self.last = nn.Linear(layer2_dims + action_dims + action_dims, 1)

    def forward(self, state, action):
        out = F.relu(self.first(torch.cat((state, action, action**2, torch.abs(action)), dim=1)))
        out = F.relu(self.second(torch.cat((out, action, action**2, torch.abs(action)), dim=1)))
        out = F.relu(self.third(torch.cat((out, action, action**2, torch.abs(action)), dim=1)))
        out = self.last(torch.cat((out, torch.abs(action), action**2), dim=1))
        # out = self.last(out)
        return out