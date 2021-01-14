import torch
import torch.nn as nn
import torch.nn.functional as F


class ActorNetwork(nn.Module):
    def __init__(self, input_dims, input_out, layer1_dims, action_space):
        super(ActorNetwork, self).__init__()
        self.first = nn.Linear(input_dims, input_out)
        
        # self.first.weight.data.normal_(0,0.01)
        self.second = nn.Linear(input_out, layer1_dims)
        
        # self.second.weight.data.normal_(0,0.01)
        self.third = nn.Linear(layer1_dims ,128 )
        
        self.last = nn.Linear(128, action_space)
        # self.last.weight.data.normal_(0,0.01)

    def forward(self, input):
        out = F.relu(self.first(input.float()))
        out = self.second(out.float())
        out = F.relu(out)        
        out = F.relu(self.third(out))                
        out = self.last(out)
        out /= 10
        out = torch.tanh(out)
        return out


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, action_dims, input_out, layer1_dims, layer2_dims):
        """input1_dims pos
            input2_dims steer"""
        super(CriticNetwork, self).__init__()
        self.first = nn.Linear(input_dims + action_dims, input_out)
        # self.first.weight.data.normal_(0,0.01)
        self.second = nn.Linear(input_out, layer1_dims)
        # self.second.weight.data.normal_(0,0.01)
        self.third = nn.Linear(layer1_dims, layer2_dims)
        # self.third.weight.data.normal_(0,0.01)
        #self.last = nn.Linear(layer2_dims, 1)
        self.last = nn.Linear(layer2_dims + action_dims, 1)
        # self.last.weight.data.normal_(0,0.01)

    def forward(self, state, action):
        out = F.relu(self.first(torch.cat((state, action), dim=-1)))
        out = F.tanh(self.second(out.float()))
        out = F.relu(self.third(out.float()).float())
        out = self.last(torch.cat((out, torch.abs(action)), dim=-1))
        #out = self.last(out)
        return out