from Network import CriticNetwork, ActorNetwork
from Memory import Memory
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn


class Agent:
    def __init__(self, gamma, tau, actorlr, criticlr, variance, action_dim, mem_size, batch_size, state_dim, reward_dim,
                 eps, training_or_validatioın):
        self.memory = Memory(mem_size=mem_size, batch_size=batch_size, state_dim=state_dim,
                             action_dim=action_dim, reward_dim=reward_dim)
        self.epsilon = eps
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_weights = torch.tensor([0.1, -100, 200]).float()
        # self.reward_weights = torch.zeros(reward_dim)
        self.curr_reward = 0

        self.actor = ActorNetwork(input_dims=state_dim, input_out=256, layer1_dims=128, layer2_dims=128, layer3_dims=128,
                                  action_space=action_dim)
        self.actor_target = ActorNetwork(input_dims=state_dim, input_out=256, layer1_dims=128, layer2_dims=128,
                                         layer3_dims=128, action_space=action_dim)

        self.actor.load_state_dict(self.actor_target.state_dict())
        self.actor_optimizer = optim.Adam(params=self.actor.parameters(), lr=actorlr)

        self.critic = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=128, layer1_dims=128,
                                    layer2_dims=128)
        self.critic_target = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=128, layer1_dims=128,
                                           layer2_dims=128)

        self.critic.load_state_dict(self.critic_target.state_dict())
        self.critic_optimizer = optim.Adam(params=self.critic.parameters(), lr=criticlr)

        self.critic_criterion = nn.MSELoss()
        print(f"Actor:\n {self.actor}")
        print(f"Critic:\n {self.critic}")
        self.variance = variance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic.to(self.device)
        self.critic_target.to(self.device)
        self.isTraining = training_or_validatioın

    def add_noise(self):
        noise = np.random.normal(0, self.variance, self.action_dim)
        noise = torch.tensor(noise).unsqueeze(0)
        return noise

    def action_selection(self, inp):
        action = self.actor_target.forward(inp)
        acct = action.detach()
        # print("Actor Value: " , acct.item())
        acct = acct.cpu()
        if self.isTraining:
            act = (acct + self.add_noise()).squeeze(0)
        else:
            act = acct.squeeze(0)
        return act.numpy().item()

    def learn(self, state_, action_, reward_, new_state_, done_):

        self.memory.store(state_, new_state_, action_, reward_, done_)
        if self.memory.mem_index <= self.batch_size:
            return

        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch = self.memory.sample()
        state_batch = state_batch.to(self.device)
        new_state_batch = new_state_batch.to(self.device)
        action_batch = action_batch.to(self.device)
        reward_batch = reward_batch.to(self.device)
        terminal_batch = terminal_batch.to(self.device)
        q_vals = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(new_state_batch)
        next_Q = self.critic_target.forward(new_state_batch, next_actions)
        qprime = reward_batch.sum(axis=1).unsqueeze(1) + (
                self.gamma * next_Q * (1 - terminal_batch.unsqueeze(1)))

        critic_loss = self.critic_criterion(qprime, q_vals)

        # update networks
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor loss
        actor_forward = self.actor.forward(state_batch)
        policy_loss = -self.critic.forward(state_batch, actor_forward).mean()

        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, episode, id_num):

        torch.save(self.actor.state_dict(), f"model_params_ddpg/{id_num}_{episode}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"model_params_ddpg/{id_num}_{episode}_actortarget.pth")
        torch.save(self.critic.state_dict(), f"model_params_ddpg/{id_num}_{episode}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"model_params_ddpg/{id_num}_{episode}_critictarget.pth")

    def load_models(self, episode, id_num):

        self.actor.load_state_dict(
            torch.load(f"model_params_ddpg/{id_num}_{episode}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"model_params_ddpg/{id_num}_{episode}_actortarget.pth", map_location=self.device))
        self.critic.load_state_dict(
            torch.load(f"model_params_ddpg/{id_num}_{episode}_critic.pth", map_location=self.device))
        self.critic_target.load_state_dict(
            torch.load(f"model_params_ddpg/{id_num}_{episode}_critictarget.pth", map_location=self.device))
