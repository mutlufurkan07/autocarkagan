from Network import ActorNetwork, CriticNetwork
from Memory import Memory
import numpy as np
import torch.optim as optim
import copy
import torch
import torch.nn as nn
import os


class Agent:
    def __init__(self, gamma, tau, actorlr, criticlr, std, action_dim, mem_size, batch_size, state_dim, reward_dim,
                 max_action, training_or_validation, beta, target_noise_mag, experiment_id):

        self.path = "model_params_wd3/" + str(experiment_id)
        if not os.path.exists(self.path):
            os.mkdir(self.path)
        self.memory = Memory(mem_size=mem_size, batch_size=batch_size, state_dim=state_dim,
                             action_dim=action_dim, reward_dim=reward_dim)
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.curr_reward = 0
        self.beta = beta
        self.max_action = max_action

        self.actor = ActorNetwork(input_dims=state_dim, input_out=200, layer1_dims=128, layer2_dims=64, layer3_dims=64,
                                  action_space=action_dim)
        self.actor_target = ActorNetwork(input_dims=state_dim, input_out=200, layer1_dims=128, layer2_dims=64,
                                         layer3_dims=64, action_space=action_dim)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic1 = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=200, layer1_dims=64,
                                     layer2_dims=64)
        self.critic1_target = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=200,
                                            layer1_dims=64, layer2_dims=64)
        self.critic1.load_state_dict(self.critic1_target.state_dict())

        self.critic2 = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=200, layer1_dims=64,
                                     layer2_dims=64)
        self.critic2_target = CriticNetwork(input_dims=state_dim, action_dims=action_dim, input_out=200,
                                            layer1_dims=64,
                                            layer2_dims=64)
        self.critic2.load_state_dict(self.critic2_target.state_dict())

        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actorlr)
        self.critic1_optim = optim.Adam(params=self.critic1.parameters(), lr=criticlr)
        self.critic2_optim = optim.Adam(params=self.critic2.parameters(), lr=criticlr)

        self.critic_criterion = nn.MSELoss()
        self.target_noise_mag = target_noise_mag
        self.std = std
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.actor.to(self.device)
        self.actor_target.to(self.device)
        self.critic1.to(self.device)
        self.critic2.to(self.device)
        self.critic1_target.to(self.device)
        self.critic2_target.to(self.device)
        self.isTraining = training_or_validation
        self.actor_update_index = 0
        self.actor_update_frequency = 2

    def add_noise(self):
        noise = np.random.normal(0, self.std, self.action_dim)
        noise = torch.tensor(noise).unsqueeze(0)
        return noise

    def action_selection(self, picture, input):
        action = self.actor_target.forward(picture, input)

        acct = action.detach()
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

        action_batch = action_batch.clamp(-self.max_action, self.max_action)
        Q1_pred = self.critic1.forward(state_batch, action_batch)
        Q2_pred = self.critic2.forward(state_batch, action_batch)

        next_actions = (self.actor_target.forward(new_state_batch) + torch.normal(mean=torch.tensor(0.0), std=torch.tensor(0.1)).clamp(-self.target_noise_mag, self.target_noise_mag)).clamp(-self.max_action, self.max_action)
        Q1 = self.critic1_target(new_state_batch, next_actions)
        Q2 = self.critic2_target(new_state_batch, next_actions)
        y_td3 = torch.min(Q1, Q2)
        y_avg = (Q1 + Q2) / 2
        y = reward_batch.sum(axis=1).unsqueeze(1) + self.gamma * (self.beta * y_td3 + (1 - self.beta) * y_avg) * (1 - terminal_batch.unsqueeze(1))

        critic1_loss = self.critic_criterion(Q1_pred, y)
        self.critic1_optim.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_optim.step()

        critic2_loss = self.critic_criterion(Q2_pred, y)
        self.critic2_optim.zero_grad()
        critic2_loss.backward(retain_graph=True)
        self.critic2_optim.step()

        if self.actor_update_index % self.actor_update_frequency == 0:
            # Actor loss
            actor_forward = self.actor.forward(state_batch)
            policy_loss = -self.critic1.forward(state_batch, actor_forward).mean()

            self.actor_optim.zero_grad()
            policy_loss.backward()
            self.actor_optim.step()

            # soft update
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

            for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
                target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        self.actor_update_index += 1

    def save_models(self, episode, id):

        torch.save(self.actor.state_dict(), f"{self.path}/{id}_{episode}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"{self.path}/{id}_{episode}_actortarget.pth")
        torch.save(self.critic1.state_dict(), f"{self.path}/{id}_{episode}_critic.pth")
        torch.save(self.critic1_target.state_dict(), f"{self.path}/{id}_{episode}_critictarget.pth")

    def load_models(self, episode, id):

        self.actor.load_state_dict(torch.load(f"{self.path}/{id}_{episode}_actor.pth", map_location=self.device))
        self.actor_target.load_state_dict(
            torch.load(f"{self.path}/{id}_{episode}_actortarget.pth", map_location=self.device))
        self.critic1.load_state_dict(
            torch.load(f"{self.path}/{id}_{episode}_critic.pth", map_location=self.device))
        self.critic1_target.load_state_dict(
            torch.load(f"{self.path}/{id}_{episode}_critictarget.pth", map_location=self.device))
