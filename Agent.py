from Network import CriticNetwork, ActorNetwork
from Memory import Memory
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

class Agent:
    def __init__(self, gamma, tau, actorlr, criticlr, variance, action_dim, mem_size, batch_size, state_dim, reward_dim , eps):
        self.memory = Memory(mem_size= mem_size, batch_size= batch_size, state_dim=state_dim,
                              action_dim=action_dim, reward_dim=reward_dim)
        self.epsilon = eps
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.reward_weights = torch.tensor([0.1, -100, 200]).float()
        # self.reward_weights = torch.zeros(reward_dim)
        self.curr_reward = 0

        self.actor = ActorNetwork(input_dims = state_dim, input_out= 512, layer1_dims = 256 , action_space = action_dim)
        self.actor_optim = optim.Adam(params=self.actor.parameters(), lr=actorlr)
        self.actor_target = self.actor

        self.critic = CriticNetwork(input_dims = state_dim, action_dims= action_dim, input_out= 128, layer1_dims=64, layer2_dims=32
                                    )
        self.critic_optim = optim.Adam(params=self.critic.parameters(), lr=criticlr)
        self.critic_target = self.critic 
        self.critic_criterion = nn.MSELoss()

        self.variance = variance
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)

    def add_noise(self):
        noise = np.random.normal(0,self.variance, self.action_dim)
        noise = torch.tensor(noise).unsqueeze(0)
        return noise

    def action_selection(self, input):
        action = self.actor_target.forward(input)
        acct = action.detach()
        # print("Actor Value: " , acct.item())
        act = (acct + self.add_noise()).squeeze(0)
        return act.numpy().item() 

            

        # act = (acct).squeeze(0)
        


    def learn(self, state_, action_, reward_,new_state_ , done_):
        
        self.memory.store(state_, new_state_ , action_, reward_, done_)
        if self.memory.mem_index <= self.batch_size:
            return        
        curr_Q_value= self.critic.forward(torch.tensor(state_, dtype = torch.float32),torch.tensor(action_, dtype= torch.float32).unsqueeze(0))
        # print("Current Q Value : {}".format(curr_Q_value.item()))
        # print(torch.tensor(action_).unsqueeze(0))
        state_batch, new_state_batch, action_batch, reward_batch, terminal_batch = self.memory.sample()
        Qvals = self.critic.forward(state_batch, action_batch)
        next_actions = self.actor_target.forward(new_state_batch)
        next_Q = self.critic_target.forward(new_state_batch, next_actions)
        Qprime = reward_batch.sum(axis = 1).unsqueeze(1) + (self.gamma * next_Q * (1 - terminal_batch.unsqueeze(1))).detach().numpy()


        critic_loss = self.critic_criterion(Qprime, Qvals)


        # update networks
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()
        
        # Actor loss
        actor_forward = self.actor.forward(state_batch)
        policy_loss = -self.critic.forward(state_batch,actor_forward).mean()


        self.actor_optim.zero_grad()
        policy_loss.backward()
        self.actor_optim.step()

        # update target networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

        for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data * self.tau + target_param.data * (1.0 - self.tau))

    def save_models(self, episode, id):
        
        torch.save(self.actor.state_dict(), f"model_params_ddpg/{id}_{episode}_actor.pth")
        torch.save(self.actor_target.state_dict(), f"model_params_ddpg/{id}_{episode}_actortarget.pth")
        torch.save(self.critic.state_dict(), f"model_params_ddpg/{id}_{episode}_critic.pth")
        torch.save(self.critic_target.state_dict(), f"model_params_ddpg/{id}_{episode}_critictarget.pth")

    def load_models(self, episode, id):

        self.actor.load_state_dict(torch.load( f"model_params_ddpg/{id}_{episode}_actor.pth",  map_location=self.device))
        self.actor_target.load_state_dict(torch.load( f"model_params_ddpg/{id}_{episode}_actortarget.pth",  map_location=self.device))
        self.critic.load_state_dict(torch.load( f"model_params_ddpg/{id}_{episode}_critic.pth",  map_location=self.device))
        self.critic_target.load_state_dict(torch.load( f"model_params_ddpg/{id}_{episode}_critictarget.pth",  map_location=self.device))


    # def reward_function(self, reward_batch):
    #     # reward_batch = [step, collision, goal]
    #     #new_reward_batch = np.dot(reward_batch, self.reward_weights)
    #     # print("reward is: " , new_reward_batch)
    #     return reward_batch.sum()

