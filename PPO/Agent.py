import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from Network import ActorCritic

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    def __init__(self, state_dim, hidden_dim,  action_dim, action_std, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.actor = ActorCritic(state_dim, hidden_dim, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, hidden_dim, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.actor.state_dict())

        self.MseLoss = nn.MSELoss()

    def action_selection(self, state, memory):
        action, log_prob = self.policy_old.act(state, memory)
        return action.squeeze(0).cpu().numpy(), log_prob.cpu()

    def update(self, memory):
        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards,dtype=float).float().to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # convert list to tensor
        old_states = torch.squeeze(torch.tensor(memory.states).to(device), 1).detach()
        old_actions = torch.squeeze(torch.stack(memory.actions).to(device), 1).detach()
        old_logprobs = torch.squeeze(torch.stack(memory.logprobs), 1).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.actor.evaluate(old_states.float(), old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.actor.state_dict())

    def save_models(self, episode, id):

        torch.save(self.actor.actor.state_dict(), f"model_params_td3/{id}_{episode}_actor.pth")
        torch.save(self.actor.critic.state_dict(), f"model_params_td3/{id}_{episode}_critic.pth")

    def load_models(self, episode, id):

        self.actor.actor.load_state_dict(torch.load(f"model_params_td3/{id}_{episode}_actor.pth", map_location=self.device))
        self.actor.critic.load_state_dict(torch.load(f"model_params_td3/{id}_{episode}_critic.pth", map_location=self.device))