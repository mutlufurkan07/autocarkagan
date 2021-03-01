import torch
import torch.nn as nn
from Network import ActorCritic
from Memory import Memory
import os


class Agent:
    def __init__(self, state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim, action_std, lr, betas, gamma, K_epochs,
                 eps_clip, update_horizon, device):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.device = device
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.memory = Memory(horizon=update_horizon, state_dim=state_dim, action_dim=action_dim)

        self.policy = ActorCritic(state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim, action_std).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas)

        self.policy_old = ActorCritic(state_dim, hidden_dim1, hidden_dim2, hidden_dim3, action_dim, action_std).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.critic_loss = nn.MSELoss()

    def action_selection(self, state):
        # state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action, log_prob = self.policy_old.act(torch.as_tensor(state, dtype=torch.float32).to(self.device))
        return action.squeeze(0).cpu().numpy(), log_prob.cpu()

    def update(self):

        old_states, old_actions, rews, old_logprobs, terminals = self.memory.sample()

        # Monte Carlo estimate of rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(rews), reversed(terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states.float().to(self.device),
                                                                        old_actions.to(self.device))

            # logprobs, state_values, dist_entropy = self.evaluate(old_states.float(), old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach().to(self.device))

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surrogate1 = ratios * advantages
            surrogate2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surrogate1, surrogate2) + 0.5 * self.critic_loss(state_values,
                                                                               rewards) - 0.01 * dist_entropy
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def save_models(self, episode, ID):
        if not os.path.exists("model_params_ppo"):
            os.mkdir("model_params_ppo")
        torch.save(self.policy.actor.state_dict(), f"model_params_ppo/{ID}_{episode}_actor.pth")

    def load_models(self, episode, ID):

        self.policy.load_state_dict(torch.load(f"model_params_ppo/{ID}_{episode}_actor.pth", map_location=self.device))

