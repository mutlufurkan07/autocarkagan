import torch


class Memory:
    # We do not need to convert the tensors to cuda because it is only used once (store-sample).
    def __init__(self, horizon, state_dim, action_dim):
        self.actions = torch.zeros((horizon, action_dim), dtype=torch.float32)
        self.states = torch.zeros((horizon, state_dim), dtype=torch.float32)
        self.rewards = torch.zeros(horizon, dtype=torch.float32)
        self.logprobs = torch.zeros(horizon, dtype=torch.float32)
        self.dones = torch.zeros(horizon)

        self.mem_index = 0
        self.horizon = horizon
        self.state_dim = state_dim
        self.action_dim = action_dim

    def clear_memory(self):
        self.actions = torch.zeros((self.horizon, self.action_dim), dtype=torch.float32)
        self.states = torch.zeros((self.horizon, self.state_dim), dtype=torch.float32)
        self.rewards = torch.zeros(self.horizon, dtype=torch.float32)
        self.logprobs = torch.zeros(self.horizon, dtype=torch.float32)
        self.dones = torch.zeros(self.horizon)

        self.mem_index = 0

    def store(self, s, a, r, logprob, d):
        assert self.mem_index < self.horizon + 2, "Memory Pointer Exceeded The Horizon"
        self.states[self.mem_index] = s.clone()
        self.actions[self.mem_index] = torch.tensor(a, dtype=torch.float32)
        self.rewards[self.mem_index] = r
        self.logprobs[self.mem_index] = logprob
        self.dones[self.mem_index] = torch.tensor(d, dtype=torch.float32)

        self.mem_index += 1

    def sample(self):
        return self.states, self.actions, self.rewards, self.logprobs, self.dones
