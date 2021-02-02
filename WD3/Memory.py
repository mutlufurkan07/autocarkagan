import numpy as np
import torch

class Memory():
    def __init__(self, mem_size, batch_size, state_dim, action_dim, reward_dim):
        self.mem_size = mem_size
        self.batch_size = batch_size
        self.state_memory = torch.zeros((mem_size, state_dim), dtype= torch.float32)
        self.new_state_memory = torch.zeros((mem_size, state_dim) , dtype= torch.float32)
        self.action_memory = torch.zeros((mem_size, action_dim), dtype= torch.float32)
        self.reward_memory = torch.zeros((mem_size, reward_dim), dtype= torch.float32)
        self.terminal_memory = torch.zeros(mem_size)
        self.mem_full = False
        self.mem_index = 0

    def store(self, state, new_state, action, reward, terminal):
        if self.mem_index >= self.mem_size - 1:
            index = np.random.randint(self.mem_size)
            self.mem_full = True
        else:
            index = self.mem_index
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = int(terminal)
        self.mem_index += 1


    def sample(self):
        if self.mem_full:
            batch_index = np.random.choice(self.mem_size -1, self.batch_size, replace= False)
        else:
            batch_index = np.random.choice(self.mem_index, self.batch_size, replace= False)
        state_batch = self.state_memory[batch_index]
        new_state_batch = self.new_state_memory[batch_index]
        action_batch = self.action_memory[batch_index]
        reward_batch = self.reward_memory[batch_index]
        terminal_batch = self.terminal_memory[batch_index]

        return state_batch, new_state_batch, action_batch, reward_batch, terminal_batch


