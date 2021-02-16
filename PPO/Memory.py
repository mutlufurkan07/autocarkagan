class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def store(self, s, a, r, p, d):
        self.actions.append(a)
        self.states.append(s)
        self.logprobs.append(p)
        self.rewards.append(r)
        self.is_terminals.append(d)