class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.state_values = []
        self.is_terminals = []
    
    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.state_values[:]
        del self.is_terminals[:]

    def add(self, states, actions, logprobs, state_vals, rewards, is_terminals):
        self.states.append(states.copy())
        self.actions.append(actions.copy())
        self.logprobs.append(logprobs.copy())
        self.state_values.append(state_vals.copy())
        self.rewards.append(rewards.copy())
        self.is_terminals.append(is_terminals.copy())
