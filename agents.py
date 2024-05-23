import numpy as np
from abc import ABC, abstractmethod
from environments import Environment


class Agent(ABC):
    def __init__(self, env: Environment) -> None:
        super().__init__()
        self.env: Environment = env
        self.reset()

    @abstractmethod
    def update(self, state, action, reward):
        pass

    def select_action(self, state):
        return np.argmax(self.q_value[state])

    def reset(self):
        self.q_value = np.zeros((self.env.state_size, self.env.action_size))
        self.policy = np.zeros(self.env.state_size)


class QlearningAgent(Agent):
    def __init__(self, env: Environment, learning_rate: float, discount_factor: float) -> None:
        super().__init__(env)
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor

    def update(self, state, action, reward, state_prime):
        """Update Q(s,a):= Q(s,a) + lr [R(s,a) + gamma * max Q(s',a') - Q(s,a)]"""
        delta = (
            reward
            + self.discount_factor * np.max(self.q_value[state_prime])
            - self.q_value[state, action]
        )        
        return self.q_value[state, action] + self.learning_rate * delta


class UcbAgent(Agent):
    def __init__(self, env: Environment, c=2):
        super().__init__(env)
        self.c = c  # exploration parameter
        self.counts = np.zeros(env.action_size)

    def select_action(self, state):
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            action = np.random.randint(self.env.action_size)
        else:
            ucb_values = self.q_value[state] + self.c * np.sqrt(np.log(total_counts) / (self.counts + 1e-05))
            action = np.argmax(ucb_values)
        self.counts[action] += 1
        return action
    
    def reset(self):
        super().reset()
        self.counts = np.zeros(self.env.action_size)


class EpsilonGreedyAgent(Agent):
    def __init__(self, env: Environment, epsilon=0.1):
        super().__init__(env)
        self.epsilon = epsilon  # probability of choosing a random arm (exploration)
    
    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.env.action_size)  # explore
        else:
            return np.argmax(self.q_value[state])  # exploit
