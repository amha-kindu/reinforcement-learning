from abc import ABC, abstractmethod


class Environment(ABC):
    def __init__(self, state_size: int, action_size: int) -> None:
        self.state_size = state_size
        self.action_size = action_size

    @abstractmethod
    def step(self, state, action):
        pass

    @abstractmethod
    def action_space(self):
        pass

    @abstractmethod
    def state_space(self):
        return

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def is_terminal(self, state):
        pass

    @abstractmethod
    def is_obstacle(self, state):
        pass


class ModelBasedEnv(Environment):
    @abstractmethod
    def transitions(self, state, action):
        pass

    @abstractmethod
    def reward(self, state, action, next_state):
        pass


class ModelFreeEnv(Environment):
    pass
