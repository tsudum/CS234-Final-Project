import random
import numpy as np
from collections import deque


class ReplayBuffer:
    """
    replay buffer used by dqn to store past transitions.

    each stored element is a tuple:
    (state, action, reward, next_state, done)

    the buffer has a fixed size. when it fills up, the oldest entry is discarded.
    """

    def __init__(self, capacity=100000):
        # max number of transitions that can be stored
        self.capacity = capacity

        self.buffer = deque(maxlen=self.capacity)

    def addTransition(self, state, action, reward, next_state, done):
        """
        add one transition from environment into buffer
        """

        transition = (state, action, reward, next_state, done)

        # append the transition to the buffer
        self.buffer.append(transition)

    def sampleBatch(self, batch_size):
        """
        randomly sample a batch of transitions from the buffer.

        purpose: to break correlations between consecutive experiences
        and stabilizes dqn training
        """

        # randomly select batch_size transitions from the buffer
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        # these need to be numpy arrays so we can feed them into pytorch
        states = np.array(states, dtype=np.float32)
        actions = np.array(actions, dtype=np.int64)
        rewards = np.array(rewards, dtype=np.float32)
        next_states = np.array(next_states, dtype=np.float32)

        # done flags indicate whether the episode ended
        dones = np.array(dones, dtype=np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """
        return the current number of stored transitions.
        """

        return len(self.buffer)