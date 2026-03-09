import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from src.agents.DQNNetwork import DQNNetwork
from src.agents.replayBuffer import ReplayBuffer


class DQNAgent:
    """
    deep q-learning agent for MineRL treechop
    """

    def __init__(
        self, # play around with these later on
        state_channels=4,
        num_actions=8,
        learning_rate=1e-4,
        gamma=0.99,
        replay_buffer_capacity=100000,
        batch_size=32,
        epsilon_start=1.0,
        epsilon_end=0.1,
        epsilon_decay=100000,
        target_update_frequency=1000,
        device=None,
    ):
        # bookkeeping
        self.state_channels = state_channels
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size

        # epsilon-greedy exploration settings
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        # how often to copy online network weights into target network
        self.target_update_frequency = target_update_frequency

        # number of gradient updates
        self.training_step_count = 0

        # choose cpu or cuda automatically
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # online q-network used for action selection and learning
        self.q_network = DQNNetwork(
            input_channels=self.state_channels,
            num_actions=self.num_actions,
        ).to(self.device)

        # target q-network used to build stable td targets
        self.target_network = DQNNetwork(
            input_channels=self.state_channels,
            num_actions=self.num_actions,
        ).to(self.device)

        # initialize target network with same weights as online network
        self.updateTargetNetwork()

        self.optimizer = optim.Adam(
            self.q_network.parameters(),
            lr=learning_rate,
        )

        # for td learning
        self.loss_function = nn.MSELoss()
        
        self.replay_buffer = ReplayBuffer(capacity=replay_buffer_capacity)

    def getEpsilon(self, current_step):
        """
        compute epsilon using linear decay from epsilon_start to epsilon_end
        """

        decay_progress = min(current_step / self.epsilon_decay, 1.0)
        epsilon = self.epsilon_start + decay_progress * (self.epsilon_end - self.epsilon_start)

        return epsilon

    def selectAction(self, state, current_step):
        """
        choose an action using epsilon-greedy exploration
        """

        epsilon = self.getEpsilon(current_step)

        # with probability epsilon, explore using a random action
        if random.random() < epsilon:
            return random.randrange(self.num_actions)

        # otherwise, use the online q-network to choose the greedy/best action
        state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)

        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            action_index = torch.argmax(q_values, dim=1).item()

        return action_index

    def storeTransition(self, state, action, reward, next_state, done):
        """
        add one transition to the replay buffer
        """

        self.replay_buffer.addTransition(state, action, reward, next_state, done)

    def updateTargetNetwork(self):
        """
        copy online network weights into the target network
        """

        self.target_network.load_state_dict(self.q_network.state_dict())

    def updateNetwork(self):
        """
        run one dqn training update using a random minibatch from replay buffer
        """

        # only train when enough transitions have occured, stops premature training
        if len(self.replay_buffer) < self.batch_size:
            return None

        # sample a minibatch of transitions from replay memory
        states, actions, rewards, next_states, dones = self.replay_buffer.sampleBatch(
            self.batch_size
        )

        # converting into torch tensors
        states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_tensor = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_tensor = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_states_tensor = torch.tensor(next_states, dtype=torch.float32, device=self.device)
        dones_tensor = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # compute q-values for the current states using the online network
        current_q_values = self.q_network(states_tensor)

        # select only the q-values corresponding to the actions actually taken
        chosen_q_values = current_q_values.gather(1, actions_tensor.unsqueeze(1)).squeeze(1)

        # compute target q-values using the target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states_tensor)
            max_next_q_values = torch.max(next_q_values, dim=1)[0]

            # if done = 1, the future term is removed because we have reached the end of that episode
            target_q_values = rewards_tensor + self.gamma * max_next_q_values * (1.0 - dones_tensor)

        # compute td loss between predicted q-values and target q-values
        loss = self.loss_function(chosen_q_values, target_q_values)

        # run one gradient descent step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # track how many optimization steps have been taken
        self.training_step_count += 1

        # periodically refresh the target network
        if self.training_step_count % self.target_update_frequency == 0:
            self.updateTargetNetwork()

        return loss.item()