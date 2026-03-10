import torch
import torch.nn as nn


class PPONetwork(nn.Module):
    """
    actor-critic convolutional network for MineRL treechop observations

    shares a CNN backbone between the policy (actor) and value function (critic),
    then splits into two separate heads
    """

    def __init__(self, input_channels=4, num_actions=8):
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # shared convolutional feature extractor (same architecture as DQNNetwork)
        self.convolution_layers = nn.Sequential(
            nn.Conv2d(
                in_channels=self.input_channels,
                out_channels=32,
                kernel_size=8,
                stride=4,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2,
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1,
            ),
            nn.ReLU(),
        )

        self.flattened_size = self.computeFlattenedSize()

        # shared fully connected trunk
        self.shared_fc = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
        )

        # actor head: outputs logits over discrete actions
        self.actor_head = nn.Linear(512, self.num_actions)

        # critic head: outputs a single scalar value V(s)
        self.critic_head = nn.Linear(512, 1)

    def computeFlattenedSize(self):
        """
        run a dummy tensor through conv layers to figure out the flattened size
        """
        dummy_input = torch.zeros(1, self.input_channels, 64, 64)
        conv_output = self.convolution_layers(dummy_input)
        return conv_output.view(1, -1).size(1)

    def forward(self, observation_batch):
        """
        Args:
            observation_batch: torch.Tensor of shape [batch, channels, H, W]
        Returns:
            action_logits: torch.Tensor of shape [batch, num_actions]
            state_values:  torch.Tensor of shape [batch]
        """
        conv_features = self.convolution_layers(observation_batch)
        flattened = conv_features.view(conv_features.size(0), -1)
        shared_features = self.shared_fc(flattened)

        action_logits = self.actor_head(shared_features)
        state_values = self.critic_head(shared_features).squeeze(-1)

        return action_logits, state_values

    def getActionDistribution(self, observation_batch):
        """
        returns a Categorical distribution over actions given observations
        """
        action_logits, _ = self.forward(observation_batch)
        return torch.distributions.Categorical(logits=action_logits)

    def getValue(self, observation_batch):
        """
        returns only the critic value V(s)
        """
        _, state_values = self.forward(observation_batch)
        return state_values