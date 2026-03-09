import torch
import torch.nn as nn


class DQNNetwork(nn.Module):
    """
    convolutional q-network for MineRL treechop observations
    """

    def __init__(self, input_channels=4, num_actions=8):
        super().__init__()

        self.input_channels = input_channels
        self.num_actions = num_actions

        # convolutional feature extractor
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

        # fully connected layers that should output one q-value per action
        self.fully_connected_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 512),
            nn.ReLU(),
            nn.Linear(512, self.num_actions),
        )

    def computeFlattenedSize(self):
        """
        convolution layers output a feature map, not a flat vector, so we 
        run a dummy tensor through conv layers to determine flattened size
        """

        dummy_input = torch.zeros(1, self.input_channels, 64, 64)
        conv_output = self.convolution_layers(dummy_input)
        flattened_size = conv_output.view(1, -1).size(1)

        return flattened_size

    def forward(self, observation_batch):
        
        # pass images through CNN
        conv_features = self.convolution_layers(observation_batch)
        
        # flatten spatial features
        flattened_features = conv_features.view(conv_features.size(0), -1)

        # compute q-vals
        q_values = self.fully_connected_layers(flattened_features)

        return q_values