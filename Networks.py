import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


""" Base Networks """

""" Fully Connected Neural Network """

""" - 3 Layers (2 Hidden, 1 Output)
    - 100 units each for hidden layers
    - ReLu activations
    """

class FullyConnectedNN(nn.Module):
    def __init__(self, in_features = 28*28, out_features = 10, hidden_units = 100, hidden_layers = 2):
        super(FullyConnectedNN, self).__init__()

        # Create layer list
        self.layers = []

        # Input to first layer
        self.layers.append(nn.Linear(in_features, hidden_units))
        self.layers.append(nn.ReLU())

        # Hidden layers
        for i in range(hidden_layers - 1):
            self.layers.append(nn.Linear(hidden_units, hidden_units))
            self.layers.append(nn.ReLU())
        
        # Output layer
        self.layers.append(nn.Linear(hidden_units, out_features))

        # Convert to sequential
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        output = self.layers(x)
        return output


""" Convolutional Neural Network """

""" - 3 Layers (2 Hidden, 1 Output)
    - First two layers: Convolutional Layers with 64 channels, 3x3 convolutions, followed by 2x2 MaxPooling
    - All layers use ReLu activations
    """

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        
        # Create Network
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 10))
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        output = self.layers(x)
        return output