""" Base Networks """

""" Fully Connected Neural Network """

""" - 3 Layers (2 Hidden, 1 Output)
    - 100 units each for hidden layers
    - ReLu activations
    """

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.data import Subset
from torch.distributions import Categorical, Normal, StudentT
from torch.optim import SGD
from torch.optim.lr_scheduler import PolynomialLR
import torchvision
from torchvision import datasets, transforms
import torchmetrics
from torchmetrics.functional import calibration_error
import math
import matplotlib.pyplot as plt
import random
from collections import deque, OrderedDict
from tqdm import trange
import tqdm
import copy
import typing
from typing import Sequence, Optional, Callable, Tuple, Dict, Union

class FullyConnectedNN(nn.Module):
    def __init__(self, in_features = 28*28, out_features = 10, hidden_units = 100, hidden_layers = 2):
        super(FullyConnectedNN, self).__init__()

        # Input to first layer
        self.hidden_layers = nn.ModuleList()
        self.hidden_layers.append(nn.Linear(in_features, hidden_units))

        # Hidden layers
        for i in range(hidden_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_units, hidden_units))
        
        # Output layer
        self.output_layer = nn.Linear(hidden_units, out_features)

    def forward(self, x):
        x = x.reshape(-1,28*28)
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        class_probs = self.output_layer(x)
        return class_probs


""" Convolutional Neural Network """

""" - 3 Layers (2 Hidden, 1 Output)
    - First two layers: Convolutional Layers with 64 channels, 3x3 convolutions, followed by 2x2 MaxPooling
    - All layers use ReLu activations
    """

class ConvolutionalNN(nn.Module):
    def __init__(self):
        super(ConvolutionalNN, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1)
        # Second convolutional layer
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=1)
        # Fully connected layer
        self.fc1 = nn.Linear(64 * 7 * 7, 10)
        
    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = F.relu(F.max_pool2d(F.relu(self.conv1(x)), 2))
        x = F.relu(F.max_pool2d(F.relu(self.conv2(x)), 2))
        x = x.view(-1, 64 * 7 * 7)
        class_probs = self.fc1(x)
        return class_probs

