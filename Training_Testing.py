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

from Priors import IsotropicGaussian, StudentTPrior, LaplacePrior, InverseGamma, GaussianMixture, MixedLaplaceUniform, MixedLU
from Networks import FullyConnectedNN, ConvolutionalNN
from Fortuin_SGLD import Fortuin_SGLD
from BayesianNN import BayesianNN

# MNIST dataset
transform = transforms.Compose([torchvision.transforms.ToTensor()])
train_set = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_set = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# subsample from trainset
n_subsamples_train = 2000 # size of subset
sub_train_idx = random.sample(range(60000),n_subsamples_train)
sub_train_set = Subset(train_set, sub_train_idx)

lol = BayesianNN(sub_train_set,network = FullyConnectedNN(), prior=IsotropicGaussian(), num_epochs=200)
lol.train()
