""" Bayesian Neural Network """

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

from SGLD import SGLD


class BNN_MCMC:
    def __init__(self, dataset_train, network, prior,
     num_epochs = 300, max_size = 100, burn_in = 100, lr = 1e-3, sample_interval = 1):

        # Hyperparameters and general parameters
        self.learning_rate = lr
        self.num_epochs = num_epochs
        self.burn_in = burn_in
        self.sample_interval = sample_interval
        self.max_size = max_size

        self.batch_size = 128
        self.print_interval = 50
        
        # Data Loader
        self.data_loader = DataLoader(dataset_train, batch_size=self.batch_size, shuffle=True)
        self.sample_size = dataset_train.__len__()

        # Set Prior
        self.prior = prior

        # Initialize the network
        self.network = network

        # Set optimizer
        self.optimizer = SGLD(self.network.parameters(), lr=self.learning_rate, num_data=self.batch_size)
        self.optimizer = SGD(self.network.parameters(), lr=self.learning_rate)

        # Scheduler for polynomially decreasing learning rates
        self.scheduler = PolynomialLR(self.optimizer, total_iters = self.num_epochs, power = 0.5)

        # Deque to store model samples
        self.model_sequence = deque()

    def train(self):
        num_iter = 0
        print('Training Model')

        self.network.train()
        progress_bar = trange(self.num_epochs)

        N = self.sample_size

        for _ in progress_bar:
            num_iter += 1

            for batch_idx, (batch_x, batch_y) in enumerate(self.data_loader):
                self.network.zero_grad()
                n = len(batch_x)

                # Perform forward pass
                current_logits = self.network(batch_x)

                # Calculate log_likelihood of weights for a given prior

                parameters = self.network.state_dict()     # extract weights from network
                param_values = list(parameters.values())    # list weights
                param_flat = np.concatenate([v.flatten() for v in param_values])    # flattern
                log_prior = self.prior.log_likelihood(param_flat)              # calculate log_lik

                # Calculate the loss
                loss = N/n*F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y) - log_prior#/#len(param_flat)

                # Backpropagate to get the gradients
                loss.backward(retain_graph=True)

                # Update the weights
                self.optimizer.step()

                # Update Metrics according to print_interval
                if batch_idx % self.print_interval == 0:
                    current_logits = self.network(batch_x)
                    current_accuracy = (current_logits.argmax(axis=1) == batch_y).float().mean()
                    progress_bar.set_postfix(loss=loss.item(), acc=current_accuracy.item(),
                    nll_loss=N/n*F.nll_loss(F.log_softmax(current_logits, dim=1), batch_y).item(),
                    log_prior_normalized = - log_prior.item()/len(param_flat),
                    lr = self.optimizer.param_groups[0]['lr'])

            # Decrease lr based on scheduler
            self.scheduler.step()
            
            # Save the model samples if past the burn-in epochs according to sampling interval
            if num_iter > self.burn_in and num_iter % self.sample_interval == 0:
                self.model_sequence.append(copy.deepcopy(self.network))
                # self.network.state_dict()

            # If model_sequence to big, delete oldest model
            if len(self.model_sequence) > self.max_size:
                self.model_sequence.popleft()

    def predict_probabilities(self, x: torch.Tensor) -> torch.Tensor:
        self.network.eval()

        # Sum predictions from all models in model_sequence
        estimated_probability = torch.zeros((len(x), 10))

        for model in self.model_sequence:

            self.network.load_state_dict(model.state_dict())
            logits = self.network(x).detach()
            estimated_probability += F.softmax(logits, dim=1)
        
        # Normalize the combined predictions to get average predictions
        estimated_probability /= len(self.model_sequence)

        assert estimated_probability.shape == (x.shape[0], 10)  
        return estimated_probability
    
    def test_accuracy(self,x):
        # test set
        x_test = x[:][0].clone().detach()
        y_test = x[:][1].clone().detach()      

        # predicted probabilities
        class_probs = self.predict_probabilities(x_test)

        # accuracy
        accuracy = (class_probs.argmax(axis=1) == y_test).float().mean()
        return  accuracy #print(f'Test Accuracy: {accuracy.item():.4f}')

    def test_calibration(self,x):
        # test set
        x_test = x[:][0].clone().detach()
        y_test = x[:][1].clone().detach()         

        # predicted probabilities
        class_probs = self.predict_probabilities(x_test)

        calib_err = calibration_error(class_probs, y_test, n_bins = 30, task = "multiclass", norm="l1", num_classes=10)
        return calib_err #print(f'Calibration Error: {calib_err.item():.4f}')