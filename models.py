import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torchvision
import tqdm
import matplotlib.pyplot as plt
import torch.distributions as dist
import abc

from torch.utils.data import DataLoader
from torch.distributions.distribution import Distribution
from priors import *



class Linear_Layer(nn.Module):
    """
    Bayesian Linear Layer that will be used as a building block for the Bayesian Neural Network
    https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/models/layers.py
    """ 
    def __init__(self, in_features, out_features, bias = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.with_bias = bias

        # create a prior for the weights and biases using the Isotropic Gaussian prior
        self.weight_prior = Isotropic_Gaussian(mu = torch.tensor(0.), sigma = torch.tensor(1.))

        if self.with_bias:
            self.bias_prior = Isotropic_Gaussian(mu = torch.tensor(0.), sigma = torch.tensor(1.))

        
        # create a variational posterior for the weights and biases as Instance of  Multivariate Gaussian
        self.mu_weight = torch.nn.Parameter(torch.zeros(out_features, in_features))
        self.sigma_weight = torch.nn.Parameter(torch.ones(out_features, in_features))

        self.weight_posterior = Isotropic_Gaussian(self.mu_weight, self.sigma_weight)
        
        if self.with_bias:
            self.mu_bias = torch.nn.Parameter(torch.zeros(out_features))
            self.sigma_bias = torch.nn.Parameter(torch.ones(out_features))
            self.bias_posterior = Isotropic_Gaussian(self.mu_bias, self.sigma_bias)
        else:
            self.register_parameter('bias', None)
        

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Same procedure as explained in the paper of Blundell et al. (2015):
        use reparameterization trick to sample weights and biases from the variational posterior
        """
        # usample random noise from the standard normal distribution
        epsilon = torch.randn_like(self.weight_posterior.mu)

        # sample weights and biases from the variational posterior
        weight = self.weight_posterior.mu + torch.multiply(epsilon, self.weight_posterior.sigma)

        # compute the log prior of the weights and biases
        log_prior = self.weight_prior.log_likelihood(weight)

        # compute the log variational posterior of the weights and biases
        log_posterior = self.weight_posterior.log_likelihood(weight)

        # adjust for the bias
        if self.with_bias:
            epsilon = torch.randn_like(self.bias_posterior.mu)
            bias = self.bias_posterior.mu + torch.multiply(epsilon, self.bias_posterior.sigma)
            log_prior += self.bias_prior.log_likelihood(bias)
            log_posterior += self.bias_posterior.log_likelihood(bias)
        else:
            bias = None


        kl_divergence = log_posterior - log_prior
        
        # compute the output of the layer
        output = F.linear(x, weight, bias)
        return output, kl_divergence




class Bayesian_Neural_Network(nn.Module):
    """
    Bayesian Neural Network that will be trained using the BNN implementation
    """ 
    def __init__(self, input_dim, output_dim, hidden_dims):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims

        # create the layers of the network
        self.layers = nn.ModuleList()

        # create the input layer
        self.layers.append(Linear_Layer(self.input_dim, self.hidden_dims[0]))
        #self.layers.append(nn.ReLU(inplace=True))

        # create the hidden layers
        for i in range(len(self.hidden_dims) - 1):
            self.layers.append(Linear_Layer(self.hidden_dims[i], self.hidden_dims[i + 1]))
            #self.layers.append(nn.ReLU(inplace=True))

        # create the output layer
        self.layers.append(Linear_Layer(self.hidden_dims[-1], self.output_dim))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the network
        """
        kl_divergence = torch.tensor(0.0)
        for ind, layer in enumerate(self.layers):
            x, kl = layer(x)
            #kl_divergence += kl
            if ind < len(self.layers) - 1:
                x = F.relu(x)

        return x, kl


    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict the output of the network and return the output
        """
        with torch.no_grad():
            output, _ = self.forward(x)
            return output


    def predict_probs(self, x: torch.Tensor, num_samples: int = 50) -> torch.Tensor:
        """
        Predict the output probabilities for the given features using the predictive distribution by sampling from the BNN
        """
        with torch.no_grad():
            samples = torch.stack([F.softmax(self.forward(x)[0], dim = 1) for _ in range(num_samples)], dim=0)
            prob_estimates = torch.mean(samples, dim=0)
            return prob_estimates



