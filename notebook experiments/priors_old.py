import numpy as np
import torch
import torch.nn as nn
import torchvision
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.data import Subset
import torch.nn.functional as F
import matplotlib.pyplot as plt
import random
from torch.distributions import Categorical, Normal, StudentT
import torch.distributions as dist



# Framework for priors --------------------------------------------------------


class Prior:
    """
    This class is a base class for all priors that we use in this project.
    It implements the log_likelihood and sample methods.
    """
    def __init__(self):
        pass
    def sample(self,n):
        pass
    def log_likelihood(self,values):
        pass


# Isotropic Gaussian prior ----------------------------------------------------

class Isotropic_Gaussian(Prior):
    """
    Isotropic Gaussian prior
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super(Isotropic_Gaussian, self).__init__()
        # assert sigma > 0
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood for the given x values
        """
        return dist.Normal(self.mu, self.sigma).log_prob(x).sum() 

    def sample(self) -> torch.Tensor:
        """
        Sample from the prior
        """
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps


# Multivariate Gaussian prior -------------------------------------------------

class MultivariateDiagonalGaussian(Prior):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, Temperature: float = 1.0):
        super(MultivariateDiagonalGaussian, self).__init__()  
        self.mu = mu
        self.rho = rho
        self.sigma = torch.log(1 + torch.exp(rho))
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this
        return dist.Normal(self.mu, self.sigma).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        # TODO: Implement this
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps



# Laplace prior ---------------------------------------------------------------



# Student-t prior -------------------------------------------------------------


# Gaussian mixture prior ------------------------------------------------------

# Normal Inverse Gamma prior --------------------------------------------------

# Spike and slab prior --------------------------------------------------------




# Customized Laplace and Uniform Mixture prior --------------------------------


## Uniform at the middle, Laplace at the sides, 50% weight on uniform, 25% weight each side.

#   if x < -1:              f(x) = 0.67957*exp(x)
#   if -1 <= x <= 1:        f(x) = 1/4
#   if x > 1:               f(x) = 0.67957*exp(-x)

##

class MixedLaplaceUniform(Prior):
    def __init__(self):
        self.a = np.exp(1)/4

    def sample(self, size=1) -> torch.tensor:
        """Generates samples from the mixed probability distribution."""
        samples = np.zeros(size)
        for i in range(size):
            u = np.random.uniform(0,1)
            if u < 1/4:
                samples[i] = np.log(u/self.a)   # Solved CDF to sample x 
            elif u <= 3/4:
                samples[i] = (u - 1/4)*4 - 1            # Solved CDF to sample x
            else:
                b = 1/np.exp(1) - ((u-0.75)/self.a)
                c = 1/b
                samples[i] = np.log(c)
        return torch.tensor(samples)
        
    def log_likelihood(self, values: torch.tensor) -> torch.tensor:
        log_values = []
        for value in values:
            if value < -1:
                val = value + np.log(self.a)
                log_values.append(val)
            elif value <= 1:
                val = torch.tensor(np.log(1/4))
                log_values.append(val)
            else:
                val = -value +np.log(self.a)
                log_values.append(val)

        return sum(log_values)


# Pre-train the prior on FashionMNIST -----------------------------------------