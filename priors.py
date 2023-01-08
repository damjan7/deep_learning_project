import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
import torch.utils.data
import torch.distributions as dist
import abc




class Prior(nn.Module, abc.ABC):
    """
    This class is a base class for all priors.
    It implements the log_likelihood and sample methods.
    The forward method is not used, but is required by nn.Module.
    This part of the code is inspired by the code from Vincent Fortuin:
    https://github.com/ratschlab/bnn_priors/blob/main/bnn_priors/prior/base.py
    """
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def log_likelihood(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute the log-likelihood for the given x values
        """
        pass
    
    @abc.abstractmethod
    def sample(self) -> torch.Tensor:
        """
        Sample from the prior
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Don't use this method, we only implement it because nn.Module requires it
        Vincent Fortuin uses the forward to return the parameter value using self.p
        """
        return self.log_likelihood(x)



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