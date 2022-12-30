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
        #assert sigma > 0
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
        return dist.Normal(self.mu, self.sigma).sample()



class MultivariateDiagonalGaussian(Prior):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """
    def __init__(self, mu: torch.Tensor, sigma: torch.Tensor):
        super(MultivariateDiagonalGaussian, self).__init__()  
        self.mu = mu
        self.sigma = sigma

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this
        sigma = self.sigma
        mu = self.mu
        sigma_resized = torch.reshape(sigma, (-1,))
        mu_resized = torch.reshape(mu, (-1,))
        values_resized = torch.reshape(values, (-1,))
        COV = torch.diag(sigma_resized)
        p = torch.tensor(sigma_resized.size())  # dimension of diagonal matrix
        m = torch.tensor(values_resized.size())  # number of samples (=X)

        loglik = -m * p / 2 - m / 2 + torch.log(torch.linalg.det(COV)) - 1 / 2 * (values_resized - mu_resized).t() * torch.inverse(COV) * (values_resized - mu_resized)
        return loglik

    def sample(self) -> torch.Tensor:
        # TODO: Implement this
        sigma = self.sigma
        mu = self.mu
        sigma_resized = torch.reshape(sigma, (-1,))
        mu_resized = torch.reshape(mu, (-1,))
        COV = torch.diag(sigma_resized)
        val = torch.normal(mu_resized, COV)
        return val