""" PRIORS """


import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions import Normal
from torch.utils.data import Subset
from torch.distributions import Categorical, Normal, StudentT

""" Framework Priors """

class Prior:
    def __init__(self):
        pass

    def sample(self,n):
        pass

    def log_likelihood(self,values):
        pass


""" Gaussian """

class IsotropicGaussian(Prior):
    def __init__(self, mean=0, std=1, Temperature: float=1.0):
        self.mean = mean
        self.std = std
        self.Temperature = Temperature

    def sample(self, n):
        return np.random.normal(self.mean, self.std, size=n)

    def log_likelihood(self, weights):
        weights_detached_with_grad = weights.clone().detach().requires_grad_(True)
        return Normal(self.mean, self.std).log_prob(weights_detached_with_grad).sum() / self.Temperature


class ISOGaussian(nn.Module):
    def __init__(self, mean=0, std=1, Temperature: float=1.0):
        self.mean = mean
        self.std = std
        self.Temperature = Temperature

    def sample(self, n):
        return np.random.normal(self.mean, self.std, size=n)

    def log_likelihood(self, weights):
        return Normal(self.mean, self.std).log_prob(weights).sum() / self.Temperature


""" Student-T """

class StudentTPrior(Prior):
    def __init__(self, df=10, loc=0, scale=1, Temperature: float= 1.0):
        super().__init__()
        self.df = df
        self.loc = loc
        self.scale = scale
        self.Temperature = Temperature

    def log_likelihood(self, values) -> torch.Tensor:
        return StudentT(self.df, self.loc, self.scale).log_prob(torch.tensor(values)).sum() / self.Temperature

    def sample(self,n):
        return StudentT(self.df, self.loc, self.scale).sample((n,))


""" Laplace """

class LaplacePrior(Prior):
    def __init__(self, mu = torch.tensor(0.), rho = torch.tensor(1.), Temperature: float=1.0):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sig = torch.log(1 + torch.exp(rho))   # transform rho
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.Laplace(self.mu, self.sig).log_prob(torch.tensor(values)).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        return dist.Laplace(self.mu, self.sig).sample(torch.Tensor(1).shape).view(self.mu.shape)


""" Inverse Gamma """

from scipy.special import gamma

class InverseGamma(Prior):
    def __init__(self, shape: torch.Tensor, rate: torch.Tensor, Temperature: float = 1.0):
        """
        shape: shape parameters of the distribution
        rate: rate parameters of the distribution
        """
        self.shape = shape
        self.rate = rate
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            target: Torch tensor of floats, point(s) to evaluate the logprob
        Returns:
            loglike: float, the log likelihood
        """
        x = (self.rate**self.shape) / gamma(self.shape)
        y = torch.tensor(values)**(-self.shape - 1)
        z = torch.exp(-self.rate / values)
        return torch.log(x * y * z)

    def sample(self) -> torch.Tensor:
        # sample from gamma and return 1/x
        x = dist.Gamma(self.shape, self.rate).sample()
        return 1/x


class GaussianMixture(Prior):
    """
    Mixture of 2 gaussians with same mean but different variances
    """
    def __init__(self,  mu: torch.Tensor, rho1: torch.Tensor, rho2: torch.Tensor, mixing_coef: float=0.7 ,Temperature: float = 1.0):
        super().__init__()
        self.mu = mu
        self.rho1 = rho1
        self.rho2 = rho2
        self.sig1 = torch.log(1 + torch.exp(rho1))  # transform rho
        self.sig2 = torch.log(1 + torch.exp(rho2))  # transform rho
        self.mixing_coef = mixing_coef
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        p1 = dist.Normal(self.mu, self.sig1).log_prob(values)
        p2 = dist.Normal(self.mu, self.sig2).log_prob(values)
        log_lik = (p1 * self.mixing_coef + p2 * (1-self.mixing_coef)).sum() / self.Temperature
        return log_lik

    def sample(self) -> torch.Tensor:
        eps = torch.randn_like(self.mu)
        sample1 = self.mu + self.sig1 * eps
        eps = torch.randn_like(self.mu)
        sample2 = self.mu + self.sig2 * eps
        return sample1 * self.mixing_coef + sample2 * (1-self.mixing_coef)


""" Mixed Laplace-Uniform """

## Uniform at the middle, Laplace at the sides, 50% weight on uniform, 25% weight each side.

#   if x < -1:              f(x) = a*exp(x)
#   if -1 <= x <= 1:        f(x) = 1/4
#   if x > 1:               f(x) = a*exp(-x)

#   a = e/4



""" Mixed Laplace Uniform, maybe computational advantage """


class MixedLaplaceUniform(Prior):
    def __init__(self, Temperature:float=1.0):
        self.a = torch.exp(torch.tensor(1))/4
        self.Temperature = Temperature

    def sample(self, size=1) -> torch.tensor:
        """Generates samples from the mixed probability distribution."""
        u = torch.rand(size)
        first_case = torch.log(u/self.a)
        second_case = (u - 1/4)*4 - 1
        third_case = torch.log(1/(1/np.exp(1) - ((u-0.75)/self.a)))
        samples = torch.where(u < 1/4, first_case, 
                              torch.where(u <= 3/4, second_case, third_case))
        return samples
        
    def log_likelihood(self, values: torch.tensor) -> torch.tensor:
        log_likelihoods = torch.where(values < -1, values + torch.log(self.a), torch.where(values <= 1, torch.tensor(np.log(1/4)), -values + torch.log(self.a)))
        return log_likelihoods.sum() / self.Temperature
