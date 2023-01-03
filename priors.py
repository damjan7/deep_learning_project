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


class LaplacePrior(Prior):
    """
    Laplace Prior
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, Temperature: float=1.0):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sig = torch.log(1 + torch.exp(rho))   # transform rho
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.Laplace(self.mu, self.sig).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        return dist.Laplace(self.mu, self.sig).sample(torch.Tensor(1).shape).view(self.mu.shape)  # sample from laplace



class StudentTPrior(Prior):
    """
    Student-T Prior
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, Temperature: float= 1.0, df: torch.Tensor = 10):
        super().__init__()
        self.df = df
        self.mu = mu
        self.rho = rho
        self.rho = torch.log(1 + torch.exp(rho))  # transform rho
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.StudentT(self.df, self.mu, self.rho).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        return dist.StudentT(self.df, self.mu, self.loc).sample(self.mu.shape)  # sample from student-T



class SpikeSlabPrior(Prior):
    """
    theta is the parameter for the bernoulli distribution
    z ~ bern(theta)
    if z=0, x=0
    if z=1, x ~ p_slab ("slab distribution")
    We use the normal distribution as the slab distribution
    p_theta(x) = theta * p_spike(x) + (1-theta) * p_slab(x)
    """

    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, theta: torch.tensor = torch.tensor(0.8), Temperature: float = 1.0):
        super().__init__()
        self.theta = theta
        self.mu = mu
        self.rho = rho
        self.sig = torch.log(1 + torch.exp(rho))  # transform rho
        self.Temperature = Temperature

        # self.mu.shape --> this is the shape we need a sample from
        # input_size * output_size
        # hence for every entry on that grid we sample from bernoulli as well

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # just for testing, doesnt actually make sense
        return dist.Normal(self.mu, self.sig).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        bern = dist.Bernoulli(torch.tensor([self.theta]))
        bern_rvs = bern.sample(torch.Size([self.mu.shape])).view(self.mu.shape)
        #now everywhere where bern_rvs is = 1, sample from the slab distribution
        eps = torch.randn_like(self.mu)
        rvs_normal = self.mu + self.sig * eps
        return rvs_normal * bern_rvs  # element-wise mat mul



### NOT USED
class Wishart(Prior):
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, Temperature: float= 1.0, df: torch.Tensor = 10):
        super().__init__()
        self.df = df
        self.mu = mu
        self.rho = rho
        self.rho = torch.log(1 + torch.exp(rho))  # transform rho
        self.Temperature = Temperature

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.StudentT(self.df, self.mu, self.rho).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        return dist.StudentT(self.df, self.mu, self.loc).sample(self.mu)  # sample from student-T
