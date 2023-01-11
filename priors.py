import numpy as np
import torch
import torch.nn as nn
import torch.distributions as dist
from scipy.special import gamma
import math



# Framework for priors --------------------------------------------------------

class Prior:
    """
    This class is a base class for all priors that we use in this project.
    It enforces the implementation of the the log_likelihood and sample methods.
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
    Isotropic Gaussian prior with mean (loc) and standard deviation (scale) as parameters.
    """
    def __init__(self, loc: float = 0, scale: float = 1.0, Temperature: float = 1.0):
        super().__init__()
        assert scale > 0, "Scale must be positive"
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Isotropic Gaussian"

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.Normal(self.loc, self.scale).log_prob(values).sum() / self.Temperature

    def sample(self, n):
        return dist.Normal(self.loc, self.scale).sample((n,))


# Multivariate Gaussian prior -------------------------------------------------


# TODO: Implement this

class Multivariate_Diagonal_Gaussian(Prior):
    """
    Multivariate diagonal Gaussian distribution,
    i.e., assumes all elements to be independent Gaussians
    but with different means and standard deviations.
    This parameterizes the standard deviation via a parameter rho as
    sigma = softplus(rho).
    """
    def __init__(self, mu: torch.Tensor, rho: torch.Tensor, Temperature: float = 1.0):
        super().__init__()
        self.mu = mu
        self.rho = rho
        self.sigma = torch.log(1 + torch.exp(rho))
        self.Temperature = Temperature
        self.name = "Multivariate Diagonal Gaussian"

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        # TODO: Implement this
        return dist.Normal(self.mu, self.sigma).log_prob(values).sum() / self.Temperature

    def sample(self) -> torch.Tensor:
        # TODO: Implement this
        eps = torch.randn_like(self.mu)
        return self.mu + self.sigma * eps



# Student-t prior -------------------------------------------------------------

class StudentT_prior(Prior):
    """
    Student-T Prior with degrees of freedom (df), mean (loc) and scale (scale) as parameters.
    """
    def __init__(self, df: float = 10, loc: float = 0, scale: float = 1.0, Temperature: float = 1.0):
        super().__init__()
        self.df = torch.tensor(df, dtype=torch.float32)
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Student-T"

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.StudentT(self.df, self.loc, self.scale).log_prob(values).sum() / self.Temperature

    def sample(self, n):
        return dist.StudentT(self.df, self.loc, self.scale).sample((n,))  


# Laplace prior ---------------------------------------------------------------

class Laplace_prior(Prior):
    """
    Laplace Prior with mean (loc) and scale (scale) as parameters.
    """
    def __init__(self, loc: float = 0, scale: float = 1.0, Temperature: float = 1.0):
        super().__init__()
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.scale = torch.tensor(scale, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Laplace"

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return dist.Laplace(self.loc, self.scale).log_prob(values).sum() / self.Temperature

    def sample(self, n) -> torch.Tensor:
        return dist.Laplace(self.loc, self.scale).sample((n,))



# Gaussian mixture prior ------------------------------------------------------

class Gaussian_Mixture(Prior):
    """
    Mixture of two Gaussians with means (loc1, loc2), standard deviations (scale1, scale2) and mixing coefficient (mixing_coef) as parameters.
    """
    def __init__(self, loc1: float = 0, scale1: float = 3.0, loc2: float = 0, scale2: float = 1.0,
                mixing_coef: float = 0.7, Temperature: float = 1.0):
        super().__init__()
        self.loc1 = torch.tensor(loc1, dtype=torch.float32)
        self.loc2 = torch.tensor(loc2, dtype=torch.float32)
        self.scale1 = torch.tensor(scale1, dtype=torch.float32)
        self.scale2 = torch.tensor(scale2, dtype=torch.float32)
        self.mixing_coef = torch.tensor(mixing_coef, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Gaussian Mixture"

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        p1 = dist.Normal(self.loc1, self.scale1).log_prob(values)
        p2 = dist.Normal(self.loc2, self.scale2).log_prob(values)
        log_lik = (p1 * self.mixing_coef + p2 * (1-self.mixing_coef)).sum() / self.Temperature
        return log_lik

    def sample(self, n) -> torch.Tensor:
        sample1 = dist.Normal(self.loc1, self.scale1).sample((n,))
        sample2 = dist.Normal(self.loc2, self.scale2).sample((n,))
        return sample1 * self.mixing_coef + sample2 * (1-self.mixing_coef)


# Normal Inverse Gamma prior --------------------------------------------------

class Inverse_Gamma(Prior):
    """ 
    Inverse Gamma distribution with shape (shape) and rate (rate) as parameters.
    This distribution is needed for the Normal Inverse Gamma prior.
    """
    def __init__(self, shape: float = 1.0, rate: float = 1.0, Temperature: float = 1.0):
        super().__init__()
        self.shape = torch.tensor(shape, dtype=torch.float32)
        self.rate = torch.tensor(rate, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Computes the value of the predictive log likelihood at the target value
        """
        x = (self.rate**self.shape) / gamma(self.shape)
        y = values**(-self.shape - 1)
        z = torch.exp(-self.rate / values)
        return torch.log(x * y * z)

    def sample(self, n) -> torch.Tensor:
        # sample from gamma and return 1/x
        x = dist.Gamma(self.shape, self.rate).sample((n,))
        return 1/x



class Normal_Inverse_Gamma(Prior):
    """ 
    Normal Inverse Gamma distribution with mean (mu), precision (lam), shape (alpha) and rate (beta) as parameters.
    """
    def __init__(self, loc: float = 0, lam: float = 1, alpha: float = 1, beta: float = 1, Temperature: float = 1.0):
        """
        loc: loc of the normal distribution
        lam: precision of the normal distribution
        alpha: shape parameter of the inverse gamma distribution
        beta: rate parameter of the inverse gamma distribution
        """
        super().__init__()
        self.loc = torch.tensor(loc, dtype=torch.float32)
        self.lam = torch.tensor(lam, dtype=torch.float32)
        self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.beta = torch.tensor(beta, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Normal Inverse Gamma"
    

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        """
        Compute the likelihood of the inverse gamma distribution for an x and a variance
        """
        # manually compute the likelihood
        var = torch.var(values)

        #log_like = torch.xlogy(0.5, self.lam / (2 * math.pi * var)) + \
        #            torch.xlogy(self.alpha,self.beta) - \
        #            torch.lgamma(self.alpha) - \
        #            torch.xlogy(self.alpha + 1,var) - \
        #            (2*self.beta + self.lam * (values - self.loc)**2) / (2 * var)
        #
        #return log_like.mean() / self.Temperature

        #new try:
        # for all pos integers, gamma function:
        # F(a) = (a-1)!

        var = self.beta / (self.alpha + 1 + 0.5)  # according to formula on wikipedia
        part1 = self.lam ** 0.5 / (2 * np.pi * var) ** 0.5
        part2 = self.beta ** self.alpha / np.math.factorial(self.alpha-1)
        part3 = (1/var) ** (self.alpha + 1)
        part4 = (-2*self.beta - self.lam * (values - self.mu) ** 0.5) / (2*var)
        likelihood = part1 * part2 * part3 * np.exp(part4)
        log_likelihood = np.log(likelihood)
        return log_likelihood / self.Temperature


    def sample(self, n) -> torch.Tensor:
        # sample variance from inverse gamma and sample x from normal given the variance
        var = torch.div(1, dist.Gamma(self.alpha, self.beta).sample((1,)))
        x = dist.Normal(self.loc, torch.sqrt(var/self.lam)).sample((n,))
        return x, var


# Spike and slab prior --------------------------------------------------------

class GaussianSpikeNSlab(Prior):
    """
    theta is the parameter for the bernoulli distribution
    z ~ bern(theta)
    if z=0, then x=0 approximately ("spike distribution", modelled as a very narrow normal distribution)
    if z=1, then x ~ p_slab ("slab distribution")
    We use the normal distribution as the slab distribution
    p_theta(x) = theta * p_spike(x) + (1-theta) * p_slab(x)
    """

    def __init__(self, loc_slab: float = 0, scale_slab: float = 1, loc_spike: float = 0, scale_spike: float = 1e-16, theta: float = 0.8, Temperature: float = 1.0):
        """
        loc_slab: mean of the normal distribution
        scale_slab: standard deviation of the normal distribution
        loc_spike: mean of the spike distribution
        scale_spike: standard deviation of the spike distribution, should be very small to simulate a spike
        theta: parameter of the bernoulli distribution for the mixture of the spike and the slab
        """
        super().__init__()
        self.theta = torch.tensor(theta, dtype=torch.float32)
        self.loc_slab = torch.tensor(loc_slab, dtype=torch.float32)
        self.scale_slab = torch.tensor(scale_slab, dtype=torch.float32)
        self.loc_spike = torch.tensor(loc_spike, dtype=torch.float32)
        self.scale_spike = torch.tensor(scale_spike, dtype=torch.float32)
        self.Temperature = torch.tensor(Temperature, dtype=torch.float32)
        self.name = "Gaussian Spike and Slab"

        mix = dist.Categorical(probs=torch.tensor([1-self.theta, self.theta]))
        comp = dist.Normal(torch.tensor([self.loc_spike, self.loc_slab]), torch.tensor([self.scale_spike, self.scale_slab]))
        self.spike_n_slab = dist.MixtureSameFamily(mix, comp) 

    def log_likelihood(self, values: torch.Tensor) -> torch.Tensor:
        return self.spike_n_slab.log_prob(values).sum() / self.Temperature

    def sample(self,n) -> torch.Tensor:
        return self.spike_n_slab.sample((n,))


# Customized Laplace and Uniform Mixture prior --------------------------------


## Uniform at the middle, Laplace at the sides, 50% weight on uniform, 25% weight each side.

#   if x < -1:              f(x) = 0.67957*exp(x)
#   if -1 <= x <= 1:        f(x) = 1/4
#   if x > 1:               f(x) = 0.67957*exp(-x)


class MixedLaplaceUniform(Prior):
    """
    A mixture of Laplace and Uniform distributions.
    we use a Uniform fistribution in the middle within interval [-1, 1] and a Laplace distribution at the sides.
    The distribution is continuous. 
    """
    def __init__(self, Temperature:float=1.0):
        super().__init__()
        self.a = torch.exp(torch.tensor(1))/4
        self.Temperature = Temperature
        self.name = "Mixed Laplace and Uniform"
        
    def log_likelihood(self, values: torch.tensor) -> torch.tensor:
        log_likelihoods = torch.where(values < -1, values + torch.log(self.a), torch.where(values <= 1, torch.tensor(np.log(1/4)), -values + torch.log(self.a)))
        return log_likelihoods.sum() / self.Temperature

    def sample(self, size=1) -> torch.tensor:
        """Generates samples from the mixed probability distribution."""
        u = torch.rand(size)
        first_case = torch.log(u/self.a)
        second_case = (u - 1/4)*4 - 1
        third_case = torch.log(1/(1/np.exp(1) - ((u-0.75)/self.a)))
        samples = torch.where(u < 1/4, first_case, 
                              torch.where(u <= 3/4, second_case, third_case))
        return samples



# Pre-train the prior on FashionMNIST -----------------------------------------