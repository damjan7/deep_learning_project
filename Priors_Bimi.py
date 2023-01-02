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