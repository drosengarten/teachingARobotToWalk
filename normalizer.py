class Normalizer():
    """Normalizer Class is from https://gist.github.com/ADLsourceCode/60c72efee14e318fd4eb177f8121dc5b"""
    def __init__(self, n_inputs):
        import numpy as np
        #Here we create empty arrays the size of our inputs
        self.n = np.zeros(n_inputs)
        self.mean = np.zeros(n_inputs)
        self.mean_diff = np.zeros(n_inputs)
        self.var = np.zeros(n_inputs)

    def observe(self, x):
        #From our inputs we gate array average "mean" and calculate the variance
        self.n += 1
        last_mean = self.mean.copy()
        self.mean += ( x - last_mean) / self.n
        self.mean_diff += (x - last_mean) * (x - self.mean)
        self.var = (self.mean_diff / self.n).clip(min = 1e-4)  # clip so we don't divede by zero

    def normalize(self, state):
        import numpy as np
        # normalize inputs 
        o_mean = self.mean
        o_std = np.sqrt(self.var)
        return (state - o_mean) / o_std
