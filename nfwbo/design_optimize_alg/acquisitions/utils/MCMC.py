"""
https://github.com/amzn/emukit/blob/master/emukit/samplers/mcmc_sampler.py
Not sure about the advantages brought by ensemble MCMC over vanilla MCMC
"""
import numpy as np
import emcee


class MCMCSampler(object):
    def __init__(self, config, logger=None):
        """
        Creates an instance of the sampler.
        self.space = space
        """
        self.logger = logger
        self.config = config
        self.configure()

    def configure(self):
        config = self.config
        self.dim = config["dim"]
        self.bounds = config["bounds"]
        self.burn_in_steps = 100#config["burn_in_steps"]
        self.n_steps = config["n_steps"]

    def get_samples(self, n_samples, log_p_function):
        """
        Generates samples.

        Parameters:
            n_samples - number of samples to generate
            log_p_function - a function that returns log density for a specific sample
            burn_in_steps - number of burn-in steps for sampling

        Returns a tuple of two lists: (samples, log_p_function values for samples)
        """
        self.logger.debug('start emcee.ensemble')
        X_init = np.random.uniform(low=self.bounds[0, :], high=self.bounds[1, :], size=(n_samples, self.dim))
        sampler = emcee.EnsembleSampler(n_samples, X_init.shape[1], log_p_function)

        self.logger.debug('start burn in')
        # Burn-In
        state = list(sampler.run_mcmc(X_init, self.burn_in_steps))
        samples = state[0]

        self.logger.debug('start mcmc sampling')
        # MCMC Sampling
        state = list(sampler.run_mcmc(samples, self.n_steps))
        samples = state[0]
        samples_log = state[1]

        # make sure we have an array of shape (n samples, dim)
        if len(samples.shape) == 1:
            samples = samples.reshape(-1, 1)
        samples_log = samples_log.reshape(-1, 1)

        return samples, samples_log



