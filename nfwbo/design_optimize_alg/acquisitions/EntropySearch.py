import numpy as np
import scipy

from design_optimize_alg.acquisitions.utils.MCMC import MCMCSampler
from design_optimize_alg.acquisitions.utils import epmgp
from design_optimize_alg.acquisitions.ExpectedImprovement import ExpectedImprovement


class EntropySearch(object):
    def __init__(self, gps, config, logger=None):
        """
        http://jmlr.csail.mit.edu/papers/volume13/hennig12a/hennig12a.pdf
        """
        self.logger = logger
        self.gps = gps
        self.config = config
        self.configure()
        self.p_min_entropy = None

    def configure(self):
        config = self.config
        self.num_representer_points = config["num_representer_points"]
        self.num_samples = config["num_samples"]
        self._config_sampler()
        self._config_proposal_function()

    def _config_proposal_function(self):
        bounds = self.config['sampler']['bounds']
        ei = ExpectedImprovement(self.gps, self.logger)

        def proposal_func(x):
            if len(x.shape) == 1:
                x_ = x[None, :]
            else:
                x_ = x
            if np.any(np.isnan(np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF)))):
                raise ValueError('proposal return cannot be nan')
            elif np.all([np.greater_equal(x_, bounds[0, :]), np.greater_equal(bounds[1, :], x_)]) and np.all(np.greater(ei.evaluate(x_)[0], 0)):
                return np.log(np.clip(ei.evaluate(x_)[0], 0., np.PINF))
            else:
                return np.array([np.NINF])

        self.proposal_func = proposal_func

    def _config_sampler(self):
        config = self.config['sampler']
        self.sampler = MCMCSampler(config, self.logger)

    def update(self):
        '''
        p_xopt: distribution over optimal x
        '''
        self._update_p_xopt()

    def _update_p_xopt(self):
        '''
        http://jmlr.csail.mit.edu/papers/volume13/hennig12a/hennig12a.pdf
        Section 2.2, 2.3
        '''
        self.logger.debug('start sampling representer points')
        self.representer_points, self.representer_points_log = self._sample_representer_points()
        '''
        if not np.all(self.representer_points_log < 0):
            raise ValueError('log representer_points should be all less than zero')
        '''
        #self.logger.debug('check the representer points sampled by MC, self.representer_points={}'.format(self.representer_points))
        mu, _ = self.gps.predict(self.representer_points)
        mu = np.ndarray.flatten(mu)
        var = self.gps.predict_covariance(self.representer_points)

        # Computes the probability of a given point to be the minimum
        self.logP, self.dlogPdMu, self.dlogPdSigma, self.dlogPdMudMu = epmgp.joint_min(mu, var, with_derivatives=True)
        '''
        if not np.all(self.logP < 0):
            raise ValueError('log P estimated by epmgp should be all less than zero')
        '''
        self.logP = self.logP[:, np.newaxis]

        # Calculate the entropy of the distribution over the minimum given the current model
        self.p_min_entropy = np.sum(np.multiply(np.exp(self.logP), np.add(self.logP, self.representer_points_log)),
                                axis=0)
        if self.p_min_entropy > 0:
            self.logger.debug('self.p_min_entropy={}'.format(self.p_min_entropy))
            #raise ValueError('negative entropy should always be less than zero')

        return self.logP

    def _sample_representer_points(self):
        self.logger.debug('start sampler get_samples')
        repr_points, repr_points_log = self.sampler.get_samples(self.num_representer_points, self.proposal_func)
        if np.any(np.isnan(repr_points_log)) or np.any(np.isposinf(repr_points_log)):
            raise RuntimeError("Sampler generated representer points with invalid log values: {}".format(repr_points_log))

        # Removing representer points that have 0 probability of being the minimum
        self.logger.debug("before remove, representer shape = {}".format(repr_points.shape))
        idx_to_remove = np.where(np.isneginf(repr_points_log))[0]
        if len(idx_to_remove) > 0:
            #raise ValueError('should no points of 0 probability')
            idx = list(set(range(self.num_representer_points)) - set(idx_to_remove))
            repr_points = repr_points[idx, :]
            repr_points_log = repr_points_log[idx]
        self.logger.debug("after remove, representer shape = {}".format(repr_points.shape))
        if repr_points.shape[0] < 1:
            raise ValueError("should increase num of burn_in_steps and n_step")
        return repr_points, repr_points_log

    def evaluate(self, x):
        '''
        the entropy reduction of the optimal location of x for the objective function by observing x
        '''
        if x.shape[0] > 1:
            #self.logger.debug('in es.evaluate, x.shape={}'.format(x.shape))
            results = np.zeros([x.shape[0], 1])
            for j in range(x.shape[0]):
                results[j] = self.evaluate(x[j, None, :])
            return results

        # Number of representer points locations
        N = self.logP.size

        # Evaluate innovations, i.e how much does mean and variance at the
        # representer points change if we would evaluate x
        dMdx, dVdx = self._innovations(x)
        dVdx = dVdx[np.triu(np.ones((N, N))).T.astype(bool), np.newaxis]
        dMdx_squared = dMdx.dot(dMdx.T)
        trace_term = np.sum(np.sum(np.multiply(self.dlogPdMudMu, \
                                               np.reshape(dMdx_squared, (1, dMdx_squared.shape[0], dMdx_squared.shape[1]))),2),1)[:, np.newaxis]

        # Determinnistic part of change:
        deterministic_change = self.dlogPdSigma.dot(dVdx) + 0.5 * trace_term

        # Stochastic part of change:
        w = scipy.stats.norm.ppf(np.linspace(1. / (self.num_samples + 1),
                                                  1 - 1. / (self.num_samples + 1),
                                                  self.num_samples))[np.newaxis, :]
        stochastic_change = (self.dlogPdMu.dot(dMdx)).dot(w)

        # Update p_xopt
        predicted_logP = np.add(self.logP + deterministic_change, stochastic_change)
        max_predicted_logP = np.amax(predicted_logP, axis=0)

        # Normalize predictions
        max_diff = max_predicted_logP + np.log(np.sum(np.exp(predicted_logP - max_predicted_logP), axis=0))
        lselP = max_predicted_logP if np.any(np.isinf(max_diff)) else max_diff
        predicted_logP = np.subtract(predicted_logP, lselP)

        # Maximize the information gain
        H_p = np.sum(np.multiply(np.exp(predicted_logP), np.add(predicted_logP, self.representer_points_log)), axis=0)
        new_entropy = np.mean(H_p)
        entropy_change = new_entropy - self.p_min_entropy
        #self.logger.debug('x={}, entropy_change={}, negative new_entropy={}, p_min_entropy={}'.format(x, entropy_change, new_entropy, self.p_min_entropy))
        return np.array([[entropy_change]])

    def _innovations(self, x):
        """
        expected change in mean and variance at the representer points  x
        http://jmlr.csail.mit.edu/papers/volume13/hennig12a/hennig12a.pdf
        Section 2.4
        """
        # Get the standard deviation at x without noise
        var_x = self.gps.predict_covariance(x, with_noise=False)
        std_x = np.sqrt(var_x)

        # Compute the variance between the test point x and the representer points
        sigma_x_rep = self.gps.get_covariance_between_points(self.representer_points, x)
        dm_rep = sigma_x_rep / std_x

        # Compute the deterministic innovation for the variance
        dv_rep = -dm_rep.dot(dm_rep.T)

        return dm_rep, dv_rep

