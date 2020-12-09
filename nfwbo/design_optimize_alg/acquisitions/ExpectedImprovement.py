import numpy as np
import scipy.stats


class ExpectedImprovement(object):
    def __init__(self, gps, logger=None, jitter=0, config=None):
        self.gps = gps
        self.logger = logger
        self.jitter = jitter
        self.config = config

    def evaluate(self, x):
        mean, variance = self.gps.predict(x)
        #self.logger.debug('check the model of proposal function={}'.format(self.gps.model.X.shape))
        std = np.sqrt(variance)
        mean += self.jitter

        #y_opt = np.max(self.gps.Y, axis=0)
        #u = (mean - y_opt) / std

        y_opt = np.min(self.gps.Y, axis=0)
        u = (y_opt - mean) / std

        pdf = scipy.stats.norm.pdf(u)
        cdf = scipy.stats.norm.cdf(u)
        improvement = std * (u * cdf + pdf)

        return improvement

