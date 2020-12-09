import numpy as np


class GpUCB(object):
    '''
    https://arxiv.org/abs/0912.3995
    '''
    def __init__(self, gps, logger, config=None):
        self.gps = gps
        self.config = config
        self.xdim = config['xdim']

    def evaluate(self, x, iterations, alpha=-1, v=.1, delta=.1):
        x = np.reshape(x, (-1, self.xdim))
        mean, var = self.gps.predict(x)
        std = np.sqrt(var)
        if alpha is -1:
            alpha = np.sqrt(v * (2 * np.log((iterations ** ((self.xdim / 2) + 2))
                                            * (np.pi ** 2) / (3 * delta))))
        return mean + (alpha * std)

    def update(self):
        pass

