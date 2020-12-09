from copy import deepcopy
import GPy
import numpy as np

from design_optimize_alg.surrogates.models.gpy_model_wrappers import GPyModelWrapper


class SinglegpModel(GPyModelWrapper):
    def __init__(self, X_init, Y_init, config):
        self.config = config
        self._X = deepcopy(X_init)
        self._Y = Y_init
        kernel = GPy.kern.RBF(input_dim=X_init.shape[1], ARD=True)
        kernel += GPy.kern.White(input_dim=X_init.shape[1])

        gp = GPy.models.GPRegression(self._X, self._Y, kernel=kernel)
        gp['.*Gaussian'] = 10 **(-6)
        gp['.*Gaussian'].fix()
        gp['.*white'] = self.config["whitenoise"]
        if 'lsu' in self.config and self.config['lsu'] > 0:
            gp['.*lengthscale'].constrain_bounded(0, self.config['lsu'])
        super(SinglegpModel, self).__init__(gpy_model=gp, n_restarts=self.config["n_restarts"])

    def add_data(self, X, Y):
        self._X = np.concatenate((self._X, X))
        self._Y = np.concatenate((self._Y, Y))
        self.model.set_XY(self.X, self.Y)

    @property
    def X(self):
        X = deepcopy(self._X)
        return X

    @property
    def Y(self):
        return self._Y
