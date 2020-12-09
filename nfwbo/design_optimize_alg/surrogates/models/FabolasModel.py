'''
https://github.com/EmuKit/emukit/blob/master/emukit/examples/fabolas/fabolas_model.py
'''

from copy import deepcopy
from typing import Tuple

import GPy
from paramz import ObsAr
import numpy as np

from design_optimize_alg.surrogates.models.gpy_model_wrappers import GPyModelWrapper
from design_optimize_alg.surrogates.kernels.FabolasKernel import FabolasKernel


def linear(s):
    return s


def quad(s):
    return (1 - s) ** 2


def transform(s, s_min, s_max, mode='ori'):
    if mode == "log":
        s_transform = (np.log2(s) - np.log2(s_min + 1e-10)) / (np.log2(s_max) - np.log2(s_min + 1e-10))
    elif mode == "ori":
        s_transform = s
    else:
        raise NotImplementedError
    return s_transform


def retransform(s_transform, s_min, s_max, mode="ori"):
    if mode == "log":
        s = np.rint(2 ** (s_transform * (np.log2(s_max) - np.log2(s_min + 1e-10)) + np.log2(s_min + 1e-10)))
    elif mode == "ori":
        s = s_transform
    else:
        raise NotImplementedError
    return s


class FabolasModel(GPyModelWrapper):

    def __init__(self, X_init, Y_init, z_min, z_max, config, basis_func=linear, noise=1e-6):
        """
        Fabolas Gaussian processes model which models the validation error / cost of
        hyperparameter configurations across training dataset subsets.

        :param X_init: training data points
        :param Y_init: training targets
        :param zdim: dimension of fidelity space
        :param basis_func: basis function which describes the change in performance across dataset subsets
        :param noise: observation noise added to the diagonal of the kernel matrix
        """
        s_min = z_min
        s_max = z_max
        self.config = config
        self.transform_mode = self.config["transform_mode"]
        if s_min.shape[0] != s_max.shape[0]:
            raise ValueError
        self.zdim = s_min.shape[0]
        self.noise = noise
        self.s_min = s_min
        self.s_max = s_max
        self._X = deepcopy(X_init)
        self._X[:, -self.zdim:] = transform(self._X[:, -self.zdim:], self.s_min, self.s_max, self.transform_mode)
        self._Y = Y_init
        self.basis_func = basis_func
        kernel = self._configKernel()

        gp = GPy.models.GPRegression(self._X, self._Y, kernel=kernel)
        gp['.*Gaussian'] = 10 **(-6)
        gp['.*Gaussian'].fix()
        gp['.*white'] = self.config["whitenoise"]
        #gp['.*white'].fix()

        #gp.kern.set_prior(GPy.priors.Uniform(0, 5))
        gp.likelihood.constrain_positive()
        super(FabolasModel, self).__init__(gpy_model=gp, n_restarts=self.config['n_restarts'])

    def _configKernel(self):
        config = self.config['kernels']
        if config['xkernel'] == 'M52':
            kernel = GPy.kern.Matern52(input_dim=self._X.shape[1] - self.zdim, active_dims=[i for i in range(self._X.shape[1] - self.zdim)],
                                   variance=np.var(self._Y), ARD=True)
        elif config['xkernel'] == 'RBF':
            kernel = GPy.kern.RBF(input_dim=self._X.shape[1] - self.zdim, active_dims=[i for i in range(self._X.shape[1] - self.zdim)], ARD=True)
        elif config['xkernel'] == 'nofactor':
            kernel = GPy.kern.RBF(input_dim=self._X.shape[1], ARD=True)
        else:
            raise NotImplementedError

        if config['zkernel'] == 'fabolas':
            kernel *= FabolasKernel(input_dim=self.zdim, active_dims=[i for i in range(self._X.shape[1] - self.zdim, self._X.shape[1])], basis_func=self.basis_func)
        elif config['zkernel'] == 'M52':
            kernel *= (GPy.kern.Matern52(input_dim=self.zdim, active_dims=[i for i in range(self._X.shape[1] - self.zdim, self._X.shape[1])]))
        elif config['zkernel'] == 'RBF':
            kernel *= (GPy.kern.RBF(input_dim=self.zdim, active_dims=[i for i in range(self._X.shape[1] - self.zdim, self._X.shape[1])]))
        else:
            raise NotImplementedError

        kernel += GPy.kern.White(input_dim=self.zdim, active_dims=list(range(self._X.shape[1] - self.zdim, self._X.shape[1])), variance=1e-6)
        return kernel

    def predict(self, X):
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get predictions
        :return: (mean, variance) Arrays of size n_points x 1 of the predictive distribution at each input location
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max, self.transform_mode)
        return super(FabolasModel, self).predict(X_)

    def add_data(self, X, Y):
        """
        Add training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self._X = np.concatenate((self._X, X), axis=0)
        self._X[:, -1] = transform(self._X[:, -1], self.s_min, self.s_max, self.transform_mode)
        self._Y = np.concatenate((self._Y, Y), axis=0)
        try:
            self.model.set_XY(self._X, self._Y)
        except:
            kernel = self._configKernel()
            self.model = GPy.models.GPRegression(self._X, self._Y, kernel=kernel)
            self.model['.*Gaussian'] = 10 ** (-6)
            self.model['.*Gaussian'].fix()
            self.model['.*white'] = self.config["whitenoise"]
            self.model.likelihood.constrain_positive()

    def set_data(self, X, Y):
        """
        Sets training data in model

        :param X: New training features
        :param Y: New training outputs
        """
        self._X = deepcopy(X)
        self._X[:, -1] = transform(self._X[:, -1], self.s_min, self.s_max, self.transform_mode)
        self._Y = Y
        self.model.set_XY(self._X, self.Y)
        '''
        try:
            self.model.set_XY(self._X, self.Y)
        except:
            kernel = GPy.kern.Matern52(input_dim=self._X.shape[1] - 1,
                                       active_dims=[i for i in range(self._X.shape[1] - 1)],
                                       variance=np.var(self.Y), ARD=True)
            kernel *= FabolasKernel(input_dim=1, active_dims=[self._X.shape[1] - 1], basis_func=self.basis_func)
            kernel *= GPy.kern.OU(input_dim=1, active_dims=[self._X.shape[1] - 1])

            self.model = GPy.models.GPRegression(self._X, self.Y, kernel=kernel, noise_var=self.noise)
            self.model.likelihood.constrain_positive()
        '''
    def get_f_minimum(self):
        """
        Predicts for all observed data points the validation error on the full dataset and returns
        the smallest mean prediciton

        :return: Array of size 1 x 1
        """
        proj_X = deepcopy(self._X)
        proj_X[:, -1] = np.ones(proj_X.shape[0]) * self.s_max
        mean_highest_dataset = self.model.predict(proj_X)

        return np.min(mean_highest_dataset, axis=0)

    @property
    def X(self):
        X = deepcopy(self._X)
        X[:, -1] = retransform(X[:, -1], self.s_min, self.s_max, self.transform_mode)
        return X

    @property
    def Y(self):
        return self._Y

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        :param X: (n_points x n_dimensions) array containing locations at which to get gradient of the predictions
        :return: (mean gradient, variance gradient) n_points x n_dimensions arrays of the gradients of the predictive
                 distribution at each input location
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max, self.transform_mode)

        return super(FabolasModel, self).get_prediction_gradients(X_)

    def predict_covariance(self, X: np.ndarray, with_noise: bool = True) -> np.ndarray:
        """
        Calculates posterior covariance between points in X
        :param X: Array of size n_points x n_dimensions containing input locations to compute posterior covariance at
        :param with_noise: Whether to include likelihood noise in the covariance matrix
        :return: Posterior covariance matrix of size n_points x n_points
        """
        X_ = deepcopy(X)
        X_[:, -1] = transform(X_[:, -1], self.s_min, self.s_max, self.transform_mode)

        return super(FabolasModel, self).predict_covariance(X_, with_noise)

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Calculate posterior covariance between two points
        :param X1: An array of shape 1 x n_dimensions that contains a data single point. It is the first argument of the
                   posterior covariance function
        :param X2: An array of shape n_points x n_dimensions that may contain multiple data points. This is the second
                   argument to the posterior covariance function.
        :return: An array of shape n_points x 1 of posterior covariances between X1 and X2
        """
        X_1 = deepcopy(X1)
        X_1[:, -1] = transform(X_1[:, -1], self.s_min, self.s_max, self.transform_mode)
        X_2 = deepcopy(X2)
        X_2[:, -1] = transform(X_2[:, -1], self.s_min, self.s_max, self.transform_mode)

        return super(FabolasModel, self).get_covariance_between_points(X1, X2)
