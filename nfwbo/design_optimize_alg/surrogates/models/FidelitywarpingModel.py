import sys
from copy import deepcopy

import numpy as np
import GPy
from GPy.core import GP
from GPy import kern, likelihoods
from paramz import ObsAr
import torch

from design_optimize_alg.surrogates.models.gpy_model_wrappers import GPyModelWrapper
from design_optimize_alg.surrogates.transformations.NNWarpingFunction import NNwarpingFunction


class FidelitywarpingModel(GP):
    def __init__(self, X, Y, kernel, warping_function_name=None, warping_indices=None, warping_hid_dims = None, warping_out_dim = None, zdim = None, \
                 X_indices =None, X_warped_indices = None,  normalizer=False, Xmin=None, Xmax=None, epsilon=None):
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        self.Xdim = X.shape[1]
        self.X_untransformed = X.copy()
        self.kernel = kernel

        self.warping_function_name = warping_function_name
        self.warping_indices = warping_indices
        self.warping_hid_dims = warping_hid_dims
        self.warping_out_dim = warping_out_dim
        self.warping_functions = dict()
        self.X_indices = X_indices
        self.X_warped_indices = X_warped_indices
        self.update_warping_functions = True # set as False if random embedding

        if warping_function_name is None:
           raise NotImplementedError
        else:
            for k, v in warping_indices.items():
                if len(v) > 0:
                    self.warping_functions[k] = NNwarpingFunction(v, self.warping_hid_dims[k], self.warping_out_dim[k], self.X_warped_indices[k], k)
                else:
                    self.warping_functions[k] = None

        self.X_warped_th = self.transform_data(self.X_untransformed)
        if self.warping_function_name == 'nn':
            X_np = []
            for k, v in self.X_warped_th.items():
                if type(v) == torch.Tensor:
                    X_np.append(v.detach().numpy())
                else:
                    X_np.append(v)
            self.X_warped = np.concatenate(X_np, axis=1)
        likelihood = likelihoods.Gaussian()
        super(FidelitywarpingModel, self).__init__(self.X_warped, Y, likelihood=likelihood, kernel=kernel, normalizer=normalizer, name='FidelitywarpingModel')

        # Add the parameters in the warping function to the model parameters hierarchy
        for k, v in self.warping_functions.items():
            if v is not None:
                self.link_parameter(v)

    def parameters_changed(self):
        """
        Update the gradients of parameters for warping function
        This method is called when having new values of parameters for warping function, kernels
        and other parameters in a normal GP
        """
        self.X_th = self.transform_data(self.X_untransformed)
        if self.warping_function_name == 'nn':
            X_np = []
            for k, v in self.X_th.items():
                if type(v) == torch.Tensor:
                    X_np.append(v.detach().numpy())
                else:
                    X_np.append(v)
        self.X = np.concatenate(X_np, axis=1)
        super(FidelitywarpingModel, self).parameters_changed()
        # the gradient of log likelihood w.r.t. input AFTER warping is a product of dL_dK and dK_dX
        dL_dWX = self.kern.gradients_X(self.grad_dict['dL_dK'], self.X)

        for k, v in self.warping_functions.items():
            if v is not None:
                if self.update_warping_functions:
                    self.warping_functions[k].unfix()
                    self.warping_functions[k].update_grads(self.X_th[k], dL_dWX)

    def _fix_warping_param(self):
        self.update_warping_functions = False
        for k, v in self.warping_functions.items():
            if v is not None:
                self.warping_functions[k].fix()

    def _unfix_warping_param(self):
        self.update_warping_functions = True

    def set_XY(self, X=None, Y=None):
        self.update_model(False)
        if Y is not None:
            if self.normalizer is not None:
                self.normalizer.scale_by(Y)
                self.Y_normalized = ObsAr(self.normalizer.normalize(Y))
                self.Y = Y
            else:
                self.Y = ObsAr(Y)
                self.Y_normalized = self.Y
        if X is not None:
            self.X_untransformed = ObsAr(X)
        self.update_model(True)

    def transform_data(self, X, test_data=False):
        X_w_th = dict()
        for k, v in self.warping_functions.items():
            if v is not None:
                X_w_th[k] = self.warping_functions[k].f(X, test_data)
            else:
                X_w_th[k] = X[:, self.X_indices[k]]
        return X_w_th

    def log_likelihood(self):
        return GP.log_likelihood(self)

    def predict(self, Xnew, full_cov=False, include_likelihood=True):
        if Xnew.shape[1] != self.Xdim:
            raise ValueError
        Xnew_warped_th = self.transform_data(Xnew, test_data=True)
        if self.warping_function_name == 'nn':
            X_np = []
            for k, v in Xnew_warped_th.items():
                if type(v) == torch.Tensor:
                    X_np.append(v.detach().numpy())
                else:
                    X_np.append(v)
            Xnew_warped = np.concatenate(X_np, axis=1)
        mean, var = super(FidelitywarpingModel, self).predict(Xnew_warped, kern=self.kernel, full_cov=full_cov, include_likelihood=include_likelihood)
        return mean, var

    def predict_Xwarped(self, Xnew):
        Xnew_warped_th = self.transform_data(Xnew, test_data=True)
        if self.warping_function_name == 'nn':
            X_np = []
            for k, v in Xnew_warped_th.items():
                if type(v) == torch.Tensor:
                    X_np.append(v.detach().numpy())
                else:
                    X_np.append(v)
            Xnew_warped = np.concatenate(X_np, axis=1)
        return Xnew_warped

    def posterior_covariance_between_points(self, X1, X2):
        X1_warped_th = self.transform_data(X1, test_data=True)
        X2_warped_th = self.transform_data(X2, test_data=True)
        if self.warping_function_name == 'nn':
            X1_np = []
            for k, v in X1_warped_th.items():
                if type(v) == torch.Tensor:
                    X1_np.append(v.detach().numpy())
                else:
                    X1_np.append(v)
            X1_warped = np.concatenate(X1_np, axis=1)

            X2_np = []
            for k, v in X2_warped_th.items():
                if type(v) == torch.Tensor:
                    X2_np.append(v.detach().numpy())
                else:
                    X2_np.append(v)
            X2_warped = np.concatenate(X2_np, axis=1)
        return super(FidelitywarpingModel, self).posterior_covariance_between_points(X1_warped, X2_warped)


class FidelitywarpingModelWrapper(GPyModelWrapper):

    def __init__(self, X_init, Y_init, z_min, z_max, config):
        self.config = config
        self.zdim = z_min.shape[0]
        self.z_min = z_min
        self.z_max = z_max
        self._X = deepcopy(X_init)
        self._Y = Y_init

        gp = self.config_gp()
        super(FidelitywarpingModelWrapper, self).__init__(gpy_model=gp, n_restarts=self.config['n_restarts'])

    def config_gp(self):
        self.w_dim = self.config['w_dim']
        kernel = GPy.kern.RBF(self.w_dim, ARD=True)
        kernel += GPy.kern.White(input_dim=self.w_dim)

        gp = FidelitywarpingModel(self._X, self._Y, kernel, warping_function_name=self.config['warping_function_name'], warping_indices=self.config['warping_indices'],\
        warping_hid_dims=self.config['warping_hid_dims'], warping_out_dim=self.config['warping_out_dim'], zdim=self.zdim, X_indices=self.config['X_indices'], \
        X_warped_indices=self.config['X_warped_indices'])
        gp['.*Gaussian_noise'] = 10 **(-6)
        gp['.*Gaussian_noise'].fix()
        gp['.*white'] = self.config["whitenoise"]
        return gp

    def predict(self, X):
        X_ = deepcopy(X)
        return super(FidelitywarpingModelWrapper, self).predict(X_)

    def predict_Xwarped(self, X):
        X_ = deepcopy(X)
        return super(FidelitywarpingModelWrapper, self).predict_Xwarped(X_)

    def add_data(self, X, Y):
        self._X = np.concatenate((self._X, X))
        self._Y = np.concatenate((self._Y, Y))
        self.model.set_XY(self.X, self.Y)

    def set_data(self, X, Y):
        self._X = deepcopy(X)
        self._Y = Y
        self.model.set_XY(self._X, self._Y)

    @property
    def X(self):
        X = deepcopy(self._X)
        return X

    @property
    def Y(self):
        return self._Y


