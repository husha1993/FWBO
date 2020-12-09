from typing import Tuple

import numpy as np
import GPy


class GPyModelWrapper:
    def __init__(self, gpy_model, n_restarts=3):
        self.model = gpy_model
        self.n_restarts = n_restarts

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        m, var = self.model.predict(X)
        var = np.clip(var, 0., np.inf)
        return m, var

    def predict_with_full_covariance(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.model.predict(X, full_cov=True)

    def get_prediction_gradients(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        d_mean_dx, d_variance_dx = self.model.predictive_gradients(X)
        return d_mean_dx[:, :, 0], d_variance_dx

    def set_data(self, X: np.ndarray, Y: np.ndarray) -> None:
        self.model.set_XY(X, Y)

    def update(self):
        self.model.optimize_restarts(self.n_restarts, robust=True)

    def calculate_variance_reduction(self, x_train_new: np.ndarray, x_test: np.ndarray) -> np.ndarray:
        covariance = self.model.posterior_covariance_between_points(x_train_new, x_test)
        variance_prediction = self.model.predict(x_train_new)[1]
        return covariance**2 / variance_prediction

    def predict_covariance(self, X: np.ndarray, with_noise: bool=True) -> np.ndarray:
        _, v = self.model.predict(X, full_cov=True, include_likelihood=with_noise)
        v = np.clip(v, 1e-10, np.inf)

        return v

    def get_covariance_between_points(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        return self.model.posterior_covariance_between_points(X1, X2)

    @property
    def X(self) -> np.ndarray:
        return self.model.X

    @property
    def Y(self) -> np.ndarray:
        return self.model.Y

    def fix_model_hyperparameters(self, sample_hyperparameters: np.ndarray) -> None:
        if self.model._fixes_ is None:
            self.model[:] = sample_hyperparameters
        else:
            self.model[self.model._fixes_] = sample_hyperparameters
        self.model._trigger_params_changed()

    def fix_warping_functions(self):
        if self.model.name == 'FidelitywarpingModel':
            self.model._fix_warping_param()

    def unfix_warping_functions(self):
        if self.model.name == "FidelitywarpingModel":
            self.model._unfix_warping_param()


