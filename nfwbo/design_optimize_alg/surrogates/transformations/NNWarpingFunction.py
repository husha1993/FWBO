import os
import time

import numpy as np
from GPy.util.input_warping_functions import InputWarpingFunction
from GPy.core.parameterization import Param
import torch
import torch.nn as nn


def mlp(input_dim, mlp_dims, last_act=False):
    layers = []
    mlp_dims = [input_dim] + mlp_dims
    for i in range(len(mlp_dims) - 1):
        layers.append(nn.Linear(mlp_dims[i], mlp_dims[i + 1]).double())
        if i != len(mlp_dims) - 2 or last_act:
            layers.append(nn.Sigmoid().double())

    net = nn.Sequential(*layers)
    return net


class NNwarpingFunction(InputWarpingFunction):
    def __init__(self, warping_indices, hidden_dims, out_dim, warped_indices, name):
        super(NNwarpingFunction, self).__init__(name='nn_warping_'+name)
        self.warping_indices = warping_indices
        self.warped_indices = warped_indices
        self.nnwarping = NNwarping(len(warping_indices), hidden_dims, out_dim)
        self.params_name = list(self.nnwarping.state_dict().keys())
        self.params_value = [_.numpy() for _ in list(self.nnwarping.state_dict().values())]
        self.params = [Param(self.params_name[_], self.params_value[_]) for _ in range(len(self.params_value))]
        for param in self.params:
            self.link_parameter(param)

        # training statistics
        self.params_updated_num = 0

    def f(self, X, test=False):
        # update the torch nets
        state_dict = dict()
        for param in self.params:
            state_dict[param.name] = torch.tensor(np.array(param), requires_grad=True, dtype=torch.float64)
        self.nnwarping.load_state_dict(state_dict)

        self.params_updated_num += 1
        if test:
            state_dict = dict()

        if test:
            with torch.no_grad():
                X_warped_th = self.nnwarping(X[:, self.warping_indices])
        else:
            X_warped_th= self.nnwarping(X[:, self.warping_indices])
        return X_warped_th

    def fgrad_X(self, X):
        '''
        Compute the gradient of warping function with respect to X
        '''
        X_warped = X.copy()
        pass

    def update_grads(self, X_warped_th, dL_dWX):
        '''
        To update the gradients of marginal log likelihood with respect to the parameters of warping function
        input: WX = f(X|M)
        Output:dL_dM = dL_dWX * dWX_dM, i.e.,
        dL_dM = (dL_dWX)[:, w_idx[0]] * (dWX_dM)[:, w_idx[0]] + (dL_dWX)[:, w_idx[1]] * (dWX_dM)[:, w_idx[1]] +...+ \
        (dL_dWX)[:, w_idx[-1]] * (dWX_dM)[:, w_idx[-1]]
        '''
        if X_warped_th.shape[0] != dL_dWX.shape[0]:
            raise ValueError('dimension mismatch')
        dWX_dM = self.nnwarping.get_params_gradient(X_warped_th, self.warped_indices)
        dL_dM = dict()
        for name, g in dWX_dM.items():
            dL_dM[name] = np.zeros(self.nnwarping.state_dict()[name].shape)
            for n_idx in range(X_warped_th.shape[0]):
                for w_idx in range(X_warped_th.shape[1]):
                    dL_dM[name] += dL_dWX[n_idx, self.warped_indices[w_idx]] * dWX_dM[name][n_idx][w_idx].numpy()
        # update gradients for GPy
        for param in self.params:
            name = param.name
            param.gradient = dL_dM[name]


class NNwarping(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        self.nets = mlp(input_dim, hidden_dims + [output_dim], last_act=True)

    def forward(self, X):
        X = torch.tensor(X, dtype=torch.float64)
        X_warped = self.nets(X)
        return X_warped

    def get_params_gradient(self, X_W, warped_indices):
        if len(warped_indices) != X_W.shape[1]:
            raise ValueError('mismatch between x_w and warped_indices')
        N = X_W.shape[0]
        params_gradient = {}
        for name, _ in self.named_parameters():
            params_gradient[name] = [[] for _ in range(N)]
        for n_idx in range(N):
            for w_idx in range(X_W.shape[1]):
                for name, param in self.named_parameters():
                    grad_idx = torch.zeros(X_W.shape, dtype=torch.float64)
                    grad_idx[n_idx, w_idx] = 1
                    g = torch.autograd.grad(X_W, param, grad_outputs=grad_idx, retain_graph=True)
                    params_gradient[name][n_idx].append(g[0])

        return params_gradient


