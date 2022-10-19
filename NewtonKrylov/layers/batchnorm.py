'''
batchnorm.py
===============

Batchnorm1d and Batchnorm2d.
'''

import cupy as np #import numpy as np
from layers.layer import Parameter, Module


class BatchNormalization(Module):
    def __init__(self, shape, affine=True, is_test=False):
        super().__init__()
        self.gamma = np.random.uniform(0.9, 1.1, shape)
        self.gamma = self.register_parameter(self.gamma)
        self.beta = np.random.uniform(-0.1, 0.1, shape)
        self.beta = self.register_parameter(self.beta)
        self.eps = 1e-5
        self.affine = affine
        self.is_test = is_test

        self.coe = 0.0
        self.overall_ave = np.ones(shape).astype(np.float32)
        self.overall_var = np.zeros(shape).astype(np.float32)


    def complex_var(self, x, axis=0):
        mean_x = np.mean(x, axis=axis)
        center_x = (x - mean_x)
        return np.mean(center_x * center_x, axis=axis)

    def forward_internal(self, sample_diff, sample_std):
        self.normalized = sample_diff / sample_std
        self.gamma_s = self.gamma.tensor / sample_std
        return self.gamma.tensor * self.normalized + self.beta.tensor

    def cforward_internal(self, sample_diff, sample_std):
        self.cnormalized = sample_diff / sample_std
        self.cgamma_s = self.gamma.ctensor / sample_std
        return self.gamma.ctensor * self.cnormalized + self.beta.ctensor

    def forward(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        if self.is_test:
            sample_ave = self.overall_ave
            sample_std = np.sqrt(self.overall_var)
        else:
            sample_ave = x_flat.mean(axis=0)
            sample_var = self.complex_var(x_flat, axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            self.overall_ave = (1 - self.coe) * self.overall_ave + self.coe * sample_ave
            self.overall_var = (1 - self.coe) * self.overall_var + self.coe * sample_var
        output = (x_flat - sample_ave) / sample_std if not self.affine else self.forward_internal(x_flat - sample_ave, sample_std)
        return output.reshape(x.shape)

    def backward(self, eta):
        eta_flat = eta.reshape(-1, eta.shape[-1])
        if not self.affine: return
        self.beta.gradient = eta_flat.sum(axis=0)
        self.gamma.gradient = (eta_flat * self.normalized).sum(axis=0)
        eta_out = self.gamma_s * (eta_flat - self.normalized * self.gamma.gradient / eta_flat.shape[0] - self.beta.gradient / eta_flat.shape[0])
        return eta_out.reshape(eta.shape)

    def cforward(self, x):
        x_flat = x.reshape(-1, x.shape[-1])
        if self.is_test:
            sample_ave = self.overall_ave
            sample_std = np.sqrt(self.overall_var)
        else:
            sample_ave = x_flat.mean(axis=0)
            sample_var = self.complex_var(x_flat, axis=0)
            sample_std = np.sqrt(sample_var + self.eps)
            # self.overall_ave = (1 - self.coe) * self.overall_ave + self.coe * sample_ave
            # self.overall_var = (1 - self.coe) * self.overall_var + self.coe * sample_var
        output = (x_flat - sample_ave) / sample_std if not self.affine else self.cforward_internal(x_flat - sample_ave, sample_std)
        return output.reshape(x.shape)

    def cbackward(self, eta):
        eta_flat = eta.reshape(-1, eta.shape[-1])
        if not self.affine: return
        self.beta.cgradient = eta_flat.sum(axis=0)
        self.gamma.cgradient = (eta_flat * self.cnormalized).sum(axis=0)
        eta_out =  self.cgamma_s * (eta_flat - self.cnormalized * self.gamma.cgradient / eta_flat.shape[0] - self.beta.cgradient / eta_flat.shape[0])
        return eta_out.reshape(eta.shape)