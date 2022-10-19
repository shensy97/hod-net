'''
conv.py
====================

Implement convolution layers.
Modified from https://github.com/leeroee/NN-by-Numpy/blob/master/package/layers/conv.py.
'''

import cupy as np #import numpy as np
from layers.layer import Parameter, Module


def split_by_strides(X, kh, kw, s=1):
    N, H, W, C = X.shape
    oh = (H - kh) // s + 1
    ow = (W - kw) // s + 1
    shape = (N, oh, ow, kh, kw, C)
    strides = (X.strides[0], X.strides[1] * s, X.strides[2] * s, *X.strides[1:])
    return np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)


class Conv(Module):

    def __init__(self, shape, method='VALID', stride=1, require_grad=True, bias=True, **kwargs):
        '''
        shape = (out_channel, kernel_size, kernel_size, in_channel)
        input shape : N, iH, iW, iC
        '''
        super().__init__()
        k = 1 / (shape[1] * shape[2] * shape[3])
        self.weights = np.random.uniform(-np.sqrt(k), np.sqrt(k), shape, dtype=np.float32)
        self.use_bias = bias
        if self.use_bias:
            self.bias = np.zeros(shape[0], dtype=np.float32)
        self.method = method
        self.s = stride
        self.kn = shape[0]
        self.ksize = shape[1]
        self.require_grad = require_grad
        self.reset_param()

    def padding(self, x, forward=True):
        p = self.ksize // 2 if self.method == 'SAME' else self.ksize - 1
        if forward:
            return x if self.method == 'VALID' else np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')
        else:
            return np.pad(x, ((0, 0), (p, p), (p, p), (0, 0)), 'constant')

    def forward(self, x):
        '''
        x.shape = N, iH, iW, iC
        x_split.shape = N, oH, oW, kh, kw, iC
        return.shape = N, oH, oW, oC
        '''
        x = self.padding(x)
        if self.s > 1:
            self.oh = x.shape[1] - self.ksize + 1
            self.ow = x.shape[2] - self.ksize + 1
        self.x_split = split_by_strides(x, self.ksize, self.ksize, self.s)
        a = np.tensordot(self.x_split, self.weights.tensor, axes=[(3, 4, 5), (1, 2, 3)])
        if self.use_bias:
            a = a + self.bias.tensor
        return a

    def cforward(self, x):
        cx = self.padding(x)
        if self.s > 1:
            self.oh = x.shape[1] - self.ksize + 1
            self.ow = x.shape[2] - self.ksize + 1
        self.cx_split = split_by_strides(cx, self.ksize, self.ksize, self.s)
        a = np.tensordot(self.cx_split, self.weights.ctensor, axes=[(3, 4, 5), (1, 2, 3)])
        if self.use_bias:
            a = a + self.bias.ctensor
        return a

    def backward(self, eta):
        '''
        eta.shape = N, oH, oW, oC
        W_rot180.shape = iC, kh, kw, oC
        eta_split.shape = N, iH, iW, kh, kw, oC
        return.shape = N, iH, iW, iC
        '''
        if self.require_grad:

            batch_size = eta.shape[0]
            self.weights.gradient = np.tensordot(eta, self.x_split, [(0, 1, 2), (0, 1, 2)])

            if self.use_bias:
                self.bias.gradient = np.reshape(eta, [eta.shape[0], -1, self.kn]).sum(axis=(0, 1))

        if self.s > 1:
            temp = np.zeros((eta.shape[0], self.oh, self.ow, eta.shape[3]), dtype=np.float32)
            temp[:, ::self.s, ::self.s, :] = eta
            eta = temp
        eta_pad = self.padding(eta, False)
        W_rot180 = self.weights.tensor[:, ::-1, ::-1, :].transpose(3, 1, 2, 0)
        eta_split = split_by_strides(eta_pad, self.ksize, self.ksize, self.s)
        return np.tensordot(eta_split, W_rot180, axes=[(3, 4, 5), (1, 2, 3)])


    def cbackward(self, eta):
        if self.require_grad:
            batch_size = eta.shape[0]
            self.weights.cgradient = np.tensordot(eta, self.cx_split, [(0, 1, 2), (0, 1, 2)])
            if self.use_bias:
                self.bias.cgradient = np.reshape(eta, [eta.shape[0], -1, self.kn]).sum(axis=(0, 1))

        if self.s > 1:
            temp = np.zeros((eta.shape[0], self.oh, self.ow, eta.shape[3]), dtype=np.float32)
            temp[:, ::self.s, ::self.s, :] = eta
            eta = temp
        eta_pad = self.padding(eta, False)
        W_rot180 = self.weights.ctensor[:, ::-1, ::-1, :].transpose(3, 1, 2, 0)
        eta_split = split_by_strides(eta_pad, self.ksize, self.ksize, self.s)
        return np.tensordot(eta_split, W_rot180, axes=[(3, 4, 5), (1, 2, 3)])


    def reset_param(self):
        self.weights = self.register_parameter(self.weights)
        if self.use_bias:
            self.bias = self.register_parameter(self.bias)


