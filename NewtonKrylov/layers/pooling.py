'''
pooling.py
====================

Max-pooling and average-pooling.
'''
import cupy as np #import numpy as np
from layers.layer import Parameter, Module

class MaxPooling(Module):
    def __init__(self, size, **kwargs):
        '''
        Maxpooling layer
        '''
        super().__init__()
        self.size = size
        self.inputs = None
        self.outputs = None

    def forward(self, x):
        self.inputs = x
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        out = out.max(axis=(2, 4))
        self.mask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        self.outputs = out
        return self.outputs

    def cforward(self, inputs):
        x = inputs
        out = x.reshape(x.shape[0], x.shape[1] // self.size, self.size, x.shape[2] // self.size, self.size, x.shape[3])
        out = out.max(axis=(2, 4))
        self.cmask = out.repeat(self.size, axis=1).repeat(self.size, axis=2) != x
        return out

    def backward(self, eta):
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        eta[self.mask] = 0
        return eta

    def cbackward(self, eta):
        eta = eta.repeat(self.size, axis=1).repeat(self.size, axis=2)
        eta[self.cmask] = 0
        return eta

class MeanPooling(Module):
    def __init__(self, size, **kwargs):
        super().__init__()
        self.size = size

    def forward(self, x):
        out = x.reshape(x.shape[0], x.shape[1]//self.size, self.size, x.shape[2]//self.size, self.size, x.shape[3])
        return out.mean(axis=(2, 4))

    def cforward(self, x):
        out = x.reshape(x.shape[0], x.shape[1] // self.size, self.size, x.shape[2] // self.size, self.size, x.shape[3])
        out = out.mean(axis=(2, 4))
        return out

    def backward(self, eta):
        return (eta / self.size**2).repeat(self.size, axis=1).repeat(self.size, axis=2)

    def cbackward(self, eta):
        return (eta / self.size**2).repeat(self.size, axis=1).repeat(self.size, axis=2)
