'''
layer.py
====================

Fully connect layers
'''

import cupy as np #import numpy as np
from layers.layer import Parameter, Module

class Linear(Module):
    def __init__(self, in_features, out_features, bias=False):
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias
        self.weights = np.zeros((in_features, out_features), dtype=np.float32)
        self.bias = np.zeros(out_features, dtype=np.float32)
        self.reset_param()

    def forward(self, inputs):
        self.inputs = inputs
        self.outputs = np.matmul(inputs, self.weights.tensor)
        self.weights.ctensor = self.weights.tensor.astype(np.complex64)
        if self.use_bias:
            self.outputs += self.bias.tensor
            self.bias.ctensor = self.bias.tensor.astype(np.complex64)
        return self.outputs

    def cforward(self, inputs, sync_ctensor=False):
        '''
        inputs: complex type
        '''
        self.cinputs = inputs
        if sync_ctensor == True:
            self.weights.ctensor = self.weights.tensor.astype(np.complex64)
        coutputs = np.matmul(inputs, self.weights.ctensor)
        if self.use_bias:
            coutputs += self.bias.ctensor
        return coutputs

    def backward(self, grad_first_output):
        self.weights.gradient = self.inputs.T @ grad_first_output
        if self.use_bias:
            self.bias.gradient = grad_first_output.sum(axis=0)
        grad_first_input = np.matmul(grad_first_output, self.weights.tensor.T)

        return grad_first_input

    def cbackward(self, grad_first_output):
        self.weights.cgradient = self.cinputs.T @ grad_first_output
        if self.use_bias:
            self.bias.cgradient = grad_first_output.sum(axis=0)
        grad_first_input = np.matmul(grad_first_output, self.weights.ctensor.T)
        return grad_first_input

    def reset_param(self):
        self.weights = np.zeros((self.in_features, self.out_features), dtype=np.float32)
        fan_in = np.sqrt(3. / self.in_features)
        self.weights += np.random.uniform(-fan_in, fan_in, (self.in_features, self.out_features)).astype(np.float32)
        self.weights = self.register_parameter(self.weights)
        if self.use_bias:
            self.bias = np.zeros((self.out_features), dtype=np.float32)
            self.bias += np.random.rand(self.out_features)
            self.bias = self.register_parameter(self.bias)


class Transform(Module):
    '''
    Transform a tensor from (input_shape) to (output_shape)
    '''
    def __init__(self, input_shape, output_shape, **kwargs):
        super().__init__()
        self.inputs = None
        self.outputs = None

        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, x):
        self.inputs = x
        self.outputs = x.reshape(self.output_shape)
        return self.outputs

    def cforward(self, inputs):
        return inputs.reshape(self.output_shape)

    def backward(self, eta):
        return eta.reshape(self.input_shape)

    def cbackward(self, eta):
        return eta.reshape(self.input_shape)