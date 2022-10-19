'''
layer.py
====================

Basic modules
'''

import cupy as np #import numpy as np
import time

class Parameter(object):
    def __init__(self, tensor):
        self.iter = 0
        self.dtype = tensor.dtype
        self.tensor = tensor
        self.ctensor = tensor.astype(np.complex64)
        self.gradient = np.zeros(self.tensor.shape, dtype=np.float32)
        # self.delta_w = np.zeros_like(self.gradient)

    def __getitem__(self, i):
        return self.tensor[i]

class Module(object):
    '''
    Module is similar to nn.Module in pytorch
    '''
    def __init__(self):
        self.parameters = []
        self.inputs = []
        self.outputs = []

        self.inner_layers = None
        self.training = True

    def forward(self, inputs, *argv):
        ''' forward pass of a module '''
        raise NotImplementedError

    def cforward(self, inputs, *argv):
        ''' complex forward pass'''
        return

    def update(self, optimizer):
        for param in self.parameters:
            optimizer.update(param)

    def register_parameter(self, tensor):
        param = Parameter(tensor)
        self.parameters.append(param)
        return param


class Sequential(Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = layers

        for layer in layers:
            if layer.inner_layers is not None:
                for block_layer in layer.inner_layers:
                    for real_layer in block_layer.inner_layers:
                        self.parameters.extend(real_layer.parameters)
            else:
                self.parameters.extend(layer.parameters)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.layers:
            layer.inputs = outputs
            outputs = layer.forward(outputs)
            layer.outputs = outputs
        return outputs

    def cforward(self, inputs):
        outputs = inputs
        if inputs.dtype != np.complex64:
            outputs = inputs.astype(np.complex64)
        for layer in self.layers:
            outputs = layer.cforward(outputs)
        return outputs

    def backward(self, grad_first_output):
        temp_grad_first_output = grad_first_output
        for layer in reversed(self.layers):
            temp_grad_first_output = layer.backward(temp_grad_first_output)
        return

    def cbackward(self, grad_first_output):
        temp_grad_first_output = grad_first_output
        for layer in reversed(self.layers):
            temp_grad_first_output = layer.cbackward(temp_grad_first_output)
        return

    def adam_update(self, optimizer):
        for param in self.parameters:
            optimizer.update(param)

    def CG_update(self, optimizer, Ap_calculator, loss_calculator, grad_calculator,
                            global_grad_calculator,
                            X, Y, epochs, batch_id, global_X=None, global_Y=None):

        optimizer.Krylov_step(self.parameters, Ap_calculator, loss_calculator,
                              grad_calculator, global_grad_calculator,
                              X, Y, global_X, global_Y, epochs, batch_id)


class AutoEncoder(Module):
    def __init__(self, encoders, decoders):
        super().__init__()
        self.encoder_layers = encoders
        self.decoder_layers = decoders
        self.h = None

        for layer in encoders:
            self.parameters.extend(layer.parameters)
        
        for layer in decoders:
            self.parameters.extend(layer.parameters)

    def forward(self, inputs):
        outputs = inputs
        for layer in self.encoder_layers:
            layer.inputs = outputs
            outputs = layer.forward(outputs)
            layer.outputs = outputs
        for layer in self.decoder_layers:
            layer.inputs = outputs
            outputs = layer.forward(outputs)
            layer.outputs = outputs
        return outputs

    # def cforward(self, inputs):
    #     outputs = inputs
    #     if inputs.dtype != np.complex64:
    #         outputs = inputs.astype(np.complex64)
    #     for layer in self.layers:
    #         outputs = layer.cforward(outputs)
    #     return outputs

    def backward(self, grad_first_output):
        temp_grad_first_output = grad_first_output
        for layer in reversed(self.decoder_layers):
            temp_grad_first_output = layer.backward(temp_grad_first_output)
        
        for layer in reversed(self.encoder_layers):
            temp_grad_first_output = layer.backward(temp_grad_first_output)
        return

    # def cbackward(self, grad_first_output):
    #     temp_grad_first_output = grad_first_output
    #     for layer in reversed(self.layers):
    #         temp_grad_first_output = layer.cbackward(temp_grad_first_output)
    #     return

    def adam_update(self, optimizer):
        for param in self.parameters:
            optimizer.update(param)
