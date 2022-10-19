'''
activation.py
====================

Activation functions
'''
import cupy as np #import numpy as np
from layers.layer import Module

class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.grad_first = None
        self.gradient = None
        self.use_optimize = False

    def calc_grad(self, X, sig, use_complex=False):
        # X = np.clip(X, -100, 100)
        # sig = 1. / (1. + np.exp(-X))
        if use_complex:
            self.cgradient = sig * (1 - sig)
            return self.cgradient
        else:
            self.gradient = sig * (1 - sig)
            return self.gradient

    def forward(self, X, require_grad=True):
        self.inputs = X
        self.outputs = 1 / (1 + np.exp(-X))
        if require_grad:
            self.calc_grad(X, self.outputs)
        return self.outputs

    def cforward(self, X, require_grad=True):
        inputs = X
        outputs = 1. / (1. + np.exp(-X))
        if require_grad:
            self.calc_grad(X, outputs, use_complex=True)
        return outputs

    def comp2ex_forward(self, X):
        return np.comp2ex(1.) / (np.comp2ex(1.) + np.exp(-X))

    def backward(self, grad_first_output):
        grad_first_input = self.gradient * grad_first_output
        return grad_first_input

    def cbackward(self, grad_first_output):
        grad_first_input = self.cgradient * grad_first_output
        return grad_first_input


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.grad_first = None
        self.gradient = None
        self.use_optimize = False

    def calc_grad(self, X, use_complex=False):
        X_ = np.zeros_like(X)
        X_[X >= 0] = 1
        if use_complex:
            self.cgradient = X_
            return self.cgradient
        else:
            self.gradient = X_
            return self.gradient

    def forward(self, X, require_grad=True):
        self.inputs = X
        self.outputs = np.maximum(X, 0)
        if require_grad:
            self.calc_grad(X)
            # self.calc_hessian(X)
        return self.outputs

    def cforward(self, X, require_grad=True):
        inputs = X
        outputs = np.maximum(X, 0)
        if require_grad:
            self.calc_grad(X, use_complex=True)
        return outputs

    def backward(self, grad_first_output):
        grad_first_input = self.gradient * grad_first_output
        return grad_first_input

    def cbackward(self, grad_first_output):
        grad_first_input = self.cgradient * grad_first_output
        return grad_first_input


class ELU(Module):
    def __init__(self):
        super().__init__()
        self.inputs = None
        self.outputs = None
        self.coutputs = None
        self.grad_first = None
        self.gradient = None
        self.use_optimize = False
        self.alpha = 0.9

    def calc_grad(self, X, use_complex=False):
        if use_complex:
            self.cgradient = np.full_like(self.coutputs, 1)
            self.cgradient[self.coutputs < 0] = self.coutputs[self.coutputs < 0] + self.alpha
            return self.cgradient
        else:
            self.gradient = np.full_like(self.outputs, 1)
            self.gradient[self.outputs < 0] = self.outputs[self.outputs < 0] + self.alpha
            return self.gradient

    def forward(self, X, require_grad=True):
        self.inputs = X
        self.outputs = np.maximum(X, 0)  + np.minimum(0, self.alpha * (np.exp(X) - 1))
        if require_grad:
            self.calc_grad(X)
            # self.calc_hessian(X)
        return self.outputs

    def cforward(self, X, require_grad=True):
        self.coutputs = np.maximum(X, 0)  + np.minimum(0, self.alpha * (np.exp(X) - 1))
        if require_grad:
            self.calc_grad(X, use_complex=True)
        return self.coutputs


    def backward(self, grad_first_output):
        grad_first_input = self.gradient * grad_first_output
        return grad_first_input

    def cbackward(self, grad_first_output):
        grad_first_input = self.cgradient * grad_first_output
        return grad_first_input
