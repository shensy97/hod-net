import cupy as np #import numpy as np
import numpy
import time
import pickle as pk
from utils.net_utils import collect_gradient, collect_weights, set_weights, generate_random_weight_list


class FCNet():
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.count_batches = 0

        self.dh = 1e-16
        self.X_list = []
        self.Y_list = []

        self.len_parameters = 0
        for param in self.model.parameters:
            self.len_parameters += param.tensor.size
    
    def predict(self, X, Y):
        Y = np.array(Y)
        Y_pred = self.model.forward(X)
        L = self.loss.forward(Y_pred, Y)
        return Y_pred, L
    
    def fit_train(self, X, Y, epoch, global_batch_size):
        Y_pred = self.model.forward(X)
        L_before = self.loss.forward(Y_pred, Y)
        
        self.model.backward(self.loss.d_forward(Y_pred, Y))
        self.model.adam_update(self.optimizer)
        return None, [L_before]


class CGNet():
    '''
    Key function:
      complex_hessian_free: obtain Hessian-vector product.
    '''
    def __init__(self, model, loss, optimizer):
        self.model = model
        self.loss = loss
        self.optimizer = optimizer
        self.count_batches = 0

        self.dh = 1e-16
        self.X_list = []
        self.Y_list = []

        self.len_parameters = 0
        for param in self.model.parameters:
            self.len_parameters += param.tensor.size

    def loss_calculator(self, X, Y):
        Y_pred = self.model.forward(X)
        L = self.loss.forward(Y_pred, Y)
        return L

    def grad_calculator(self, X, Y, return_loss=False):
        Y_pred = self.model.forward(X)
        if return_loss:
            L = self.loss.forward(Y_pred, Y)
        dE_da = self.loss.d_forward(Y_pred, Y)
        self.model.backward(dE_da)
        grads = collect_gradient(self.model.parameters)
        if return_loss:
            return grads, L
        return grads


    def global_gradient_calculator(self, Xs, Ys, return_loss=False, method="stack_part"):
        if method == "no_stack":
            grads = np.zeros(self.len_parameters, dtype=np.float32)
            L = 0
            for i in range(len(Xs)):
                X, Y = Xs[i], Ys[i]
                Y_pred = self.model.forward(X)
                L += self.loss.forward(Y_pred, Y)
                dE_da = self.loss.d_forward(Y_pred, Y)
                self.model.backward(dE_da)
                grads += collect_gradient(self.model.parameters)

            if return_loss:
                return grads / len(Xs), L / len(Xs)
            return grads / len(Xs)

        elif method == "stack_part":
            stack_num = 40
            total = len(Xs)
            mean_grads = np.zeros(self.len_parameters)
            mean_losses = 0
            for st in range(total // stack_num):
                X_part = np.vstack(Xs[st * stack_num : (st+1) * stack_num])
                Y_part = np.concatenate(Ys[st * stack_num : (st+1) * stack_num])
                Y_pred = self.model.forward(X_part)
                if return_loss:
                    L = self.loss.forward(Y_pred, Y_part)
                    mean_losses += L * stack_num
                dE_da = self.loss.d_forward(Y_pred, Y_part)
                self.model.backward(dE_da)
                grads = collect_gradient(self.model.parameters)
                mean_grads += grads * stack_num
            if total % stack_num != 0:
                X_part = np.vstack(Xs[total - total % stack_num : ])
                Y_part = np.concatenate(Ys[total - total % stack_num : ])
                Y_pred = self.model.forward(X_part)
                if return_loss:
                    L = self.loss.forward(Y_pred, Y_part)
                    mean_losses += L * (total % stack_num)
                dE_da = self.loss.d_forward(Y_pred, Y_part)
                self.model.backward(dE_da)
                grads = collect_gradient(self.model.parameters)
                mean_grads += grads * (total % stack_num)
            mean_grads = mean_grads / total
            mean_losses = mean_losses / total
            if return_loss:
                return mean_grads, mean_losses
            return mean_grads

        else:
            print("method not implemented")
            assert 0


    def complex_hessian_free(self, p, Y_gt, X_input=None):
        p_start = 0

        for param in self.model.parameters:
            len_p = param.ctensor.size
            layer_p = p[p_start: p_start+len_p].reshape(param.ctensor.shape)
            p_start += len_p
            param.ctensor = param.tensor.astype(np.complex64) + layer_p * self.dh * 1j

        Y_pred = self.model.cforward(self.model.layers[0].inputs)
        dL = self.loss.d_forward(Y_pred, Y_gt)
        self.model.cbackward(dL)

        Ap = np.zeros_like(p)
        p_start = 0
        for param in self.model.parameters:
            len_p = param.ctensor.size
            layer_Ap = param.cgradient.imag / self.dh
            Ap[p_start: p_start + len_p] = layer_Ap.reshape(-1)
            p_start += len_p

        return Ap

    def CG_update(self, X, Y_gt, epoch, batch_id=-1,
                        global_X=None, global_Y=None):
        self.model.CG_update(self.optimizer,
                             self.complex_hessian_free,
                             self.loss_calculator,
                             self.grad_calculator,
                             self.global_gradient_calculator,
                             X, Y_gt,
                             epoch, batch_id,
                             global_X, global_Y)

    def predict(self, X, Y):
        Y = np.array(Y)
        Y_pred = self.model.forward(X)
        L = self.loss.forward(Y_pred, Y)
        return Y_pred, L

    def fit_train(self, X, Y, epoch, global_batch_size):
        self.count_batches += 1
        self.X_list.append(np.array(X))
        self.Y_list.append(np.array(Y))

        if self.count_batches % global_batch_size == 0:
            L_s = []
            for i in range(global_batch_size):
                Y_pred = self.model.forward(self.X_list[i])
                L_before = self.loss.forward(Y_pred, self.Y_list[i])
                if self.optimizer.verbose:
                    self.optimizer.logger.info("[at epoch {}] local batch loss {}".format(epoch, L_before))
                self.model.backward(self.loss.d_forward(Y_pred, self.Y_list[i]))
                self.CG_update(self.X_list[i], self.Y_list[i], epoch, i,
                                           self.X_list, self.Y_list)
                Y_pred = self.model.forward(self.X_list[i])
                L_s.append(L_before)

            self.X_list = []
            self.Y_list = []
            self.count_batches = 0
            return None, L_s

        return None, None