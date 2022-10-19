import cupy as np #import numpy as np

class BasicLoss:
    def __init__(self):
        return

    def forward(self, Y_pred, Y_gt):
        return 0

    def d_forward(self, Y_pred, Y_gt):
        return 0

    def d_d_forward(self, Y_pred, Y_gt):
        return 0

    def check_d_forward(self, Y_pred, Y_gt):
        B, N = Y_pred.shape
        cy = Y_pred.astype(np.complex64)

        grad = np.zeros((B, N))

        for B1 in range(B):
            for N1 in range(N):
                cy[B1, N1] += 1e-15j
                L = self.forward(cy, Y_gt)
                grad[B1, N1] = L.imag / (1e-15)

                cy[B1, N1] -= 1e-15j
        # print(grad)
        return grad

class MSELoss(BasicLoss):
    def __init__(self):
        return

    def forward(self, Y_pred, Y_gt):
        B = Y_pred.shape[0]
        return np.sum((Y_pred - Y_gt)*(Y_pred - Y_gt)) * 0.5 / B

    def d_forward(self, Y_pred, Y_gt):
        B, N = Y_pred.shape
        return (Y_pred - Y_gt) / B

    def d_d_forward(self, Y_pred, Y_gt):
        B, N = Y_pred.shape
        return np.identity(B*N).reshape((B, N, B, N)) / B


class CrossEntropyLoss(BasicLoss):
    def __init__(self):
        self.epsilon = 1e-12
        return

    def softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def forward(self, Y_pred, Y_gt):
        m = Y_gt.shape[0]
        p = self.softmax(Y_pred)
        # print(p)
        p = np.clip(p, self.epsilon, 1.-self.epsilon)
        list_m = np.array(range(m))
        log_likelyhood = -np.log(p[list_m, Y_gt])
        loss = np.sum(log_likelyhood) / m
        return loss

    def d_forward(self, Y_pred, Y_gt):
        m = Y_gt.shape[0]
        grad = self.softmax(Y_pred)
        list_m = np.array(range(m))
        grad[list_m, Y_gt] -= 1
        grad = grad / m
        return grad


class Hinge2Loss(BasicLoss):
    def __init__(self, model):
        self.model = model
        return

    def forward(self, Y_pred, Y_gt):
        Y = Y_pred.reshape(-1)
        z = (Y * Y_gt)
        loss_hinge = np.maximum(0, 1-z)
        loss_hinge = 0.5 * np.mean(loss_hinge ** 2)
        for param in self.model.parameters:
            if len(param.tensor.shape) == 2:
                t = param.tensor.reshape(-1)
                loss_hinge += np.dot(t, t) * 0.5 * 0.01
        return loss_hinge

    def d_forward(self, Y_pred, Y_gt):
        Y = Y_pred.reshape(-1)
        z = (Y * Y_gt)
        grad = -np.array(Y_gt) / Y_pred.shape[0]
        grad[z > 1] = 0
        grad = grad * np.maximum(0, 1-z)
        return grad.reshape(Y_pred.shape)
