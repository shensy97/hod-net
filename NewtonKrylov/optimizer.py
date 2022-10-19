import cupy as np #import numpy as np
import copy
import math
import os
import time
from operator import itemgetter
# from utils.random_slice import random_pick, random_pick_by_age

def cos_distance(vec_a, vec_b):
    return np.dot(vec_a, vec_b) / np.linalg.norm(vec_a) / np.linalg.norm(vec_b)


class AdamOptimizer:
    def __init__(self, lr=0.001, momentum=0.9, beta=0.999, epsilon=1e-8):
        self.lr = lr
        # self.momentum = momentum
        # self.beta = beta
        self.beta1 = momentum
        self.beta2 = beta
        self.s = {}
        self.v = {}
        self.g = {}
        self.epsilon = epsilon

    def update(self, param):
        # print("param shape", param[0].gradient)
        if param not in self.s:
            self.s[param] = np.zeros_like(param.gradient)
            self.v[param] = np.zeros_like(param.gradient)
        param.iter += 1
        self.g[param] = copy.deepcopy(param.gradient)
        exp_avg, exp_avg_sq = self.s[param], self.v[param]
        bias_correction1 = 1 - np.power(self.beta1, param.iter)
        bias_correction2 = 1 - np.power(self.beta2, param.iter)
        grad = param.gradient
        exp_avg = self.beta1 * exp_avg + (1 - self.beta1) * grad
        exp_avg_sq = self.beta2 * exp_avg_sq + (1 - self.beta2) * grad * grad
        denom = np.sqrt(exp_avg_sq) / (np.sqrt(bias_correction2) + self.epsilon)
        step_size = self.lr / bias_correction1
        if np.min(np.abs(denom)) > 1e-300:
            param.tensor -= step_size * (exp_avg / denom)
        self.s[param], self.v[param] = exp_avg, exp_avg_sq
        # print("adam, grad", np.sum(param.gradient ** 2))
        param.gradient.fill(0)



class Krylov_optimizer:
    def __init__(self, logger, cfg):
        self.logger = logger
        self.config = cfg

        self.limit_batch_num = cfg.train.limit_batch_num

        self.taylor_threshold = cfg.optim.taylor_threshold
        self.damp_coef        = cfg.optim.damp_coef
        self.verbose          = cfg.optim.verbose
        self.CG_maxiter       = cfg.optim.CG_maxiter
        self.r_threshold      = cfg.optim.CG_quit_coef
        self.log_interval     = cfg.optim.CG_log_interval

        self.start_of_CG = False

        self.global_grad = None
        self.should_update_grad = True

    def collect_gradient(self, params):
        grad = []
        for param in params:
            grad.append(param.gradient.reshape(-1))
        grad = np.concatenate(grad)
        return grad

    def set_param(self, params, delta_w, step_len, reset=False):
        if reset:
            coef = 1.0
        else:
            coef = -1.0
        start_p = 0
        for param in params:
            len_p = param.tensor.size
            param.tensor -= coef * step_len * delta_w[start_p : start_p+len_p].reshape(param.tensor.shape)
            param.ctensor = param.tensor.astype(np.complex64)
            start_p += len_p

    def CG_solver(self, b, Hp_calculator, Y, X):
        b = -b
        n = b.size
        x = np.zeros(n, dtype=np.float32)
        norm_b = np.linalg.norm(b)

        # first_Ax = Hp_calculator(x, Y, X)
        r = b
        z = r
        p = z
        r_k_norm = np.dot(r, z)

        termination_epsilon = (norm_b * self.r_threshold) ** 2
        max_error = 0.0


        max_iter = self.CG_maxiter
        for i in range(max_iter):
            Ap = Hp_calculator(p, Y, X)
            pAp = np.dot(p, Ap)

            if pAp > 0 and pAp < 1e-8:
                pAp = pAp + self.damp_coef * np.dot(p, p)
                if self.verbose:
                    self.logger.info("pAp {} < 1e-8, damp pp".format(pAp))
                    self.logger.info("after damp pp {}, pAp {}".format(np.dot(p, p), pAp))

            if pAp < 0:
                if i == 0:
                    if self.verbose:
                        self.logger.info("first pAp < 0, walk directly")
                    pAp = np.dot(p, p)
                    Ap = p
                    alpha = np.dot(r, r) / pAp
                else:
                    if self.verbose:
                        self.logger.info("pAp {} < 0, quit directly".format(pAp))
                    break

            elif pAp > 1e15:
                break

            alpha = r_k_norm / pAp
            x = x + alpha * p
            r = r - alpha * Ap
            z = r
            r_kplus1_norm = np.dot(r, z)
            beta = r_kplus1_norm / r_k_norm
            r_k_norm = r_kplus1_norm

            if r_kplus1_norm > max_error:
                max_error = r_kplus1_norm

            if r_kplus1_norm < termination_epsilon:
                if self.verbose:
                    self.logger.info('Itr: %d' %(i))
                break
            p = beta * p + z

            if self.verbose and i % self.log_interval == 0:
                self.logger.info(" - {} residual {} xnorm {}".format(
                                  i,
                                  np.sqrt(r_kplus1_norm) / norm_b,
                                  np.linalg.norm(x))
                                )
        if self.verbose:
            self.logger.info("Total Iteration: {}".format(i))
        return x


    def Krylov_step(self, params, Hp_calculator,
                    loss_calculator, grad_calculator,
                    global_grad_calculator,
                    X, Y, global_X, global_Y, epoch_id=-1, batch_id=-1):
        if self.verbose:
            self.logger.info("======================")

        if self.start_of_CG:
            self.should_update_grad = True
            self.start_of_CG = False

        if self.should_update_grad:
            self.should_update_grad = False
            if self.verbose:
                self.logger.info("Refresh global grad")
            global_grad, global_loss = global_grad_calculator(global_X, global_Y, True, method="no_stack")
            self.global_grad = global_grad
            if self.verbose:
                self.logger.info("global_loss now {}".format(global_loss))

        local_grad = grad_calculator(X, Y)
        grad = self.global_grad.copy()
        n = grad.size

        screen_for_local_grad = np.dot(local_grad, grad)
        if screen_for_local_grad < 0 or np.isnan(screen_for_local_grad) or np.isinf(screen_for_local_grad):
            if self.verbose:
                self.logger.info("negative local grad {}, discard this batch directly".format(cos_distance(local_grad, grad)))
            return
        else:
            if self.verbose:
                self.logger.info("positive local grad {}, continue this batch".format(screen_for_local_grad))

        dw = self.CG_solver(grad, Hp_calculator, Y, X)

        post_screen = np.dot(dw, -grad)
        if post_screen < 0:
            return


        L_now = loss_calculator(X, Y)
        step_len_O = 1.0
        self.set_param(params, dw, step_len_O, reset=False)
        L_next = loss_calculator(X, Y)
        self.set_param(params, dw, step_len_O, reset=True)

        if np.isnan(L_next) or np.isinf(L_next):
            self.logger.info("Trigger Nan or inf")
            return

        second_term = 0.5 * dw.dot(Hp_calculator(dw, Y, X))
        first_term = dw.dot(local_grad)
        quadratic = second_term + first_term

        # the taylor ratio
        O_taylor = (L_next - L_now - quadratic)
        O_taylor = O_taylor / (L_next - L_now)
        O_taylor = np.abs(O_taylor)

        eta_threshold = self.taylor_threshold

        if self.verbose:
            self.logger.info("Taylor ratio threshold {}, Taylor ratio {}".format(eta_threshold, O_taylor))

        step_len = eta_threshold / O_taylor

        O_taylor_validate = O_taylor
        first_term_standard = first_term
        second_term_standard = second_term
        MAX_STEP = 50
        step_count = 0
        if O_taylor < eta_threshold:
            if self.verbose:
                self.logger.info("Tayler ratio {} is smaller than threshold, directly update".format(O_taylor))
            step_len = 1.0

            self.set_param(params, dw, step_len, reset=False)
            L_next = loss_calculator(X, Y)
            self.set_param(params, dw, step_len, reset=True)

            if L_next > L_now * 1.5:
                # may damage, set len to zero
                step_len = 0.0

        else:
            step_len = np.abs(eta_threshold / O_taylor)
            if L_next < L_now:
                step_len_default = step_len
                if self.verbose:
                    self.logger.info("guessed step length {} can be used ".format(step_len_default))
            else:
                step_len_default = np.array(0.0)

            while np.abs(O_taylor_validate) < 0.5 * eta_threshold or eta_threshold < O_taylor_validate:
                step_count += 1
                if step_count > MAX_STEP or step_len < 1e-7:
                    if self.verbose:
                        self.logger.info("Reach max step len {}, quit with step len {}".format(MAX_STEP, step_len_default))

                    step_len = step_len_default.copy()
                    break
                if np.abs(O_taylor_validate) > eta_threshold:
                    step_len *= 0.5
                else:
                    step_len *= 1.5

                self.set_param(params, dw, step_len, reset=False)
                L_next = loss_calculator(X, Y)
                self.set_param(params, dw, step_len, reset=True)

                first_term_validate  = first_term_standard  * step_len
                second_term_validate = second_term_standard * step_len * step_len
                quadratic            = second_term_validate + first_term_validate
                O_taylor_validate = (L_next - L_now - quadratic)
                O_taylor_validate = O_taylor_validate / (L_next - L_now)
                O_taylor_validate = np.abs(O_taylor_validate)

                if self.verbose:
                    self.logger.info("- O_taylor_validate {} step len {}, L_next {}, L_now {}".format(
                                     O_taylor_validate, step_len, L_next, L_now))

                if np.isnan(L_next) or np.isinf(L_next):
                    if self.verbose:
                        self.logger.info("Trigger Nan or inf")
                    step_len = 0.0
                    break
        self.logger.info("Update with step len {}, tayloy ratio {} -> {}.".format(step_len, O_taylor, O_taylor_validate))
        self.set_param(params, dw, step_len, reset=False)
        self.should_update_grad = True
