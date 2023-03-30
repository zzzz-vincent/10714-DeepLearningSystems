"""Optimization module"""
import needle as ndl
import numpy as np

class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        for theta in self.params:
            grad = theta.grad.data + self.weight_decay * theta.data
            self.u[theta] = self.u.get(theta, 0) * self.momentum + (1. - self.momentum) * grad
            updated = theta.data - self.lr * self.u[theta]
            theta.data = ndl.Tensor(updated, dtype=theta.dtype)

    def clip_grad_norm(self, max_norm=0.25):
        """
        Clips gradient norm of parameters.
        """
        total_norm = np.linalg.norm(np.array([np.linalg.norm(p.grad.detach().numpy()).reshape((1,)) for p in self.params]))
        clip_coef = max_norm / (total_norm + 1e-6)
        clip_coef_clamped = min((np.asscalar(clip_coef), 1.0))
        for p in self.params:
            p.grad = p.grad.detach() * clip_coef_clamped


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        bias_correction=True,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        self.t += 1
        for theta in self.params:
          grad = theta.grad.data + theta.data * self.weight_decay
          m_update = (1 - self.beta1) * grad
          v_update = (1 - self.beta2) * (grad ** 2)
          if theta in self.m:
            self.m[theta] *= self.beta1
            self.m[theta] += m_update
          else:
            self.m[theta] = m_update
            
          if theta in self.v:
            self.v[theta] *= self.beta2
            self.v[theta] += v_update
          else:
            self.v[theta] = v_update

          m_hat = self.m[theta] / (1 - self.beta1 ** self.t)
          v_hat = self.v[theta] / (1 - self.beta2 ** self.t)
          theta.data = theta.data - self.lr * m_hat / (v_hat ** 0.5 + self.eps)
