import numpy as np

'''
 Optimizers
'''


# Vanilla gradient descent class, only with learning rate
class StochasticGradientDescent:

    def __init__(self, lr=0.01):
        self.lr = lr

    def update(self, w, grad_w):
        return w - self.lr * grad_w
