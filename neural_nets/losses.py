import numpy as np


class Loss(object):

    def loss(self, y, y_hat):
        raise NotImplementedError()

    def gradient(self, y, y_hat):
        raise NotImplementedError()


class SquareLoss(Loss):

    def loss(self, y, y_hat):
        return 0.5 * np.square(y - y_hat)

    def gradient(self, y, y_hat):
        return y - y_hat


class CrossEntropyLoss(Loss):

    def loss(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(y * np.log(y_hat) + (1 - y) * np.log(y_hat))

    def gradient(self, y, y_hat):
        y_hat = np.clip(y_hat, 1e-15, 1 - 1e-15)
        return -(y / y_hat) + (1 - y) / (1 - y_hat)
