import numpy as np


class Activation(object):

    def __call__(self, a):
        raise NotImplementedError()

    def gradient(self, a):
        raise NotImplementedError()


class Sigmoid(Activation):

    def __call__(self, a):
        return 1 / (1 + np.exp(-a))

    def gradient(self, a):
        return self.__call__(a) * (1.0 - self.__call__(a))


class Softmax(Activation):

    def __call__(self, a):
        numerator = np.exp(a)
        denominator = np.sum(numerator, axis=-1, keepdims=True)
        return numerator / denominator

    def gradient(self, a):
        return self.__call__(a) * (1.0 - self.__call__(a))