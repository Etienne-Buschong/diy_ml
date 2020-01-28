import numpy as np


def batch_iterator(X, Y=None, batch_size = 32):
    n = X.shape[0]
    for index in range(0, n, batch_size):
        begin, end = index, min(index + batch_size, n)
        if Y is not None:
            yield X[begin:end], Y[begin:end]
        else:
            yield X[begin:end]


def vectorize_targets(Y, num_classes):
    assert Y.ndim == 1
    return np.eye(num_classes)[Y]

