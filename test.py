import pickle

from neural_nets.activations import Softmax, Sigmoid
from neural_nets.losses import CrossEntropyLoss
from neural_nets.network import Network
from neural_nets.optimizer import StochasticGradientDescent
from utils.data_transforms import vectorize_targets
from neural_nets.layers import Dense, Activation


def load():
    with open("./datasets/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

import numpy as np
train_images, train_labels, test_images, test_labels = load()
train_images, test_images = train_images.astype(np.float32), test_images.astype(np.float32)
train_images /= 255.0
net = Network(StochasticGradientDescent(), CrossEntropyLoss())
net.add_layer(Dense(100, input_shape=(784,)))
net.add_layer(Activation(Sigmoid))
net.add_layer(Dense(10))
net.add_layer(Activation(Softmax))
train_labels, test_labels = vectorize_targets(train_labels, 10), vectorize_targets(test_labels, 10)
net.train_model(train_images, train_labels, 5, 32)
net.test_batch(test_images, test_labels)
