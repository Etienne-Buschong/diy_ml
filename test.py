import pickle

from neural_nets.activations import Softmax, Sigmoid
from neural_nets.losses import CrossEntropyLoss
from neural_nets.network import Network
from neural_nets.optimizer import StochasticGradientDescent
from utils.data_transforms import vectorize_targets
from neural_nets.layers import Dense, Activation, Conv2D, Flatten


def load():
    with open("./datasets/mnist.pkl", 'rb') as f:
        mnist = pickle.load(f)
        return mnist["training_images"], mnist["training_labels"], mnist["test_images"], mnist["test_labels"]

import numpy as np
train_images, train_labels, test_images, test_labels = load()
train_images, test_images = train_images.astype(np.float32), test_images.astype(np.float32)
train_images /= 255.0
test_images /= 255.0
train_images = train_images.reshape(60000, 1, 28, 28)
test_images = test_images.reshape(10000, 1, 28, 28)
train_labels, test_labels = vectorize_targets(train_labels, 10), vectorize_targets(test_labels, 10)


# net = Network(StochasticGradientDescent(), CrossEntropyLoss())
# net.add_layer(Conv2D(16, 3, input_shape=(1, 28, 28)))
# net.add_layer(Activation(Sigmoid))
# net.add_layer(Conv2D(32, 3))
# net.add_layer(Activation(Sigmoid))
# net.add_layer(Flatten())
# net.add_layer(Dense(30))
# net.add_layer(Dense(10))
# net.add_layer(Activation(Softmax))
# net.train_model(train_images, train_labels, 5, 256)
# net.test_batch(test_images, test_labels)


# net = Network(StochasticGradientDescent(), CrossEntropyLoss())
# net.add_layer(Dense(100, input_shape=(784,)))
# net.add_layer(Activation(Sigmoid))
# net.add_layer(Dense(10))
# net.add_layer(Activation(Softmax))
# net.train_model(train_images, train_labels, 5, 32)
# net.test_batch(test_images, test_labels)


input = np.array([0, 1, 0, -1])
fft_input = 0.5 * np.fft.fft(input)
print(fft_input)
conv = 2.0 * (fft_input * fft_input)
print(conv)
back = np.fft.ifft(conv)
print(back)

