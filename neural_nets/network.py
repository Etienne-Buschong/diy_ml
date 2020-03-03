import numpy as np
from utils.data_transforms import batch_iterator


class Network:

    def __init__(self, optimizer, loss, validation_data=None):
        self.optimizer = optimizer
        self.loss_fn = loss
        self.validation_data = validation_data
        self.layers = []

    def add_layer(self, layer):
        if len(self.layers) > 0:
            layer.set_input_shape(self.layers[-1].get_output_shape())
        init_operation = getattr(layer, 'initialize', None)
        if init_operation is not None and callable(init_operation):
            layer.initialize(self.optimizer)
        self.layers.append(layer)

    def forward(self, X, training=True):
        for layer in self.layers:
            X = layer.forward(X, training)
        return X

    def backward(self, loss_grad):
        for layer in reversed(self.layers):
            loss_grad = layer.backward(loss_grad)

    def test_batch(self, X, Y):
        Y_predicted = self.forward(X)
        print(np.sum(np.argmax(Y_predicted, axis=1) == np.argmax(Y, axis=1)))
        loss = self.loss_fn.loss(Y, Y_predicted)
        return loss

    def train_batch(self, X, Y):
        Y_predicted = self.forward(X)
        mean_loss = np.mean(self.loss_fn.loss(Y, Y_predicted))
        loss_grad = self.loss_fn.gradient(Y, Y_predicted)
        self.backward(loss_grad)
        return mean_loss

    def train_model(self, X, Y, epochs, batch_size):
        for epoch in range(epochs):
            print("Epoch {}".format(epoch + 1))
            for batch_idx, (batch_X, batch_Y) in enumerate(batch_iterator(X, Y=Y, batch_size=batch_size)):
                print("\rBatch {}".format(batch_idx + 1), end='', flush=True)
                loss = self.train_batch(batch_X, batch_Y)
            print()

