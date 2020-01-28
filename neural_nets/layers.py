import numpy as np
import math
from copy import copy


class Layer(object):

    def __init__(self):
        self.input_shape = None

    def set_input_shape(self, input_shape):
        self.input_shape = input_shape

    def get_output_shape(self):
        raise NotImplementedError()

    def num_parameters(self):
        raise NotImplementedError()

    def forward(self, X, training=True):
        raise NotImplementedError()

    def backward(self, accumulated_grad):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, num_units, input_shape=None):
        super().__init__()
        self.num_units = num_units
        self.input_shape = input_shape
        self.layer_input = None  # needed for back-propagation
        self.trainable = True
        self.W = None
        self.opt_W = None
        self.b = None
        self.opt_b = None

    def initialize(self, optimizer):
        interval_limit = 6 / math.sqrt(self.input_shape[0] + self.num_units)
        self.W = np.random.uniform(-interval_limit, interval_limit, (self.input_shape[0], self.num_units))
        self.b = np.zeros((1, self.num_units))
        self.opt_W = copy(optimizer)
        self.opt_b = copy(optimizer)

    def get_output_shape(self):
        return self.num_units,

    def num_parameters(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def forward(self, X, training=True):
        self.layer_input = X
        return np.dot(X, self.W)

    def backward(self, accumulated_grad):
        # save in case that it is overwritten by training step
        W_tmp = self.W

        if self.trainable:
            # calculate the gradients with respect to params
            grad_W = np.dot(self.layer_input.transpose(), accumulated_grad)
            grad_b = np.sum(accumulated_grad, axis=0)
            self.W = self.opt_W.update(self.W, grad_W)
            self.b = self.opt_b.update(self.b, grad_b)

        return np.dot(accumulated_grad, W_tmp.transpose())


class Activation(Layer):

    def __init__(self, activation_fn):
        super().__init__()
        self.activation_function = activation_fn()
        self.layer_input = None

    def get_output_shape(self):
        return self.input_shape

    def num_parameters(self):
        return 0

    def forward(self, X, training=   True):
        self.layer_input = X
        return self.activation_function(X)

    def backward(self, accumulated_grad):
        return accumulated_grad * self.activation_function.gradient(self.layer_input)


class Conv2D(Layer):

    def __init__(self, num_filters, filter_size, stride=1, add_padding=True, input_shape=None):
        super().__init__()
        self.num_filters = num_filters
        self.filter_size = filter_size
        self.stride = stride
        self.add_padding = add_padding
        self.trainable = True
        self.W = None
        self.opt_W = None
        self.b = None
        self.opt_b = None
        self.W_as_col = None
        self.X_as_col = None
        self.layer_input = None

    def initialize(self, optimizer):
        input_channels = self.input_shape[0]
        limit = 6.0 / math.sqrt(self.filter_size * self.filter_size)
        self.W = np.random.uniform(-limit, limit, (self.num_filters, input_channels, self.filter_size, self.filter_size))
        self.b = np.zeros((self.num_filters, 1))
        self.opt_W = copy(optimizer)
        self.opt_b = copy(optimizer)

    def get_output_shape(self):
        channels, height, width = self.input_shape
        padding = calculate_padding(self.filter_size, self.add_padding)
        out_height = (height + 2 * padding - self.filter_size) // self.stride + 1
        out_width = (width + 2 * padding - self.filter_size) // self.stride + 1
        return self.num_filters, out_height, out_width

    def num_parameters(self):
        return np.prod(self.W.shape) + np.prod(self.b.shape)

    def forward(self, X, training=True):
        batch_size, channels, height, width = X.shape
        self.layer_input = X

    def backward(self, accumulated_grad):
        pass


def get_image_2_col_indices(padded_images, filter_size, padding, stride):
    # we split this to get have a better grasp what is going on even though the filter is always quadratic
    filter_width, filter_height = filter_size
    num_batches, num_channels, height, width = padded_images.shape
    out_height = (height + 2 * padding - filter_size) / stride + 1
    out_width = (width + 2 * padding - filter_size) / stride + 1

    # get the x/y-indices which would be needed for a single convolutional operation (at top left corner)
    # for height indices they repeat for a row then increment on next row
    single_ys = np.repeat(np.arange(filter_height), filter_width)
    # for width indices they increment first (in a row), then reset to zero for the next row of the convolution
    single_xs = np.tile(np.arange(filter_width), filter_height)
    # repeat the indices since convolutional goes over all channels on each pixel
    # this is just a concatenation of the single channel "templates" for channel-times
    channel_ys = np.tile(single_ys, num_channels)
    channel_xs = np.tile(single_xs, num_channels)
    # calculate the offsets which occur for each index during complete convolution (include stride into calculation)
    # offset y stays in the same position for out_width-many strides then skips to next row (including stride)
    offset_ys = stride * np.repeat(np.arange(out_height), out_width)
    # offset x increments over a single "row" of strides than resets when the convolutional filter jumps to next row
    offset_xs = stride * np.tile(np.arange(out_width), out_height)
    # generate a list of offsets for each element in channel_ys and channel_xs
    # therefore a column is equal to a single "position" of the convolutional mask
    # and a row are all different possible indices of an filter position with the different offsets
    indices_y = channel_ys.reshape(-1, 1) + offset_ys.reshape(1, -1)
    indices_x = channel_xs.reshape(-1, 1) + offset_xs.reshape(1, -1)
    # generate a channel-indices-array which is basically:
    # zeroth channel for filter_height * filter_width elements of a row, then first, the seconds
    # and therefore matching the number of rows of indices_x/indices_y
    channel_indices = np.repeat(np.arange(num_channels), filter_height * filter_width).reshape(-1, 1)
    return channel_indices, indices_y, indices_y


def image_2_col(images, filter_size, stride, add_padding):
    pad = calculate_padding(filter_size, add_padding)
    # calculate padding and pad respectively
    pad_b, pad_c, pad_h, pad_w = (0, 0), (0, 0), (pad, pad), (pad, pad)
    padded_images = np.pad(images, (pad_b, pad_c, pad_h, pad_w), mode='constant')
    ch_pos, h_pos, w_pos = get_image_2_col_indices(padded_images, filter_size, pad, stride)
    # TODO continue implementation here


def calculate_padding(filter_size, add_padding):
    if not add_padding:
        return 0
    else:
        return (filter_size - 1) // 2