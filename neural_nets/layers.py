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

    def forward(self, X, training=True):
        self.layer_input = X
        return self.activation_function(X)

    def backward(self, accumulated_grad):
        return accumulated_grad * self.activation_function.gradient(self.layer_input)


class Flatten(Layer):

    def __init__(self, input_shape=None):
        super().__init__()
        self.prev_shape = None
        self.trainable = True
        self.input_shape = input_shape

    def get_output_shape(self):
        return np.prod(self.input_shape),

    def num_parameters(self):
        return 0

    def forward(self, X, training=True):
        self.prev_shape = X.shape
        return X.reshape(self.prev_shape[0], -1)

    def backward(self, accumulated_grad):
        return accumulated_grad.reshape(self.prev_shape)


class Conv2D(Layer):

    def __init__(self, num_filters, filter_size, stride=1, add_padding=True, input_shape=None):
        super().__init__()
        self.input_shape = input_shape
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
        self.W = np.random.uniform(-limit, limit,
                                   (self.num_filters, input_channels, self.filter_size, self.filter_size))
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
        # each column is one single convolutional operation, enables dot product between filter row and image column
        self.X_as_col = image_2_col(X, self.filter_size, self.stride, self.add_padding)
        # each row is now one flattened filter, channel after channel
        self.W_as_col = self.W.reshape((self.num_filters, -1))
        # perform dot product and add bias, broadcast it onto every column (so each single convolution)
        # the result are rows. each row = 1 feature map. pixel after pixel (row first), per channel
        # shape (num_filters, out_height * out_width * batch_sizes)
        out = self.W_as_col.dot(self.X_as_col) + self.b
        # first reshape into correct axes (num_filters, out_height, out_width, batch_size)
        # because this is the data_layout, for a feature map (row) you have a single output pixel (column)
        # for all batches
        out = out.reshape((*self.get_output_shape(), batch_size))
        # now you can swap axes => for each batch => for each feature map => get 2D output
        return out.transpose(3, 0, 1, 2)

    def backward(self, accumulated_grad):
        # accumulated grad is the gradient backpropagated up until the output of the forward pass of this
        # layer. The output of this layer was (batch_size, num_feature_maps, height, width).
        # Bring this back to (maps, height, width, batch_size) and reshape to (maps, height * width * batch_size)
        # => therefore, each row is the flattened backpropagated feature map.
        accumulated_grad = accumulated_grad.transpose(1, 2, 3, 0).reshape(self.num_filters, -1)

        # we need to accumulate the gradient
        if self.trainable:
            # gradient w.r.t. to W is the convolution of the input (X) with gradient of feature maps
            # (num_filters, out_height * out_width * batch_size) x
            # (num_channels * filter_height * filter_width, out_width * out_height, batch_size)^T
            # = (num_filters, num_channels * filter_height * filter_width)
            grad_W = accumulated_grad.dot(self.X_as_col.transpose()).reshape(self.W.shape)
            grad_b = np.sum(accumulated_grad, axis=1, keepdims=True)
            self.W = self.opt_W.update(self.W, grad_W)
            self.b = self.opt_b.update(self.b, grad_b)
        # grad w.r.t. to X is convolution of gradient dout with the filter matrix in column shape
        grad_X = self.W_as_col.transpose().dot(accumulated_grad)
        accumulated_grad = spray_gradient_to_img(grad_X, self.layer_input.shape, self.filter_size,
                                                 self.stride, self.add_padding)
        return accumulated_grad


def get_image_2_col_indices(images_shape, filter_size, padding, stride):
    # we split this to get have a better grasp what is going on even though the filter is always quadratic
    filter_width, filter_height = filter_size, filter_size
    num_batches, num_channels, height, width = images_shape
    out_height = int((height + 2 * padding - filter_size) / stride + 1)
    out_width = int((width + 2 * padding - filter_size) / stride + 1)

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
    return channel_indices, indices_y, indices_x


def image_2_col(images, filter_size, stride, add_padding):
    pad = calculate_padding(filter_size, add_padding)
    # calculate padding and pad respectively
    pad_b, pad_c, pad_h, pad_w = (0, 0), (0, 0), (pad, pad), (pad, pad)
    padded_images = np.pad(images, (pad_b, pad_c, pad_h, pad_w), mode='constant')
    ch_pos, h_pos, v_pos = get_image_2_col_indices(images.shape, filter_size, pad, stride)
    num_channels = images.shape[1]
    # conv cols is a matrix of size (batch_size) x (filter_width * filter_height * channels) x (out_width * out_height)
    # so each column represent a single conv-operation, the columns are ordered like convs which are performed row-major
    conv_cols = padded_images[:, ch_pos, h_pos, v_pos]
    # reorder so that (filter_width * filter_height * num_channels) x (out_width * out_height) x (batch_size)
    # so for each element get the offsets for each conv operation
    # when you reshape you have the first conv operation for all images (first "batch_size" rows) then the next etc...
    return conv_cols.transpose(1, 2, 0).reshape(filter_size * filter_size * num_channels, -1)


def spray_gradient_to_img(gradients, images_shape, filter_size, stride, add_padding):
    filter_height, filter_width = filter_size, filter_size
    batch_size, channels, height, width = images_shape
    pad = calculate_padding(filter_size, add_padding)
    height_padded = height + 2 * pad
    width_padded = width + 2 * pad
    images_padded = np.zeros((batch_size, channels, height_padded, width_padded))

    # Get the "origin" indices to where the
    k, i, j = get_image_2_col_indices(images_shape, filter_size, pad, stride)

    gradients = gradients.reshape(channels * filter_height * filter_width, -1, batch_size)
    gradients = gradients.transpose(2, 0, 1)
    # Add column content to the images at the indices
    np.add.at(images_padded, (slice(None), k, i, j), gradients)

    # Return image without padding
    return images_padded[:, :, pad:height + pad, pad:width + pad]


def calculate_padding(filter_size, add_padding):
    if not add_padding:
        return 0
    else:
        return (filter_size - 1) // 2
