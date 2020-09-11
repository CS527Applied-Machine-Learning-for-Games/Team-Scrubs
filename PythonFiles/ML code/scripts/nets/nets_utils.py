from keras.layers import Conv1D, Conv2D, Conv3D, Conv2DTranspose, Conv3DTranspose
from keras.layers import UpSampling1D, UpSampling2D, UpSampling3D
from keras.layers import MaxPooling1D, MaxPooling2D, MaxPooling3D
from keras.layers import AveragePooling1D, AveragePooling2D, AveragePooling3D
from keras.layers import GlobalAveragePooling1D, GlobalAveragePooling2D, GlobalAveragePooling3D
from keras.layers import SpatialDropout1D, SpatialDropout2D, SpatialDropout3D
from keras.layers import GRU
from keras.layers import Dense, Activation, Lambda, Dropout, Flatten, Concatenate
from keras.layers import BatchNormalization
from keras.layers import Add, Input
from keras.models import Model
from keras import backend as K
from utils.kerasutils import get_channel_axis
import numpy as np
import tensorflow as tf


"""Commonly used functions of different networks."""

__author__ = 'Ken C. L. Wong'


def ConvND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return Conv1D(**kwargs)
    elif dim == 4:
        return Conv2D(**kwargs)
    elif dim == 5:
        return Conv3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')
    

def ConvNDTranspose(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 4:
        return Conv2DTranspose(**kwargs)
    elif dim == 5:
        return Conv3DTranspose(**kwargs)
    else:
        raise Exception('Unsupported input size.')
    

def UpSamplingND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return UpSampling1D(**kwargs)
    elif dim == 4:
        return UpSampling2D(**kwargs)
    elif dim == 5:
        return UpSampling3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')
    

def MaxPoolingND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return MaxPooling1D(**kwargs)
    elif dim == 4:
        return MaxPooling2D(**kwargs)
    elif dim == 5:
        return MaxPooling3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')
    

def AveragePoolingND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return AveragePooling1D(**kwargs)
    elif dim == 4:
        return AveragePooling2D(**kwargs)
    elif dim == 5:
        return AveragePooling3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')


def GlobalAveragePoolingND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return GlobalAveragePooling1D(**kwargs)
    elif dim == 4:
        return GlobalAveragePooling2D(**kwargs)
    elif dim == 5:
        return GlobalAveragePooling3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')


def SpatialDropoutND(x, **kwargs):
    """Choose a function based on input size."""
    dim = K.ndim(x)
    if dim == 3:
        return SpatialDropout1D(**kwargs)
    elif dim == 4:
        return SpatialDropout2D(**kwargs)
    elif dim == 5:
        return SpatialDropout3D(**kwargs)
    else:
        raise Exception('Unsupported input size.')


def cc_reg(weight=0.0):
    def inner(weight_matrix):
        dim = weight_matrix.get_shape()
        sz = tf.multiply(tf.multiply(dim[0], dim[1]), dim[2])
        kernels = tf.reshape(weight_matrix, [sz, dim[3]])
        mn = tf.reduce_mean(kernels, 0, keepdims=True)
        mmn = tf.tile(mn, [sz, 1])
        df = tf.subtract(kernels, mmn)
        dp = tf.matmul(tf.transpose(df), df)
        nm = tf.norm(df, axis=0, keepdims=True)
        mnm = tf.multiply(nm, tf.transpose(nm))
        cc = tf.square(tf.div(dp, mnm + 1E-10))
        cc_cost = tf.reduce_mean(tf.reduce_max(cc, axis=0))
        return cc_cost * float(weight)
    return inner

def conv(num_filters, kernel_size=3, kernel_cc_weight=0.0, activation='relu', conv_order='conv_first', use_batch_norm=True, name=None,
         **conv_kwargs):
    """A composite layer for convolution with BatchNormalization."""

    assert conv_order in ['conv_first', 'conv_last']


    def inner(x):
        conv_first = conv_order == 'conv_first'
        if conv_first:
            x = ConvND(x, filters=num_filters, kernel_size=kernel_size, padding='same', kernel_regularizer=cc_reg(weight=kernel_cc_weight), **conv_kwargs)(x)
        x = BatchNormalization(axis=get_channel_axis())(x) if use_batch_norm else x
        x = Activation(activation, name=name)(x) if conv_first else Activation(activation)(x)
        if not conv_first:
            x = ConvND(x, filters=num_filters, kernel_size=kernel_size, padding='same', kernel_regularizer=cc_reg(weight=kernel_cc_weight), name=name, **conv_kwargs)(x)
        return x

    return inner


def conv_transpose(num_filters, kernel_size=3, activation='relu', conv_order='conv_first', use_batch_norm=True,
                   name=None, **conv_kwargs):
    """A composite layer for deconvolution with BatchNormalization."""

    assert conv_order in ['conv_first', 'conv_last']

    def inner(x):
        conv_first = conv_order == 'conv_first'
        if conv_first:
            x = ConvNDTranspose(x, filters=num_filters, kernel_size=kernel_size, padding='same', **conv_kwargs)(x)
        x = BatchNormalization(axis=get_channel_axis())(x) if use_batch_norm else x
        x = Activation(activation, name=name)(x) if conv_first else Activation(activation)(x)
        if not conv_first:
            x = ConvNDTranspose(x, filters=num_filters, kernel_size=kernel_size, padding='same', name=name,
                                **conv_kwargs)(x)
        return x

    return inner


def dense(num_units, activation='relu', use_batch_norm=True, **dense_kwargs):
    """A fully-connected layer with BatchNormalization."""

    def inner(tensor):
        tensor = Dense(num_units, **dense_kwargs)(tensor)
        tensor = BatchNormalization()(tensor) if use_batch_norm else tensor
        tensor = Activation(activation)(tensor)
        return tensor

    return inner


def dense_layers(num_units, num_layers=2, dropout_rate=0.0, activation='relu', use_batch_norm=True, **dense_kwargs):
    """Multiple fully-connected layers with BN and dropout."""

    def inner(tensor):
        if K.ndim(tensor) != 2:
            tensor = Flatten()(tensor)
        for _ in range(num_layers):
            tensor = dense(num_units=num_units, activation=activation, use_batch_norm=use_batch_norm, **dense_kwargs)(
                tensor)
            tensor = Dropout(dropout_rate)(tensor) if dropout_rate else tensor
        return tensor

    return inner


def gru(units, activation='tanh', dropout=0.5, return_sequences=True, conv_order='conv_first', use_batch_norm=True):
    """A GRU layer with BatchNormalization."""

    def inner(tensor):
        local_gru = GRU(units=units, activation=None, implementation=2, dropout=dropout,
                        return_sequences=return_sequences)  # Default GRU activation is 'tanh'.
        tensor = local_gru(tensor) if conv_order == 'conv_first' else tensor
        tensor = BatchNormalization(axis=get_channel_axis())(tensor) if use_batch_norm else tensor
        tensor = Activation(activation)(tensor)
        tensor = local_gru(tensor) if conv_order == 'conv_last' else tensor
        return tensor

    return inner


def resize_x():
    """Resizes x by zero-padding or cropping the number of channels for skip connection with identity mapping."""

    def inner(inputs):
        """Modifies x to be the same size as tensor."""
        assert len(inputs) == 2  # x, tensor. x to be modified.
        x = inputs[0]  # To be resized
        tensor = inputs[1]  # Reference
        axis = 1 if K.image_data_format() == 'channels_first' else -1
        channels_tensor = K.int_shape(tensor)[axis]
        channels_x = K.int_shape(x)[axis]
        if channels_x < channels_tensor:  # Padding
            padding = K.zeros_like(tensor)
            padding = padding[:, channels_x:, ...] if axis == 1 else padding[..., channels_x:]
            x = K.concatenate([x, padding], axis=axis)
        elif channels_x > channels_tensor:  # Cropping
            x = x[:, :channels_tensor, ...] if axis == 1 else x[..., :channels_tensor]
        return x

    # output_shape is required by Theano. We use the shape of 'tensor' as the output shape.
    def compute_output_shape(input_shape):
        return input_shape[1]

    # Change function into Keras layer
    return Lambda(inner, output_shape=compute_output_shape)


def skipconnection(use_identity_mappings=True, name=None):
    """Skip connection of ResNet."""

    def inner(x, tensor):

        # Modifies x to be the same size as tensor.
        if use_identity_mappings:
            x = resize_x()([x, tensor])
        else:  # Activation should be linear (None)
            dim = K.ndim(x)
            axis = get_channel_axis()
            num_channels = K.int_shape(tensor)[axis]
            if dim == 2:  # batch size, channels
                x = Dense(num_channels)(x)
            elif 3 <= dim <= 5:
                x = ConvND(x, filters=num_channels, kernel_size=1, padding='same')(x)
            else:
                raise ValueError('Only 2D, 3D, 4D, and 5D tensors are supported.')

        x = Add(name=name)([tensor, x])
        return x

    return inner


def pyramid_pooling(pyramid_bins, activation='relu', conv_order='conv_first', use_batch_norm=True, upsampling=True):
    """Spatial pyramid pooling."""
    
    def inner(x):

        axis = get_channel_axis()
        shape_x = K.int_shape(x)

        # Get image size
        if axis == 1:
            img_size = shape_x[2:]
        else:
            img_size = shape_x[1:-1]
        img_size = np.array(img_size)

        # Perform pyramid pooling
        num_filters = shape_x[axis] / len(pyramid_bins)
        pooled = []
        for b_sz in pyramid_bins:
            size = tuple(img_size / b_sz)
            tmp = AveragePoolingND(x, pool_size=size)(x)
            tmp = conv(num_filters, kernel_size=1, activation=activation, conv_order=conv_order,
                       use_batch_norm=use_batch_norm)(tmp)
            if upsampling:
                tmp = UpSamplingND(tmp, size=size)(tmp)
            else:
                tmp = Flatten()(tmp)
            pooled.append(tmp)

        x = Concatenate(axis=axis)(pooled)

        return x

    return inner


def model_layer(pretrained_model, pretrained_layer_name, pretrained_keep_layers=0, input_size=(256, 256), num_input_channels=3):
    """Extracts part of a pretrained model as a layer.

    :param pretrained_model: the pretrained model.
    :param pretrained_layer_name: the name of a layer. All layers after (this layer + pretrained_keep_layers) are discarded.
    :param pretrained_keep_layers: this argument is used when the desired layer does not have an identifiable name.
    :param input_size: input size.
    :param num_input_channels: number of input channels.
    :return: a layer as part of the pretrained model.
    """

    pretrained_model.layers.pop(0)  # Remove input layer

    # Remove unwanted layers.
    for i, layer in enumerate(pretrained_model.layers):
        if layer.name != pretrained_layer_name:
            continue
        else:
            num_layers = len(pretrained_model.layers)
            for j in reversed(range(i + 1 + pretrained_keep_layers, num_layers)):
                pretrained_model.layers.pop(j)
            break

    for layer in pretrained_model.layers:
        layer.trainable = False

    # resnet.layers[-1].outbound_nodes = []
    pretrained_model.outputs = pretrained_model.layers[-1].output  # Necessary for the correct output layer

    # Create a model that can operate on x
    model = Model(inputs=pretrained_model.inputs, outputs=pretrained_model.outputs)

    # Create the desired model
    if K.image_data_format() == 'channels_first':
        inputs = Input((num_input_channels,) + input_size)
    else:
        inputs = Input(input_size + (num_input_channels,))
    outputs = model(inputs)
    return Model(inputs=inputs, outputs=outputs, name='pretrained_model')
