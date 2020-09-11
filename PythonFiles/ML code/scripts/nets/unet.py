from keras.models import Model
from keras.layers import Input, Reshape, Dropout, Activation, Permute, Concatenate, GaussianNoise
from keras.optimizers import Adam
from keras import backend as K

import numpy as np

from nets.nets_utils import conv, ConvND, UpSamplingND, MaxPoolingND
from utils.kerasutils import get_channel_axis
from nets.custom_losses import exp_categorical_crossentropy

"""Building U-Net."""

__author__ = 'Ken C. L. Wong'


def build_net(
        base_num_filters=16,
        convs_per_depth=2,
        kernel_size=3,
        num_classes=2,
        image_size=(256, 256),
        kernel_cc_weight=0.0,
        activation='relu',
        dropout_rate=0.0,
        optimizer=Adam(lr=1e-4),
        net_depth=4,
        conv_order='conv_first',
        noise_std=None,
        loss=exp_categorical_crossentropy(exp=1.0)
):
    """Builds a U-Net.
    """

    image_size = tuple(image_size)
    if any(sz / 2 ** net_depth == 0 for sz in image_size):
        raise RuntimeError('Mismatch between image size and network depth.')

    # Preset parameters of different functions
    func = UnetFunctions(base_num_filters=base_num_filters, convs_per_depth=convs_per_depth, kernel_size=kernel_size,
                         activation=activation, conv_order=conv_order)

    channels_first = K.image_data_format() == 'channels_first'

    if channels_first:
        inputs = Input((1,) + image_size)
    else:
        inputs = Input(image_size + (1,))

    x = inputs

    if noise_std is not None:
        x = GaussianNoise(noise_std)(x)

    # Contracting path
    for depth in range(net_depth):
        x = func.contract(depth,kernel_cc_weight=kernel_cc_weight)(x)

    # Deepest layer
    num_filters = base_num_filters * 2 ** net_depth
    x = Dropout(dropout_rate)(x) if dropout_rate else x
    x = func.conv_depth(num_filters=num_filters, name='encode')(x)

    # Expanding path
    for depth in reversed(range(net_depth)):
        x = func.expand(depth, kernel_cc_weight=kernel_cc_weight)(x)

    # Segmentation layer and loss function related
    x = ConvND(x, filters=num_classes, kernel_size=1, activation=None, name='segmentation')(x)
    if channels_first:
        new_shape = tuple(range(1, K.ndim(x)))
        new_shape = new_shape[1:] + new_shape[:1]
        x = Permute(new_shape)(x)
    # x = Reshape((np.prod(image_size), num_classes))(x)
    x = Activation('softmax')(x)

    model = Model(inputs=inputs, outputs=x)
    model.compile(optimizer=optimizer, loss=loss)

    return model


class UnetFunctions(object):
    """Provides U-net related functions."""

    def __init__(self, base_num_filters=16, convs_per_depth=2, kernel_size=3, activation='relu',
                 conv_order='conv_first', use_batch_norm=True, dilation_rate=1):
        self.base_num_filters = base_num_filters
        self.convs_per_depth = convs_per_depth
        self.kernel_size = kernel_size
        self.activation = activation
        self.conv_order = conv_order
        self.use_batch_norm = use_batch_norm
        self.dilation_rate = dilation_rate
        self.contr_tensors = dict()

    def local_conv(self, num_filters, dilation_rate, name=None, kernel_cc_weight=0.0):
        return conv(num_filters=num_filters, kernel_size=self.kernel_size, activation=self.activation,
                    conv_order=self.conv_order, use_batch_norm=self.use_batch_norm, name=name,
                    dilation_rate=dilation_rate, kernel_cc_weight=kernel_cc_weight)

    def conv_depth(self, num_filters, dilation_rate=None, name=None, kernel_cc_weight=0.0):
        """conv block for a depth."""

        if dilation_rate is None:
            dilation_rate = self.dilation_rate

        def composite_layer(x):
            for c in range(self.convs_per_depth):
                if c != self.convs_per_depth - 1:
                    x = self.local_conv(num_filters=num_filters, dilation_rate=dilation_rate, kernel_cc_weight= kernel_cc_weight)(x)
                else:
                    x = self.local_conv(num_filters=num_filters, dilation_rate=dilation_rate, name=name, kernel_cc_weight=kernel_cc_weight)(x)
            return x

        return composite_layer

    def contract(self, depth, kernel_cc_weight=0.0):
        """Returns a contraction layer."""

        def composite_layer(x):
            name = 'contr_%d' % depth
            num_filters = self.base_num_filters * 2 ** depth
            x = self.conv_depth(num_filters=num_filters, name=name, kernel_cc_weight=kernel_cc_weight)(x)
            self.contr_tensors[depth] = x
            x = MaxPoolingND(x)(x)
            return x

        return composite_layer

    def expand(self, depth, kernel_cc_weight=0.0):
        """Returns an expansion layer."""

        def composite_layer(x):
            """
            Arguments:
            x: tensor from the previous layer.
            """
            name = 'expand_%d' % depth
            num_filters = self.base_num_filters * 2 ** depth
            x = UpSamplingND(x)(x)
            x = Concatenate(axis=get_channel_axis(), name='concat_%d' % depth)([x, self.contr_tensors[depth]])
            x = self.conv_depth(num_filters=num_filters, name=name, kernel_cc_weight=kernel_cc_weight)(x)
            return x

        return composite_layer
