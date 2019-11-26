import tensorflow as tf
from tensorflow import image
from tensorflow.keras.layers import (BatchNormalization, Conv2D, Dropout,
                                     Flatten, MaxPooling2D, concatenate)


def build_time_diff_module(input_layer, output_shape):
    time_diff = input_layer[:, :, 1:]-input_layer[:, :, :-1]
    time_diff = image.resize(
        time_diff, list(output_shape[:-1]),
        method=image.ResizeMethod.NEAREST_NEIGHBOR)
    return time_diff


def build_start_time_diff_module(input_layer, output_shape):
    start_time = input_layer[:, :, 0]
    start_time = tf.tile(start_time[:, :, tf.newaxis],
                         [1, 1, output_shape[0]-1, 1])
    start_time_diff = input_layer[:, :, 1:]-start_time
    start_time_diff = image.resize(
        start_time_diff, list(output_shape[:-1]),
        method=image.ResizeMethod.NEAREST_NEIGHBOR)

    return start_time_diff


def build_conv_module(input_layer):
    conv_1 = Conv2D(32, (3, 3), activation='relu',
                    strides=1, padding='same')(input_layer)

    pooling_1 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_1)

    conv_2 = Conv2D(32, (3, 3), activation='relu',
                    strides=1, padding='same')(pooling_1)

    pooling_2 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_2)

    bn_1 = BatchNormalization()(pooling_2)
    dropout_1 = Dropout(0.5)(bn_1)

    conv_3 = Conv2D(64, (3, 3), activation='relu',
                    strides=1, padding='same')(dropout_1)

    pooling_3 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_3)

    conv_4 = Conv2D(64, (3, 3), activation='relu',
                    strides=1, padding='same')(pooling_3)

    pooling_4 = MaxPooling2D(pool_size=(3, 3), strides=2)(conv_4)

    bn_2 = BatchNormalization()(pooling_4)
    dropout_2 = Dropout(0.5)(bn_2)

    flatten = Flatten()(dropout_2)

    return flatten
