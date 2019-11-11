from tensorflow import random_normal_initializer
from tensorflow.keras import Sequential
from tensorflow.keras.layers import (
    Conv2D, LeakyReLU,
    BatchNormalization, ReLU,
    Conv2DTranspose, Dropout
)


def downsample_block(filters, kernel_size, batch_norm = True):
    initializer = random_normal_initializer(0.0, 0.02)
    block = Sequential()
    block.add(
        Conv2D(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    if batch_norm:
        block.add(BatchNormalization())
    block.add(LeakyReLU())
    return block


def upsample_block(filters, kernel_size, dropout = False):
    initializer = random_normal_initializer(0.0, 0.02)
    block = Sequential()
    block.add(
        Conv2DTranspose(
            filters, kernel_size,
            strides=2, padding='same',
            kernel_initializer=initializer, use_bias=False
        )
    )
    block.add(BatchNormalization())
    if dropout:
        block.add(Dropout(0.5))
    block.add(ReLU())
    return block