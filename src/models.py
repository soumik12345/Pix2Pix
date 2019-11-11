from .blocks import *
from config import *
from tensorflow import random_normal_initializer
from tensorflow.keras.layers import Conv2DTranspose, Concatenate, Input
from tensorflow.keras.models import Model


def Generator():
    down_stack = [
        downsample_block(64, 4, apply_batchnorm=False),
        downsample_block(128, 4),
        downsample_block(256, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
    ]
    up_stack = [
        upsample_block(512, 4, apply_dropout=True),
        upsample_block(512, 4, apply_dropout=True),
        upsample_block(512, 4, apply_dropout=True),
        upsample_block(512, 4),
        upsample_block(256, 4),
        upsample_block(128, 4),
        upsample_block(64, 4),
    ]
    initializer = random_normal_initializer(0., 0.02)
    last = Conv2DTranspose(
        OUTPUT_CHANNELS, 4,
        strides=2, padding='same',
        kernel_initializer=initializer, activation='tanh'
    )
    concat = Concatenate()
    inputs = Input(shape=[None,None,3])
    x = inputs
    skips = []
    for down in down_stack:
        x = down(x)
        skips.append(x)
    skips = reversed(skips[:-1])
    for up, skip in zip(up_stack, skips):
        x = up(x)
        x = concat([x, skip])
    x = last(x)
    return Model(inputs=inputs, outputs=x)