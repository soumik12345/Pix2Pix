from .blocks import *
from config import *
from tensorflow import random_normal_initializer
from tensorflow.keras.layers import (
    Conv2DTranspose, Concatenate, Input,
    concatenate, BatchNormalization, ZeroPadding2D, LeakyReLU
)
from tensorflow.keras.models import Model


def Generator():
    down_stack = [
        downsample_block(64, 4, batch_norm=False),
        downsample_block(128, 4),
        downsample_block(256, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
        downsample_block(512, 4),
    ]
    up_stack = [
        upsample_block(512, 4, dropout=True),
        upsample_block(512, 4, dropout=True),
        upsample_block(512, 4, dropout=True),
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



def Discriminator():
    initializer = random_normal_initializer(0., 0.02)
    inp = Input(shape=[None, None, 3], name='input_image')
    tar = Input(shape=[None, None, 3], name='target_image')
    x = concatenate([inp, tar])
    down1 = downsample_block(64, 4, False)(x)
    down2 = downsample_block(128, 4)(down1)
    down3 = downsample_block(256, 4)(down2)
    zero_pad1 = ZeroPadding2D()(down3)
    conv = Conv2D(
        512, 4, strides=1,
        kernel_initializer=initializer,
        use_bias=False
    )(zero_pad1)
    batchnorm1 = BatchNormalization()(conv)
    leaky_relu = LeakyReLU()(batchnorm1)
    zero_pad2 = ZeroPadding2D()(leaky_relu)
    last = Conv2D(
        1, 4, strides=1,
        kernel_initializer=initializer
    )(zero_pad2)
    return Model(inputs=[inp, tar], outputs=last)