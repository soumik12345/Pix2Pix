import tensorflow as tf
from config import *
from matplotlib import pyplot as plt


def load(image_file):
    image = tf.io.read_file(image_file)
    image = tf.image.decode_jpeg(image)
    w = tf.shape(image)[1]
    w = w // 2
    real_image = image[:, : w, :]
    input_image = image[:, w :, :]
    input_image = tf.cast(input_image, tf.float32)
    real_image = tf.cast(real_image, tf.float32)
    return input_image, real_image


def visualize(image_file, use_jitter = False):
    input_image, real_image = load(image_file)
    if use_jitter:
        input_image, real_image = random_jitter(input_image, real_image)
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (16, 16))
    plt.setp(axes.flat, xticks = [], yticks = [])
    for i, ax in enumerate(axes.flat):
        if i % 2 == 0:
            ax.imshow(input_image.numpy() / 255.0)
            ax.set_xlabel('Input_Image')
        else:
            ax.imshow(real_image.numpy() / 255.0)
            ax.set_xlabel('Real_Image')
    plt.show()


def resize(input_image, real_image, height, width):
    input_image = tf.image.resize(
        input_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    real_image = tf.image.resize(
        real_image, [height, width],
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    return input_image, real_image


def random_crop(input_image, real_image):
    stacked_image = tf.stack([input_image, real_image], axis=0)
    cropped_image = tf.image.random_crop(
        stacked_image,
        size=[2, IMG_HEIGHT, IMG_WIDTH, 3]
    )
    return cropped_image[0], cropped_image[1]


def normalize(input_image, real_image):
    input_image = (input_image / 127.5) - 1
    real_image = (real_image / 127.5) - 1
    return input_image, real_image


@tf.function()
def random_jitter(input_image, real_image):
    input_image, real_image = resize(input_image, real_image, 286, 286)
    input_image, real_image = random_crop(input_image, real_image)
    if tf.random.uniform(()) > 0.5:
        input_image = tf.image.flip_left_right(input_image)
        real_image = tf.image.flip_left_right(real_image)
    return input_image, real_image