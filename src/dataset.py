import tensorflow as tf
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


def visualize(image_file):
    input_image, real_image = load(image_file)
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