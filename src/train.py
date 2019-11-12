from os.path import join
from tensorflow.train import Checkpoint
from tensorflow.keras.optimizers import Adam


def get_optimizers():
    generator_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(2e-4, beta_1=0.5)
    return discriminator_optimizer, generator_optimizer


def get_checkpoint(discriminator_optimizer, generator_optimizer, checkpoint_dir='./training_checkpoints'):
    checkpoint_prefix = join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator, discriminator=discriminator
    )
    return checkpoint