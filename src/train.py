from tensorflow.keras.optimizers import Adam


def get_optimizers():
    generator_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(2e-4, beta_1=0.5)
    return discriminator_optimizer, generator_optimizer