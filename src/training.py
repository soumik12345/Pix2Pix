import time
from config import *
from .losses import *
from os.path import join
import tensorflow as tf
from tqdm import tqdm
from matplotlib import pyplot as plt
from tensorflow.keras.optimizers import Adam


def get_optimizers():
    generator_optimizer = Adam(2e-4, beta_1=0.5)
    discriminator_optimizer = Adam(2e-4, beta_1=0.5)
    return discriminator_optimizer, generator_optimizer


def get_checkpoint(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer,
    checkpoint_dir='./training_checkpoints'):
    
    checkpoint_prefix = join(checkpoint_dir, 'ckpt')
    checkpoint = tf.train.Checkpoint(
        generator_optimizer=generator_optimizer,
        discriminator_optimizer=discriminator_optimizer,
        generator=generator, discriminator=discriminator
    )
    return checkpoint, checkpoint_prefix


def generate_images(model, test_input, tar):
    prediction = model(test_input, training=True)
    plt.figure(figsize=(15,15))
    display_list = [test_input[0], tar[0], prediction[0]]
    title = ['Input Image', 'Ground Truth', 'Predicted Image']
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.title(title[i])
        plt.imshow(display_list[i] * 0.5 + 0.5)
        plt.axis('off')
    plt.show()


def train(
    discriminator, generator,
    discriminator_optimizer,
    generator_optimizer,
    train_dataset, test_dataset,
    checkpoint, checkpoint_prefix):

    generator_loss_history, discriminator_loss_history = [], []

    @tf.function
    def train_step(input_image, target):
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            gen_output = generator(input_image, training=True)
            
            disc_real_output = discriminator([input_image, target], training=True)
            disc_generated_output = discriminator([input_image, gen_output], training=True)
            
            gen_loss = generator_loss(disc_generated_output, gen_output, target)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            
            generator_loss_history.append(gen_loss)
            discriminator_loss_history.append(disc_loss)
        
        generator_gradients = gen_tape.gradient(
            gen_loss,
            generator.trainable_variables
        )
        discriminator_gradients = disc_tape.gradient(
            disc_loss,
            discriminator.trainable_variables
        )
        generator_optimizer.apply_gradients(
            zip(
                generator_gradients,
                generator.trainable_variables
            )
        )
        discriminator_optimizer.apply_gradients(
            zip(
                discriminator_gradients,
                discriminator.trainable_variables
            )
        )


    def fit(train_ds, test_ds, epochs):
        for epoch in tqdm(range(epochs)):
            start = time.time()
            # Train
            for input_image, target in train_ds:
                train_step(input_image, target)
            for example_input, example_target in test_ds.take(1):
                generate_images(generator, example_input, example_target)
            # saving (checkpoint) the model every 20 epochs
            if (epoch + 1) % 20 == 0:
                checkpoint.save(file_prefix = checkpoint_prefix)
            print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1, time.time()-start))
    

    fit(train_dataset, test_dataset, EPOCHS)

    return generator_loss_history, discriminator_loss_history