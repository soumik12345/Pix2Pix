# Pix2Pix

Tensorflow Implementation of the paper [Image-to-Image Translation using Conditional GANs](https://arxiv.org/abs/1611.07004) by [Philip Isola](https://arxiv.org/search/cs?searchtype=author&query=Isola%2C+P), [Jun-Yan Zhu](https://arxiv.org/search/cs?searchtype=author&query=Zhu%2C+J), [Tinghui Zhou](https://arxiv.org/search/cs?searchtype=author&query=Zhou%2C+T) and [Alexei A. Efros](https://arxiv.org/search/cs?searchtype=author&query=Efros%2C+A+A).


## Architecture

### Generator

- The Generator is a Unet-Like model with skip connections between encoder and decoder.
- Encoder Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```
- Decode Blocks is ```Conv2DTranspose -> BatchNormalization -> Dropout (optional) -> Activation (ReLU)```

![Generator Architecture](./assets/unet_like_generator.png)

### Discriminator

- PatchGAN Discriminator
- Discriminator Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```

![Discriminator Architecture](./assets/patchgan_discriminator.png)

## Loss Functions

### Generator Loss

![Generator Loss Equation](./assets/gen_loss.gif)

The Loss function can also be boiled down to

```Loss = GAN_Loss + Lambda * L1_Loss```, where GAN_Loss is Sigmoid Cross Entropy Loss and Lambda = 100 (determined by the authors)

### Discriminator Loss

The Discriminator Loss function can be written as

```Loss = disc_loss(real_images, array of ones) + disc_loss(generated_images, array of zeros)```

where `disc_loss` is Sigmoid Cross Entropy Loss.

## [Experiment 1](./Pix2Pix_Facades.ipynb)

**Dataset:** [Facades](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/facades.tar.gz)

**Generator Architecture:**

- The Generator is a Unet-Like model with skip connections between encoder and decoder.
- Encoder Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```
- Decode Blocks is ```Conv2DTranspose -> BatchNormalization -> Dropout (optional) -> Activation (ReLU)```

**Discriminator:**

- PatchGAN Discriminator
- Discriminator Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```

**Result:**

![Experiment 1 Result](./assets/exp_1_gif.gif)

## [Experiment 2](./Pix2Pix_Maps.ipynb)

**Dataset:** [Maps](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/maps.tar.gz)

**Generator Architecture:**

- The Generator is a Unet-Like model with skip connections between encoder and decoder.
- Encoder Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```
- Decode Blocks is ```Conv2DTranspose -> BatchNormalization -> Dropout (optional) -> Activation (ReLU)```

**Discriminator:**

- PatchGAN Discriminator
- Discriminator Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```

**Result:**

![Experiment 2 Result](./assets/exp_2_gif.gif)

## [Experiment 3](./Pix2Pix_Cityscapes.ipynb)

**Dataset:** [Cityscapes](https://people.eecs.berkeley.edu/~tinghuiz/projects/pix2pix/datasets/cityscapes.tar.gz)

**Generator Architecture:**

- The Generator is a Unet-Like model with skip connections between encoder and decoder.
- Encoder Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```
- Decode Blocks is ```Conv2DTranspose -> BatchNormalization -> Dropout (optional) -> Activation (ReLU)```

**Discriminator:**

- PatchGAN Discriminator
- Discriminator Block is ```Convolution -> BatchNormalization -> Activation (LeakyReLU)```

**Result:**

![Experiment 3 Result](./assets/exp_3_gif.gif)

## References

All the sources cited during building this codebase are mentioned below:

- [Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/pdf/1611.07004.pdf)
- [https://github.com/phillipi/pix2pix](https://github.com/phillipi/pix2pix)
- [Precomputed Real-Time Texture Synthesis with Markovian Generative Adversarial Networks](https://arxiv.org/abs/1604.04382)
- [Tensorflow Pix2Pix](https://github.com/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb)
- [Keras Pix2Pix](https://github.com/eriklindernoren/Keras-GAN/blob/master/pix2pix/pix2pix.py)