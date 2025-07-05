# Simple GAN Implementation

This repository contains a clean and simple implementation of a Generative Adversarial Network (GAN) using TensorFlow/Keras. The implementation demonstrates the core concepts of GANs through a straightforward example using the Fashion-MNIST dataset.

## What is a GAN?

A Generative Adversarial Network (GAN) is a machine learning architecture consisting of two neural networks competing against each other in a zero-sum game framework. The two networks are:

1. **Generator (G)**: Creates fake data that resembles real data
2. **Discriminator (D)**: Distinguishes between real and fake data

The generator tries to create data so realistic that the discriminator cannot tell it's fake, while the discriminator tries to get better at detecting fake data. This adversarial training process leads to the generator producing increasingly realistic data.

## Architecture Overview

### Generator Network
The generator takes random noise as input and transforms it into realistic images:
- **Input**: Random noise vector (latent dimension: 32)
- **Architecture**: 
  - Dense layer (128 units) → BatchNormalization → LeakyReLU
  - Dense layer (256 units) → BatchNormalization → LeakyReLU  
  - Dense layer (512 units) → BatchNormalization → LeakyReLU
  - Dense layer (image_size) → Tanh activation → Reshape to image
- **Output**: Generated image (28×28×1 for Fashion-MNIST)

### Discriminator Network
The discriminator classifies images as real or fake:
- **Input**: Image (28×28×1)
- **Architecture**:
  - Flatten → Dense layer (512 units) → LeakyReLU
  - Dense layer (256 units) → LeakyReLU
  - Dense layer (128 units) → LeakyReLU
  - Dense layer (1 unit) → Sigmoid activation
- **Output**: Probability that input image is real (0-1)

## Training Process

The GAN training follows this alternating process:

1. **Train Discriminator**:
   - Get a batch of real images from the dataset
   - Generate fake images using the generator
   - Train discriminator to classify real images as 1 and fake images as 0

2. **Train Generator**:
   - Generate fake images
   - Train the generator (via the combined model) to fool the discriminator
   - The generator tries to make the discriminator classify fake images as real (label 1)

3. **Repeat**: This adversarial process continues for multiple epochs

## Key Features

- **Data Preprocessing**: Images are normalized to [-1, 1] range for stable GAN training
- **Loss Function**: Binary crossentropy for both generator and discriminator
- **Optimizer**: Adam optimizer with learning rate 0.0002 and specific beta values (0.5, 0.9)
- **Batch Training**: Uses half-batch training for discriminator (real + fake samples)
- **Visualization**: Real-time visualization of generated images during training
- **Loss Tracking**: Monitors both generator and discriminator losses

## Usage

### Requirements
```bash
pip install tensorflow numpy matplotlib
```

### Running the Code
```bash
python simple_gan.py
```

### What the Code Does
1. Loads and preprocesses Fashion-MNIST dataset
2. Defines generator and discriminator architectures
3. Creates the GAN training framework
4. Trains for 100 epochs with visualization every epoch
5. Displays loss curves for both networks

### Output
- **During Training**: Shows generated images alongside real images for comparison
- **After Training**: Displays loss curves showing the training progress of both networks

## Understanding the Results

- **Good Training**: Both generator and discriminator losses should stabilize (not necessarily converge)
- **Mode Collapse**: If generator loss becomes very low while discriminator loss increases, the generator might be producing limited variety
- **Training Instability**: Large oscillations in losses indicate unstable training

## Key Parameters

- `EPOCHS = 100`: Number of training epochs
- `latent_dim = 32`: Dimension of input noise vector
- `steps_per_epoch = 50`: Training steps per epoch
- `batch_size = 128`: Batch size for training
- `learning_rate = 0.0002`: Learning rate for both networks

## Customization

You can easily modify this implementation:
- **Dataset**: Change the dataset by modifying the data loading section
- **Architecture**: Adjust network layers in `generator_fc()` and `discriminator_fc()`
- **Hyperparameters**: Modify learning rates, batch sizes, or training epochs
- **Loss Functions**: Experiment with different loss functions for improved training

## Common Issues and Solutions

1. **Mode Collapse**: Reduce learning rates or add noise to discriminator training
2. **Training Instability**: Adjust the balance between generator and discriminator training
3. **Poor Image Quality**: Increase model capacity or training time
4. **Memory Issues**: Reduce batch size or model complexity

This implementation provides a solid foundation for understanding GANs and can be extended for more complex applications and datasets.
