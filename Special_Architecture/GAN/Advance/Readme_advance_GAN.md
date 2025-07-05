# Advanced PyTorch DCGAN Implementation

## Overview

This repository contains an advanced Deep Convolutional Generative Adversarial Network (DCGAN) implementation using PyTorch. The model is designed to generate high-quality synthetic images from the Fashion-MNIST dataset, demonstrating significant improvements over traditional fully-connected GAN architectures.

## Architecture

### Generator Network
The generator uses a series of transposed convolutions (deconvolutions) to transform random noise into realistic images:

- **Input**: 100-dimensional latent vector (noise)
- **Architecture**: 5 transposed convolutional layers with batch normalization and ReLU activations
- **Output**: 64x64 grayscale images
- **Key Features**:
  - Progressive upsampling from 1x1 to 64x64
  - Batch normalization for training stability
  - Tanh activation for final output (normalized to [-1, 1])

### Discriminator Network
The discriminator uses convolutional layers to classify images as real or fake:

- **Input**: 64x64 grayscale images
- **Architecture**: 5 convolutional layers with batch normalization and LeakyReLU activations
- **Output**: Single probability score (real vs fake)
- **Key Features**:
  - Progressive downsampling from 64x64 to 1x1
  - LeakyReLU activations to prevent gradient vanishing
  - No batch normalization in the first layer

## Advanced Features

### Training Stability Enhancements
1. **Label Smoothing**: Reduces overconfidence by smoothing real labels (0.9 instead of 1.0)
2. **Noise Injection**: Adds small amounts of noise to real images during training
3. **Proper Weight Initialization**: Uses normal distribution initialization for stable training
4. **Adam Optimizer**: With carefully tuned learning rates and beta parameters

### Technical Specifications
- **Batch Size**: 128
- **Learning Rate**: 0.0002 for both generator and discriminator
- **Beta Parameters**: (0.5, 0.999) for Adam optimizer
- **Image Size**: 64x64 pixels (upscaled from 28x28 Fashion-MNIST)
- **Latent Dimension**: 100
- **Feature Maps**: 64 base feature maps with progressive scaling

## Dataset

The model trains on the Fashion-MNIST dataset, which contains:
- 60,000 training images
- 10 categories of fashion items
- Original size: 28x28, upscaled to 64x64 for better generation quality
- Normalized to [-1, 1] range

## Training Process

### Loss Function
- **Binary Cross-Entropy Loss** for both generator and discriminator
- **Adversarial Training**: Generator tries to fool discriminator, discriminator tries to distinguish real from fake

### Training Loop
1. **Discriminator Training**:
   - Train on real images with smoothed labels
   - Train on generated (fake) images
   - Update discriminator weights

2. **Generator Training**:
   - Generate fake images from random noise
   - Train generator to fool discriminator
   - Update generator weights

### Monitoring and Visualization
- Real-time loss tracking for both networks
- Periodic image generation for visual progress monitoring
- Training statistics output every 50 iterations
- Image grid visualization every 10 epochs

## Usage

### Requirements
```bash
pip install torch torchvision matplotlib numpy
```

### Running the Model
```bash
python advanced_pytorch_gan.py
```

### Output Files
- `generator_final.pth`: Trained generator model weights
- `discriminator_final.pth`: Trained discriminator model weights
- Real-time visualization of training progress
- Loss curves and generated image samples

## Key Improvements Over Simple GAN

1. **Convolutional Architecture**: Uses CNN layers instead of fully-connected layers for better spatial understanding
2. **Batch Normalization**: Improves training stability and convergence
3. **Advanced Training Techniques**: Label smoothing, noise injection, and proper initialization
4. **Higher Resolution**: Generates 64x64 images instead of 28x28
5. **Better Optimization**: Carefully tuned hyperparameters and learning rates
6. **Comprehensive Monitoring**: Detailed loss tracking and visualization

## Model Performance

The advanced DCGAN demonstrates:
- **Faster Convergence**: Stable training within 10-50 epochs
- **Higher Quality Images**: Sharp, realistic fashion item generation
- **Training Stability**: Reduced mode collapse and training oscillations
- **Scalability**: Architecture can be easily extended to higher resolutions

## Architecture Details

### Generator Layer Progression
```
Input: (100,) → Reshape: (100, 1, 1)
Layer 1: (100, 1, 1) → (512, 4, 4)    # ConvTranspose2d + BatchNorm + ReLU
Layer 2: (512, 4, 4) → (256, 8, 8)    # ConvTranspose2d + BatchNorm + ReLU
Layer 3: (256, 8, 8) → (128, 16, 16)  # ConvTranspose2d + BatchNorm + ReLU
Layer 4: (128, 16, 16) → (64, 32, 32) # ConvTranspose2d + BatchNorm + ReLU
Layer 5: (64, 32, 32) → (1, 64, 64)   # ConvTranspose2d + Tanh
```

### Discriminator Layer Progression
```
Input: (1, 64, 64)
Layer 1: (1, 64, 64) → (64, 32, 32)   # Conv2d + LeakyReLU
Layer 2: (64, 32, 32) → (128, 16, 16) # Conv2d + BatchNorm + LeakyReLU
Layer 3: (128, 16, 16) → (256, 8, 8)  # Conv2d + BatchNorm + LeakyReLU
Layer 4: (256, 8, 8) → (512, 4, 4)    # Conv2d + BatchNorm + LeakyReLU
Layer 5: (512, 4, 4) → (1, 1, 1)      # Conv2d + Sigmoid
```

## Future Enhancements

Potential improvements for even better performance:
1. **Progressive Growing**: Gradually increase image resolution during training
2. **Spectral Normalization**: Further improve training stability
3. **Self-Attention**: Add attention mechanisms for better feature learning
4. **Conditional Generation**: Generate specific types of fashion items
5. **Higher Resolution**: Scale to 128x128 or 256x256 images

## References

- Radford, A., Metz, L., & Chintala, S. (2015). Unsupervised Representation Learning with Deep Convolutional Generative Adversarial Networks
- Goodfellow, I., et al. (2014). Generative Adversarial Nets
- Fashion-MNIST Dataset: https://github.com/zalandoresearch/fashion-mnist

## License

This project is open source and available under the MIT License.
