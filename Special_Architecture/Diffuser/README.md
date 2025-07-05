# Diffusion Model Implementation

This repository contains a comprehensive implementation of a Diffusion Model using PyTorch. The implementation demonstrates the core concepts of diffusion models through a U-Net architecture trained on the Fashion-MNIST dataset for image generation.

## What is a Diffusion Model?

A Diffusion Model is a generative machine learning model that learns to generate data by reversing a gradual noising process. The model consists of two main processes:

1. **Forward Process (Diffusion)**: Gradually adds Gaussian noise to data over T timesteps until it becomes pure noise
2. **Reverse Process (Denoising)**: Learns to remove noise step by step to generate new data from pure noise

The key insight is that if we can learn to reverse each small denoising step, we can generate new samples by starting from pure noise and iteratively denoising it.

## Architecture Overview

### U-Net Architecture
The core of our diffusion model is a U-Net architecture that predicts the noise to be removed at each timestep:

- **Input**: Noisy image + timestep embedding
- **Encoder**: Downsampling path with residual blocks
  - Conv2D(64) → ResidualBlock → 28×28
  - Conv2D(128) → ResidualBlock → 14×14  
  - Conv2D(256) → ResidualBlock → 7×7
  - Conv2D(512) → ResidualBlock + Attention → 4×4
- **Decoder**: Upsampling path with skip connections
  - ConvTranspose2D(256) + Skip → 7×7
  - ConvTranspose2D(128) + Skip → 14×14
  - ConvTranspose2D(64) + Skip → 28×28
- **Output**: Predicted noise (same shape as input image)

### Key Components

#### Time Embedding
- **Sinusoidal Positional Encoding**: Encodes timestep information
- **Dense Layers**: Projects time embedding to feature space
- **Integration**: Added to bottleneck features to condition the model

#### Attention Mechanism
- **Multi-Head Self-Attention**: Applied at the bottleneck for global context
- **Residual Connections**: Maintains gradient flow
- **Layer Normalization**: Stabilizes training

#### Residual Blocks
- **Skip Connections**: Improves gradient flow and feature preservation
- **Batch Normalization**: Stabilizes training
- **ReLU Activation**: Non-linear transformations

## Diffusion Process

### Forward Diffusion (Noise Addition)
The forward process gradually corrupts data by adding Gaussian noise:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Where:
- `β_t` is the noise schedule (increases from 0.0001 to 0.02)
- `x_t` is the noisy image at timestep t
- `x_0` is the original clean image

### Reverse Diffusion (Denoising)
The reverse process learns to remove noise:

```
p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

The model learns to predict the noise `ε_θ(x_t, t)` that was added.

### Noise Schedule
- **Linear Schedule**: β values increase linearly from 0.0001 to 0.02
- **Alpha Values**: α_t = 1 - β_t
- **Cumulative Products**: Used for efficient noise addition at any timestep

## Training Process

### Training Objective
The model is trained to predict the noise that was added to clean images:

```
L = E_{x_0, ε, t} [||ε - ε_θ(√(ᾱ_t) x_0 + √(1-ᾱ_t) ε, t)||²]
```

Where:
- `ε` is the actual noise added
- `ε_θ` is the predicted noise
- `ᾱ_t` is the cumulative product of alphas

### Training Steps
1. **Sample Batch**: Get clean images from dataset
2. **Sample Timesteps**: Random timesteps t for each image
3. **Add Noise**: Apply forward diffusion to get noisy images
4. **Predict Noise**: Use U-Net to predict the added noise
5. **Compute Loss**: MSE between actual and predicted noise
6. **Update Model**: Backpropagate and update weights

### Generation Process
1. **Start with Noise**: Sample from N(0, I)
2. **Iterative Denoising**: For t = T to 1:
   - Predict noise using the model
   - Remove predicted noise
   - Add small amount of noise (except final step)
3. **Output**: Generated image

## Key Features

- **U-Net Architecture**: Encoder-decoder with skip connections for detailed reconstruction
- **Time Conditioning**: Sinusoidal embeddings to handle different noise levels
- **Attention Mechanism**: Self-attention for capturing global dependencies
- **Residual Connections**: Better gradient flow and training stability
- **Efficient Sampling**: Reduced timesteps during inference for faster generation
- **Batch Training**: Efficient training with random timestep sampling
- **Loss Tracking**: Monitor training progress
- **Visualization**: Real-time generation during training

## Usage

### Requirements
```bash
pip install torch torchvision numpy matplotlib
```

### Running the Code
```bash
python diffusion_model.py
```

### What the Code Does
1. Loads and preprocesses Fashion-MNIST dataset
2. Defines U-Net architecture with time conditioning
3. Creates diffusion model with noise schedules
4. Trains for 50 epochs with periodic image generation
5. Displays training loss curves
6. Generates final sample images

### Output
- **During Training**: Shows generated images every 10 epochs
- **After Training**: Displays loss curves and final generated samples

## Understanding the Results

### Training Indicators
- **Decreasing Loss**: Model is learning to predict noise better
- **Image Quality**: Generated images should become more realistic over time
- **Diversity**: Model should generate varied samples, not mode collapse

### Common Patterns
- **Early Training**: Generated images are very noisy
- **Mid Training**: Basic shapes and patterns emerge
- **Late Training**: Detailed, realistic images with good diversity

## Key Parameters

- `T = 1000`: Number of diffusion timesteps
- `beta_start = 0.0001`: Initial noise level
- `beta_end = 0.02`: Final noise level
- `epochs = 50`: Training epochs
- `batch_size = 32`: Training batch size
- `learning_rate = 0.0001`: Adam optimizer learning rate
- `time_embedding_dim = 128`: Dimension of time embeddings

## Customization

### Architecture Modifications
- **Model Size**: Adjust filter numbers in U-Net layers
- **Attention**: Add more attention blocks or change attention heads
- **Skip Connections**: Modify skip connection patterns
- **Activation Functions**: Experiment with different activations

### Training Modifications
- **Noise Schedule**: Try cosine or other noise schedules
- **Loss Function**: Experiment with different loss formulations
- **Sampling Strategy**: Modify timestep sampling during training
- **Data Augmentation**: Add augmentations for better generalization

### Generation Modifications
- **Sampling Steps**: Adjust number of denoising steps
- **Sampling Schedule**: Use different timestep schedules
- **Guidance**: Add classifier or classifier-free guidance
- **Conditional Generation**: Add class or text conditioning

## Advanced Features

### Implemented
- **Efficient U-Net**: Optimized architecture for image generation
- **Time Conditioning**: Proper timestep embedding and integration
- **Attention Mechanism**: Self-attention for global context
- **Fast Sampling**: Reduced timesteps for quicker generation

### Possible Extensions
- **Classifier-Free Guidance**: For better sample quality
- **Conditional Generation**: Class or text-conditioned generation
- **Different Datasets**: Extend to CIFAR-10, CelebA, etc.
- **Higher Resolution**: Scale to larger image sizes
- **Latent Diffusion**: Work in latent space for efficiency

## Common Issues and Solutions

1. **Slow Training**: Reduce model size or use mixed precision training
2. **Poor Sample Quality**: Increase model capacity or training time
3. **Mode Collapse**: Check noise schedule and model architecture
4. **Memory Issues**: Reduce batch size or model complexity
5. **Unstable Training**: Adjust learning rate or add gradient clipping

## Comparison with GANs

### Advantages of Diffusion Models
- **Training Stability**: More stable than GAN training
- **Sample Quality**: Often produces higher quality samples
- **Mode Coverage**: Better coverage of data distribution
- **Theoretical Foundation**: Strong mathematical foundation

### Disadvantages
- **Generation Speed**: Slower than GANs (multiple denoising steps)
- **Computational Cost**: More expensive training and inference
- **Memory Usage**: Requires storing noise schedules

This implementation provides a solid foundation for understanding diffusion models and can be extended for more complex applications and datasets.
