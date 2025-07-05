# ResNet (Residual Network) Implementation for MNIST

A comprehensive, educational implementation of ResNet (Residual Network) for MNIST digit classification, based on the groundbreaking paper "Deep Residual Learning for Image Recognition" by He et al.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [How ResNet Works](#how-resnet-works)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Model Configurations](#model-configurations)
- [Training](#training)
- [Key Components](#key-components)
- [Technical Details](#technical-details)
- [Results](#results)

## Overview

ResNet (Residual Network) revolutionized deep learning by introducing skip connections (residual connections) that allow training of very deep neural networks. The key innovation is the residual block that learns residual mappings instead of direct mappings, solving the vanishing gradient problem and enabling networks with hundreds of layers.

**Key Innovations:**
- **Skip Connections**: Direct connections that bypass one or more layers
- **Residual Learning**: Learning F(x) = H(x) - x instead of H(x) directly
- **Deep Architecture**: Enables training of very deep networks (50, 101, 152+ layers)
- **Batch Normalization**: Integrated throughout the network for stable training

## Architecture

### ResNet Block Types

ResNet uses two main types of building blocks:

#### 1. Basic Block (ResNet-18, ResNet-34)
```
Input (x)
    ↓
┌─────────────────────┐
│   3×3 Conv, BN      │ ← First convolution
│   ReLU              │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   3×3 Conv, BN      │ ← Second convolution
└─────────────────────┘
    ↓
    + ←─────────────────── Skip Connection (Identity or 1×1 Conv)
    ↓
┌─────────────────────┐
│   ReLU              │ ← Final activation
└─────────────────────┘
    ↓
  Output
```

#### 2. Bottleneck Block (ResNet-50, ResNet-101, ResNet-152)
```
Input (x)
    ↓
┌─────────────────────┐
│   1×1 Conv, BN      │ ← Dimension reduction
│   ReLU              │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   3×3 Conv, BN      │ ← Main computation
│   ReLU              │
└─────────────────────┘
    ↓
┌─────────────────────┐
│   1×1 Conv, BN      │ ← Dimension expansion
└─────────────────────┘
    ↓
    + ←─────────────────── Skip Connection (Identity or 1×1 Conv)
    ↓
┌─────────────────────┐
│   ReLU              │ ← Final activation
└─────────────────────┘
    ↓
  Output
```

### Complete ResNet-18 Architecture for MNIST

```
Input Image (28×28×1)
        ↓
┌─────────────────────────────────────────┐
│ Initial Convolution                     │
│ 7×7 Conv, stride=2, BN, ReLU          │ ← (28×28×1) → (14×14×64)
│ 3×3 MaxPool, stride=2                 │ ← (14×14×64) → (7×7×64)
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Layer 1: 2 × Basic Blocks              │
│ [3×3 Conv(64), BN, ReLU] × 2          │ ← (7×7×64) → (7×7×64)
│ Skip connections                        │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Layer 2: 2 × Basic Blocks              │
│ [3×3 Conv(128), BN, ReLU] × 2         │ ← (7×7×64) → (4×4×128)
│ Skip connections, stride=2 first block │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Layer 3: 2 × Basic Blocks              │
│ [3×3 Conv(256), BN, ReLU] × 2         │ ← (4×4×128) → (2×2×256)
│ Skip connections, stride=2 first block │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Layer 4: 2 × Basic Blocks              │
│ [3×3 Conv(512), BN, ReLU] × 2         │ ← (2×2×256) → (1×1×512)
│ Skip connections, stride=2 first block │
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Global Average Pooling                  │
│ AdaptiveAvgPool2d(1×1)                 │ ← (1×1×512) → (512,)
└─────────────────────────────────────────┘
        ↓
┌─────────────────────────────────────────┐
│ Classification Head                     │
│ Linear(512 → 10)                       │ ← (512,) → (10,)
└─────────────────────────────────────────┘
        ↓
    Output Classes (0-9)
```

## How ResNet Works

### 1. **The Vanishing Gradient Problem**
In very deep networks, gradients become exponentially small as they propagate backward, making it difficult to train deep layers effectively.

### 2. **Residual Learning Solution**
Instead of learning the desired mapping H(x) directly, ResNet learns the residual F(x) = H(x) - x:
- **Traditional**: H(x) = desired_output
- **ResNet**: F(x) = H(x) - x, so H(x) = F(x) + x

### 3. **Skip Connections**
```
Mathematical Representation:
y = F(x, {Wi}) + x

Where:
- x: input to the residual block
- F(x, {Wi}): residual mapping to be learned
- Wi: weights of the i-th layer
- y: output of the residual block
```

### 4. **Benefits of Residual Learning**
- **Gradient Flow**: Skip connections provide direct paths for gradients
- **Identity Mapping**: If optimal function is close to identity, easier to learn F(x) ≈ 0
- **Deep Training**: Enables training of networks with 50+ layers
- **Performance**: Better accuracy with increased depth

## Implementation Details

### Key Design Choices

1. **Block Configuration**:
   - **ResNet-18**: [2, 2, 2, 2] Basic Blocks
   - **ResNet-34**: [3, 4, 6, 3] Basic Blocks
   - **ResNet-50**: [3, 4, 6, 3] Bottleneck Blocks

2. **MNIST Adaptations**:
   - Input channels: 1 (grayscale)
   - Smaller initial kernel: 7×7 (suitable for 28×28 images)
   - Output classes: 10 (digits 0-9)

3. **Normalization**: Batch Normalization after each convolution
4. **Activation**: ReLU activation functions
5. **Initialization**: He initialization for convolutional layers

### Skip Connection Implementation

```python
# Identity mapping (same dimensions)
if self.downsample is None:
    out += x  # Direct addition

# Projection mapping (different dimensions)
else:
    out += self.downsample(x)  # 1×1 conv + BN
```

### Downsampling Strategy

When spatial dimensions change or channels increase:
```python
downsample = nn.Sequential(
    nn.Conv2d(in_channels, out_channels * expansion, 
              kernel_size=1, stride=stride, bias=False),
    nn.BatchNorm2d(out_channels * expansion)
)
```

## Usage

### Basic Usage

```python
from sample_resnet import resnet18, MNISTDataLoader, ResNetTrainer

# Create model
model = resnet18(num_classes=10, input_channels=1)

# Load data
data_loader = MNISTDataLoader(batch_size=128)
train_loader, test_loader = data_loader.get_data_loaders()

# Train model
trainer = ResNetTrainer(model, device='cuda')
history = trainer.train(train_loader, test_loader, num_epochs=10)

# Test model
results = trainer.test(test_loader)
```

### Advanced Usage

```python
# Create different ResNet variants
model_18 = resnet18()    # 11.7M parameters
model_34 = resnet34()    # 21.8M parameters
model_50 = resnet50()    # 25.6M parameters

# Custom training parameters
history = trainer.train(
    train_loader=train_loader,
    val_loader=test_loader,
    num_epochs=20,
    learning_rate=0.001,
    weight_decay=1e-4
)

# Feature extraction
features = model.get_feature_maps(input_tensor)
```

## Model Configurations

| Model | Layers | Blocks | Parameters | MNIST Accuracy |
|-------|--------|--------|------------|----------------|
| ResNet-18 | 18 | [2,2,2,2] | ~11.7M | >99% |
| ResNet-34 | 34 | [3,4,6,3] | ~21.8M | >99% |
| ResNet-50 | 50 | [3,4,6,3] | ~25.6M | >99% |
| ResNet-101 | 101 | [3,4,23,3] | ~44.5M | >99% |
| ResNet-152 | 152 | [3,8,36,3] | ~60.2M | >99% |

### Layer-wise Output Dimensions (ResNet-18 on MNIST)

| Layer | Input Size | Output Size | Parameters |
|-------|------------|-------------|------------|
| Input | (1, 28, 28) | (1, 28, 28) | 0 |
| Conv1 + Pool | (1, 28, 28) | (64, 7, 7) | 3,136 |
| Layer1 | (64, 7, 7) | (64, 7, 7) | 147,456 |
| Layer2 | (64, 7, 7) | (128, 4, 4) | 525,312 |
| Layer3 | (128, 4, 4) | (256, 2, 2) | 2,097,152 |
| Layer4 | (256, 2, 2) | (512, 1, 1) | 8,388,608 |
| AvgPool + FC | (512, 1, 1) | (10,) | 5,130 |

## Training

### Training Pipeline

1. **Data Preprocessing**:
   ```python
   transforms.Compose([
       transforms.RandomRotation(10),
       transforms.RandomAffine(0, translate=(0.1, 0.1)),
       transforms.ToTensor(),
       transforms.Normalize((0.1307,), (0.3081,))
   ])
   ```

2. **Optimization**:
   - Optimizer: Adam (lr=0.001, weight_decay=1e-4)
   - Loss: CrossEntropyLoss
   - Scheduler: StepLR (step_size=7, gamma=0.1)

3. **Training Loop**:
   - Batch processing with gradient accumulation
   - Learning rate scheduling
   - Progress monitoring and visualization

### Training Tips

1. **Learning Rate**: Start with 0.001, reduce by factor of 10 every 7 epochs
2. **Batch Size**: 128 works well for MNIST
3. **Data Augmentation**: Random rotation and translation help generalization
4. **Early Stopping**: Monitor validation accuracy to prevent overfitting

## Key Components

### 1. BasicBlock Class
- Implements the fundamental ResNet building block
- Handles skip connections and dimension matching
- Used in ResNet-18 and ResNet-34

### 2. Bottleneck Class
- More efficient block for deeper networks
- Uses 1×1 convolutions for dimension reduction/expansion
- Used in ResNet-50, ResNet-101, ResNet-152

### 3. ResNet Class
- Main network architecture
- Configurable depth and block types
- Includes feature extraction capabilities

### 4. MNISTDataLoader Class
- Handles MNIST dataset loading and preprocessing
- Implements data augmentation
- Provides visualization utilities

### 5. ResNetTrainer Class
- Complete training and evaluation pipeline
- Learning rate scheduling
- Performance monitoring and visualization

## Technical Details

### Residual Function

The residual function F(x) is defined as:
```
F(x) = W₂ * σ(W₁ * x + b₁) + b₂
```
Where σ is the ReLU activation function.

### Skip Connection Mathematics

For a residual block:
```
y = F(x, {Wᵢ}) + x
```

If dimensions don't match:
```
y = F(x, {Wᵢ}) + Wₛ * x
```
Where Wₛ is a 1×1 convolution for dimension matching.

### Gradient Flow

The gradient of the loss with respect to x:
```
∂loss/∂x = ∂loss/∂y * (∂F/∂x + 1)
```

The "+1" term ensures gradient flow even when ∂F/∂x is small.

### Computational Complexity

For ResNet-18 on MNIST:
- **FLOPs**: ~1.8 billion operations per forward pass
- **Memory**: ~45 MB for model parameters
- **Training Time**: ~2-3 minutes on GPU for 10 epochs

## Results

### Expected Performance on MNIST

| Metric | ResNet-18 | ResNet-34 | ResNet-50 |
|--------|-----------|-----------|-----------|
| Test Accuracy | >99.0% | >99.2% | >99.3% |
| Training Time (10 epochs) | ~2 min | ~3 min | ~4 min |
| Convergence | ~5 epochs | ~4 epochs | ~4 epochs |

### Training Curves

Typical training behavior:
- **Loss**: Rapid decrease in first 3-5 epochs, then gradual improvement
- **Accuracy**: Reaches >98% within 3 epochs, >99% by epoch 7
- **Validation**: Close tracking with training, minimal overfitting

### Comparison with Other Architectures

| Architecture | Parameters | MNIST Accuracy | Training Speed |
|--------------|------------|----------------|----------------|
| Simple CNN | ~60K | ~98.5% | Fast |
| LeNet-5 | ~60K | ~98.8% | Fast |
| ResNet-18 | ~11.7M | >99.0% | Medium |
| ResNet-50 | ~25.6M | >99.3% | Slow |

## Advanced Features

### 1. Feature Visualization
```python
# Extract feature maps from different layers
features = model.get_feature_maps(input_tensor)
for layer_name, feature_map in features.items():
    print(f"{layer_name}: {feature_map.shape}")
```

### 2. Model Analysis
```python
# Analyze model architecture
visualize_model_architecture(model)
```

### 3. Transfer Learning
```python
# Use pre-trained features for other tasks
model = resnet18(num_classes=10)
# Freeze early layers
for param in model.layer1.parameters():
    param.requires_grad = False
```

## Conclusion

This ResNet implementation provides:
- **Educational Value**: Clear, well-documented code for learning
- **Flexibility**: Multiple architectures and configurations
- **Performance**: State-of-the-art results on MNIST
- **Extensibility**: Easy to adapt for other datasets and tasks

The residual learning framework has become fundamental to modern deep learning, enabling the training of very deep networks and achieving breakthrough performance across many domains.

## References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. CVPR.
2. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Identity mappings in deep residual networks. ECCV.
3. Zagoruyko, S., & Komodakis, N. (2016). Wide residual networks. BMVC.

## File Structure

```
Resnet/
├── sample_resnet.py    # Main implementation
├── README.md          # This documentation
└── data/             # MNIST dataset (auto-downloaded)
```

## Requirements

- Python 3.7+
- PyTorch 1.9+
- torchvision
- matplotlib
- numpy

Install dependencies:
```bash
pip install torch torchvision matplotlib numpy
```