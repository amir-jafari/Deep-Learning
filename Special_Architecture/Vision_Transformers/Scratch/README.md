# Vision Transformer (ViT) Implementation

A clean, comprehensive implementation of Vision Transformer for image classification, based on the paper "An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale" by Dosovitskiy et al.

## Table of Contents
- [Overview](#overview)
- [Architecture](#architecture)
- [How Vision Transformer Works](#how-vision-transformer-works)
- [Implementation Details](#implementation-details)
- [Usage](#usage)
- [Model Configuration](#model-configuration)
- [Training](#training)
- [Key Components](#key-components)

## Overview

Vision Transformer (ViT) applies the transformer architecture, originally designed for natural language processing, directly to image classification tasks. Instead of processing images through convolutional layers, ViT treats images as sequences of patches, similar to how transformers process sequences of words in NLP.

## Architecture

The Vision Transformer architecture consists of the following main components:

```
Input Image (224×224×3)
        ↓
┌─────────────────────┐
│   Patch Embedding   │  ← Split image into 16×16 patches
│   (16×16 patches)   │    Convert to embeddings
└─────────────────────┘
        ↓
┌─────────────────────┐
│ Positional Embedding│  ← Add learnable position info
│   + Class Token     │    Add [CLS] token for classification
└─────────────────────┘
        ↓
┌─────────────────────┐
│  Transformer        │  ← Stack of 12 encoder blocks
│  Encoder Blocks     │    Multi-head attention + MLP
│  (×12 layers)       │
└─────────────────────┘
        ↓
┌─────────────────────┐
│ Classification Head │  ← Use [CLS] token for final prediction
│   (Linear Layer)    │
└─────────────────────┘
        ↓
    Output Classes
```

### Detailed Architecture Flow

```
1. Patch Embedding:
   Image (224×224×3) → Patches (196×768)
   
   [Image] → [Conv2d] → [Flatten] → [Permute] → [Patch Embeddings]
   224×224×3   14×14×768    196×768      196×768

2. Position Embedding:
   [CLS Token] + [Patch Embeddings] + [Positional Embeddings]
   1×768         196×768              197×768

3. Transformer Encoder (×12):
   ┌─────────────────────────────────────────┐
   │                                         │
   │  Input → LayerNorm → Multi-Head         │
   │            ↓         Attention          │
   │         Residual ←─────┘                │
   │            ↓                            │
   │         LayerNorm → MLP → Residual      │
   │                            ↓            │
   │                         Output          │
   └─────────────────────────────────────────┘

4. Classification:
   [CLS Token] → [LayerNorm] → [Linear] → [Predictions]
   768           768           10
```

## How Vision Transformer Works

### 1. **Patch Embedding**
- Divides the input image into fixed-size patches (16×16 pixels)
- Each patch is flattened and linearly projected to create patch embeddings
- For a 224×224 image with 16×16 patches, we get 196 patches
- Each patch becomes a 768-dimensional embedding vector

### 2. **Positional Encoding**
- Adds learnable positional embeddings to retain spatial information
- Prepends a learnable [CLS] token (similar to BERT) for classification
- The [CLS] token aggregates information from all patches

### 3. **Transformer Encoder**
- Uses standard transformer encoder architecture
- Each block contains:
  - **Multi-Head Self-Attention**: Allows patches to attend to each other
  - **Layer Normalization**: Applied before each sub-layer (Pre-LN)
  - **MLP**: Two linear layers with GELU activation
  - **Residual Connections**: Around both attention and MLP blocks

### 4. **Classification Head**
- Uses only the [CLS] token representation from the final layer
- Applies layer normalization followed by a linear classifier
- Outputs class probabilities

## Implementation Details

### Key Design Choices

1. **Patch Size**: 16×16 pixels (standard ViT-Base configuration)
2. **Embedding Dimension**: 768 (matches BERT-Base)
3. **Number of Heads**: 12 (768 ÷ 12 = 64 dimensions per head)
4. **Number of Layers**: 12 transformer encoder blocks
5. **MLP Ratio**: 4× (MLP hidden dimension = 4 × embedding dimension)
6. **Dropout Rate**: 0.1 for regularization

### Mathematical Formulation

**Patch Embedding:**
```
x_p = [x¹_p E; x²_p E; ...; x^N_p E]
```
Where x^i_p is the i-th flattened patch and E is the embedding matrix.

**Multi-Head Attention:**
```
Attention(Q,K,V) = softmax(QK^T/√d_k)V
MultiHead(Q,K,V) = Concat(head₁,...,head_h)W^O
```

**Transformer Block:**
```
z'_l = MSA(LN(z_{l-1})) + z_{l-1}
z_l = MLP(LN(z'_l)) + z'_l
```

## Usage

### Basic Usage

```python
from vision_transformer import VisionTransformer

# Create model with default ViT-Base configuration
model = VisionTransformer(
    image_size=224,
    patch_size=16,
    in_channels=3,
    num_classes=10,
    embed_dim=768,
    depth=12,
    num_heads=12,
    mlp_ratio=4,
    dropout_rate=0.1
)

# Forward pass
import torch
x = torch.randn(1, 3, 224, 224)  # Batch of 1 image
output = model(x)  # Shape: (1, 10)
```

### Training

```python
# Run the complete training script
python vision_transformer.py
```

This will:
- Download CIFAR-10 dataset
- Create and train a Vision Transformer
- Evaluate on test set
- Print training progress and final accuracy

## Model Configuration

### ViT-Base Configuration (Default)
- **Image Size**: 224×224
- **Patch Size**: 16×16
- **Embedding Dimension**: 768
- **Depth**: 12 layers
- **Attention Heads**: 12
- **MLP Ratio**: 4
- **Parameters**: ~86M

### Customization Options

```python
# Smaller model for faster training
small_config = {
    'image_size': 224,
    'patch_size': 16,
    'embed_dim': 384,
    'depth': 6,
    'num_heads': 6,
    'mlp_ratio': 4
}

# Larger model for better performance
large_config = {
    'image_size': 224,
    'patch_size': 16,
    'embed_dim': 1024,
    'depth': 24,
    'num_heads': 16,
    'mlp_ratio': 4
}
```

## Training

### Dataset
- **Default**: CIFAR-10 (10 classes, 32×32 images upscaled to 224×224)
- **Preprocessing**: Resize, normalize to [-1, 1]
- **Augmentation**: Basic transforms (can be extended)

### Training Configuration
- **Optimizer**: Adam with learning rate 1e-4
- **Loss Function**: Cross-entropy loss
- **Batch Size**: 32 (adjustable based on GPU memory)
- **Epochs**: 5 (for quick demonstration)

### Performance Considerations
- ViT requires large datasets or pre-training for optimal performance
- Consider using pre-trained weights for better results
- GPU recommended for reasonable training times

## Key Components

### 1. PatchEmbedding
Converts image patches to embeddings using convolution:
```python
self.conv = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
```

### 2. PositionalEmbedding
Adds spatial information and classification token:
```python
self.class_token = nn.Parameter(torch.randn(1, 1, embed_dim))
self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))
```

### 3. TransformerEncoder
Standard transformer block with pre-layer normalization:
```python
# Attention block
residual = x
x = self.norm1(x)
x, _ = self.attention(x, x, x)
x = x + residual

# MLP block
residual = x
x = self.norm2(x)
x = self.mlp(x)
x = x + residual
```

### 4. ClassificationHead
Final prediction layer using the [CLS] token:
```python
x = x[:, 0]  # Extract [CLS] token
x = self.classification_head(x)
```

## Advantages of Vision Transformer

1. **Global Context**: Self-attention allows modeling of long-range dependencies
2. **Scalability**: Performance improves with larger datasets and models
3. **Flexibility**: Can handle variable input sizes and different tasks
4. **Interpretability**: Attention maps provide insights into model focus

## Limitations

1. **Data Hungry**: Requires large datasets for optimal performance
2. **Computational Cost**: Quadratic complexity in sequence length
3. **Inductive Bias**: Lacks built-in spatial inductive biases of CNNs
4. **Small Dataset Performance**: May underperform CNNs on smaller datasets

## References

- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805)