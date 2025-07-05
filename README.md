# 🧠 Deep Learning Repository

<div align="center">

![Deep Learning](https://img.shields.io/badge/Deep%20Learning-Neural%20Networks-blue?style=for-the-badge&logo=tensorflow)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)

**A comprehensive collection of deep learning implementations, tutorials, and advanced architectures**

[🚀 Getting Started](#-getting-started) • [📚 Documentation](#-documentation) • [🏗️ Architecture](#️-architecture) • [🤝 Contributing](#-contributing)

</div>

---

## 📋 Table of Contents

- [🎯 Overview](#-overview)
- [🚀 Getting Started](#-getting-started)
- [📁 Repository Structure](#-repository-structure)
  - [🔥 PyTorch Implementations](#-pytorch-implementations)
  - [🏗️ Special Architectures](#️-special-architectures)
  - [⚡ TensorFlow Advanced](#-tensorflow-advanced)
  - [📚 TensorFlow Basics](#-tensorflow-basics)
- [💻 Installation](#-installation)
- [🎓 Usage Examples](#-usage-examples)
- [📖 Learning Path](#-learning-path)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)

## 🎯 Overview

This repository contains a comprehensive collection of deep learning implementations spanning multiple frameworks and architectures. From fundamental concepts to cutting-edge research implementations, this repository serves as both an educational resource and a practical toolkit for deep learning practitioners.

### ✨ Key Features

- 🔬 **Research-Grade Implementations**: State-of-the-art architectures with detailed explanations
- 📚 **Educational Content**: Step-by-step tutorials and lecture materials
- 🛠️ **Multiple Frameworks**: PyTorch and TensorFlow implementations
- 🎯 **Practical Examples**: Real-world applications and use cases
- 📊 **Comprehensive Coverage**: From basic MLPs to advanced transformers

## 🚀 Getting Started

### Prerequisites

- Python 3.7+
- CUDA-compatible GPU (recommended)
- 8GB+ RAM

### Quick Start

```bash
# Clone the repository
git clone https://github.com/yourusername/deep-learning-repo.git
cd deep-learning-repo

# Install dependencies
pip install -r requirements.txt

# Run a quick example
cd Pytorch/MLP/1_Simple_f_approx
python simple_function_approximation.py
```

## 📁 Repository Structure

### 🔥 [PyTorch Implementations](./Pytorch)

Complete PyTorch ecosystem with implementations ranging from basics to advanced concepts.

#### 📊 [Convolutional Neural Networks](./Pytorch/CNN)
- **[Image Classification](./Pytorch/CNN/1_ImageClassification)** - CIFAR-10, ImageNet-style classifiers
- **[Text Classification 1D](./Pytorch/CNN/2_TextClassification_1D)** - 1D CNNs for NLP tasks

#### 🧠 [Multi-Layer Perceptrons](./Pytorch/MLP)
- **[Function Approximation](./Pytorch/MLP/1_Simple_f_approx)** - Basic neural network fundamentals
- **[XOR Problem](./Pytorch/MLP/2_XOR)** - Classic non-linear classification
- **[Image Classification](./Pytorch/MLP/3_ImageClassification)** - MNIST and CIFAR-10 with MLPs
- **[Text Classification](./Pytorch/MLP/4_TextClassification)** - NLP with fully connected networks
- **[Sacred Experiments](./Pytorch/MLP/5_SacredExperiments)** - Experiment tracking and management

#### 🔄 [Recurrent Neural Networks](./Pytorch/RNN)
- **[Signal Approximation](./Pytorch/RNN/1_ChirpApprox)** - Time series modeling with RNNs
- **[Text Classification](./Pytorch/RNN/2_TextClassification)** - Sentiment analysis with LSTM/GRU
  - **[SST-2 Dataset](./Pytorch/RNN/2_TextClassification/SST-2)** - Stanford Sentiment Treebank

#### 🎓 [Lecture Materials](./Pytorch/Lecture)
Comprehensive PyTorch tutorials and educational content:
- **[PyTorch Basics](./Pytorch/Lecture/1-pytoch_basics)** - Tensors, autograd, and fundamentals
- **[Neural Network Modules](./Pytorch/Lecture/2-nn_module)** - Building blocks of PyTorch
- **[Linear Regression](./Pytorch/Lecture/3-Linear_reg)** - From scratch implementation
- **[Logistic Regression](./Pytorch/Lecture/4-Logistic_reg)** - Classification fundamentals
- **[MNIST MLP](./Pytorch/Lecture/5-MLP_Mnist)** - Handwritten digit recognition
- **[Convolutional MNIST](./Pytorch/Lecture/6-Conv_Mnist)** - CNN for image classification
- **[RNN Fundamentals](./Pytorch/Lecture/7-RNN)** - Sequence modeling basics
- **[Model Visualization](./Pytorch/Lecture/8-Visualize_Model)** - Understanding network behavior
- **[Overfitting & Autoencoders](./Pytorch/Lecture/9-%20Overfittinf-AutoEncder)** - Regularization techniques
- **[DataLoader](./Pytorch/Lecture/10-DataLoader)** - Efficient data handling

### 🏗️ [Special Architectures](./Special_Architecture)

Advanced and cutting-edge deep learning architectures with detailed implementations.

#### 🤖 [Generative Adversarial Networks](./Special_Architecture/GAN)
- **[Simple GAN](./Special_Architecture/GAN/Simple)** - Basic GAN implementation with Fashion-MNIST
- **[Advanced GAN](./Special_Architecture/GAN/Advance)** - DCGAN, WGAN, and other variants

#### 👁️ [Vision Transformers](./Special_Architecture/Vision_Transformers)
- **[From Scratch](./Special_Architecture/Vision_Transformers/Scratch)** - Complete ViT implementation
- **[Pretrained Models](./Special_Architecture/Vision_Transformers/Pretrained)** - Using pretrained transformers

#### 🏢 [ResNet](./Special_Architecture/Resnet)
Residual Networks with comprehensive documentation and multiple variants

#### 🔗 [Kolmogorov-Arnold Networks](./Special_Architecture/KAN)
Implementation of the novel KAN architecture

#### 🔄 [CNN + LSTM Hybrid](./Special_Architecture/CNN+LSTM)
Combined architectures for complex pattern recognition

#### 🌐 [Sequence-to-Sequence NLP](./Special_Architecture/Seq2Seq_NLP)
Neural Machine Translation and other seq2seq applications

### ⚡ [TensorFlow Advanced](./Tenflow_Advance)

Advanced TensorFlow implementations with production-ready code.

#### 📊 [Advanced CNNs](./Tenflow_Advance/CNN)
- **[Image Classification](./Tenflow_Advance/CNN/1_ImageClassification)** - Advanced CNN architectures
- **[Speech Recognition](./Tenflow_Advance/CNN/2_Speech_Recognition_1D)** - 1D CNNs for audio processing

#### 🧠 [Advanced MLPs](./Tenflow_Advance/MLP)
- **[Function Approximation](./Tenflow_Advance/MLP/1_Simple_f_approx)** - Advanced optimization techniques
- **[XOR Problem](./Tenflow_Advance/MLP/2_XOR)** - Advanced training strategies
- **[Image Classification](./Tenflow_Advance/MLP/3_ImageClassification)** - Production-ready classifiers
- **[Data Pipeline](./Tenflow_Advance/MLP/4_DataPipeline)** - Efficient data processing
- **[Sacred Experiments](./Tenflow_Advance/MLP/5_SacredExperiments)** - Advanced experiment management

#### 🔄 [Advanced RNNs](./Tenflow_Advance/RNN)
- **[Signal Processing](./Tenflow_Advance/RNN/1_ChirpApprox)** - Advanced time series analysis

### 📚 [TensorFlow Basics](./Tensorflow_Basic)

Foundational TensorFlow implementations perfect for beginners.

#### 📊 [Basic CNNs](./Tensorflow_Basic/CNN)
- **[Image Classification](./Tensorflow_Basic/CNN/1_ImageClassification)** - MNIST, CIFAR-10 basics
- **[Signal Classification](./Tensorflow_Basic/CNN/2_SignalClassification_1D)** - 1D signal processing

#### 🧠 [Basic MLPs](./Tensorflow_Basic/MLP)
- **[Function Approximation](./Tensorflow_Basic/MLP/1_Simple_f_approx)** - Neural network basics
- **[XOR Problem](./Tensorflow_Basic/MLP/2_XOR)** - Non-linear classification
- **[Image Classification](./Tensorflow_Basic/MLP/3_ImageClassification)** - Basic image recognition
- **[Functional API](./Tensorflow_Basic/MLP/4_FunctionalAPI)** - TensorFlow's functional approach
- **[Model Subclassing](./Tensorflow_Basic/MLP/5_ModelSubclassing)** - Custom model creation
- **[Grid Search](./Tensorflow_Basic/MLP/6_SklearnsGridSearch)** - Hyperparameter optimization

#### 🔄 [Basic RNNs](./Tensorflow_Basic/RNN)
- **[Signal Approximation](./Tensorflow_Basic/RNN/1_ChirpApprox)** - Time series basics
- **[Music Classification](./Tensorflow_Basic/RNN/2_MusicGenreClassification)** - Audio classification

#### 🎓 [Lecture Codes](./Tensorflow_Basic/Lecture_Codes)
Educational materials and tutorial implementations

## 💻 Installation

### Option 1: Conda Environment (Recommended)

```bash
# Create conda environment
conda create -n deeplearning python=3.8
conda activate deeplearning

# Install PyTorch
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install TensorFlow
pip install tensorflow

# Install additional dependencies
pip install numpy pandas matplotlib seaborn scikit-learn jupyter sacred
```

### Option 2: Virtual Environment

```bash
# Create virtual environment
python -m venv deeplearning_env
source deeplearning_env/bin/activate  # On Windows: deeplearning_env\Scripts\activate

# Install requirements
pip install torch torchvision torchaudio
pip install tensorflow
pip install numpy pandas matplotlib seaborn scikit-learn jupyter sacred
```

### Option 3: Docker

```bash
# Pull the official TensorFlow GPU image
docker pull tensorflow/tensorflow:latest-gpu-jupyter

# Run with GPU support
docker run --gpus all -p 8888:8888 -v $(pwd):/tf/notebooks tensorflow/tensorflow:latest-gpu-jupyter
```

## 🎓 Usage Examples

### Quick Start with PyTorch

```python
# Simple MLP for MNIST
cd Pytorch/MLP/3_ImageClassification
python mnist_mlp.py

# CNN for image classification
cd Pytorch/CNN/1_ImageClassification
python cifar10_cnn.py

# RNN for text classification
cd Pytorch/RNN/2_TextClassification
python sentiment_analysis.py
```

### Advanced Architectures

```python
# Vision Transformer
cd Special_Architecture/Vision_Transformers/Scratch
python vision_transformer.py

# ResNet implementation
cd Special_Architecture/Resnet
python sample_resnet.py

# GAN training
cd Special_Architecture/GAN/Simple
python simple_gan.py
```

### TensorFlow Examples

```python
# Basic neural network
cd Tensorflow_Basic/MLP/1_Simple_f_approx
python function_approximation.py

# Advanced CNN
cd Tenflow_Advance/CNN/1_ImageClassification
python advanced_cnn.py
```

## 📖 Learning Path

### 🌱 Beginner Path
1. **Start with basics**: `Tensorflow_Basic/MLP/1_Simple_f_approx`
2. **Learn CNNs**: `Tensorflow_Basic/CNN/1_ImageClassification`
3. **Explore RNNs**: `Tensorflow_Basic/RNN/1_ChirpApprox`
4. **PyTorch transition**: `Pytorch/Lecture/1-pytoch_basics`

### 🚀 Intermediate Path
1. **Advanced MLPs**: `Pytorch/MLP/3_ImageClassification`
2. **CNN architectures**: `Pytorch/CNN/1_ImageClassification`
3. **RNN applications**: `Pytorch/RNN/2_TextClassification`
4. **Experiment tracking**: `Pytorch/MLP/5_SacredExperiments`

### 🎯 Advanced Path
1. **Vision Transformers**: `Special_Architecture/Vision_Transformers`
2. **GANs**: `Special_Architecture/GAN`
3. **ResNet**: `Special_Architecture/Resnet`
4. **Novel architectures**: `Special_Architecture/KAN`

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### How to Contribute

1. **Fork the repository**
2. **Create a feature branch**: `git checkout -b feature/amazing-feature`
3. **Commit your changes**: `git commit -m 'Add amazing feature'`
4. **Push to the branch**: `git push origin feature/amazing-feature`
5. **Open a Pull Request**

### Contribution Areas

- 🐛 **Bug fixes**
- 📚 **Documentation improvements**
- ✨ **New architecture implementations**
- 🧪 **Additional examples and tutorials**
- 🔧 **Performance optimizations**

## 📊 Repository Statistics

- **Total Implementations**: 50+ neural network architectures
- **Frameworks Covered**: PyTorch, TensorFlow
- **Application Domains**: Computer Vision, NLP, Time Series, Generative Models
- **Educational Content**: 10+ comprehensive tutorials
- **Code Quality**: Documented, tested, and production-ready

## 🏆 Featured Implementations

- **🏗️ Vision Transformer**: Complete from-scratch implementation with detailed explanations
- **🤖 ResNet**: Comprehensive residual network with multiple variants
- **🎨 GANs**: Simple to advanced generative models
- **🔗 KAN**: Novel Kolmogorov-Arnold Networks
- **🌐 Seq2Seq**: Neural machine translation systems

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/deep-learning-repo/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/deep-learning-repo/discussions)
- **Email**: your.email@example.com

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**⭐ Star this repository if you find it helpful!**

Made with ❤️ by the Deep Learning Community

</div>