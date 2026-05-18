# Simple Vision Transformer (ViT) - Tiny ImageNet

## Objective

Implement and train a simplified Vision Transformer (ViT) manually using PyTorch to understand transformer-based image classification and compare its behavior with CNN architectures.

## Model

- Architecture: Simple Vision Transformer (ViT)
- Framework: PyTorch
- Dataset: Tiny ImageNet
- Input: 224 × 224 RGB images
- Output: 200-class image classification

## Architecture Overview

The model was implemented manually using core PyTorch modules instead of pretrained transformer libraries.

### Components
- Patch embedding using Conv2D
- Learnable positional embeddings
- CLS token for classification
- Transformer encoder layers
- MLP classification head

### Configuration
- Patch size: 16 × 16
- Embedding dimension: 128
- Transformer depth: 4 layers
- Attention heads: 4
- MLP hidden dimension: 256

## Dataset

- Dataset: Tiny ImageNet
- Training samples: 100,000 images
- Validation samples: 10,000 images
- Classes: 200

### Preprocessing
- RandomResizedCrop (224 × 224)
- RandomHorizontalFlip
- Normalization using ImageNet statistics

## Training Setup

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Batch size: 64
- Epochs: 10
- Device: CPU / GPU (CUDA if available)

## Results

| Model | Training Method | Accuracy |
|---|---|---|
| SimpleViT | Training from scratch | 24.64% |

## Observations and Analysis

- Training the Vision Transformer from scratch was much harder compared to CNN models like ResNet.
- The model learned slowly and required more training to improve accuracy.
- Performance remained limited without transfer learning or pretrained weights.
- Compared to CNNs, the transformer model was more sensitive to training settings and data size.
- This experiment helped in understanding how Vision Transformers process images differently from CNNs.

## Key Learnings

- Understanding patch embeddings and transformer encoder architectures
- Differences between CNN and transformer-based feature learning
- Importance of positional embeddings and CLS tokens
- Challenges of training transformers from scratch on limited datasets

## Future Work

- Apply transfer learning using pretrained ViT models
- Increase transformer depth and embedding dimension
- Compare ViT performance against ResNet architectures
- Apply compression techniques such as quantization and pruning

