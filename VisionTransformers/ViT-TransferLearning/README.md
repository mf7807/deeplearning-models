# Vision Transformer (ViT) - Transfer Learning on Tiny ImageNet

## Objective

Fine-tune a pretrained Vision Transformer (ViT) using transfer learning for image classification on the Tiny ImageNet dataset.


## Model

- Architecture: Vision Transformer (ViT)
- Framework: PyTorch
- Dataset: Tiny ImageNet
- Input: 224 × 224 RGB images
- Output: 200-class image classification


## Architecture Overview

A pretrained Vision Transformer model was used and adapted for Tiny ImageNet classification.

### Transfer Learning Strategy
- Pretrained weights loaded from ImageNet training
- Final classification head replaced for 200 classes
- Most transformer layers frozen initially
- Only classification head fine-tuned


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
- Epochs: 5
- Learning rate: 1e-4
- Device: CPU / GPU (CUDA if available)


## Results

| Model | Training Method | Accuracy |
|---|---|---|
| Pretrained ViT | Transfer Learning | 84.22% |


## Observations and Analysis

- Transfer learning improved performance significantly compared to training the vision transformer from scratch.
- The pretrained model converged much faster and achieved strong accuracy within only a few epochs.
- Freezing most layers reduced training time while still allowing good adaptation to Tiny ImageNet.
- Compared to the manually implemented ViT, the pretrained model was more stable and learned meaningful image representations.


## Key Learnings

- Understanding transfer learning in transformer-based vision models
- Importance of pretrained weights for Vision Transformers
- Differences between training from scratch and fine-tuning
- Comparison of ViT performance against CNN architectures


## Future Work

- Unfreeze additional transformer layers for deeper fine-tuning
- Apply compression techniques such as quantization and pruning

