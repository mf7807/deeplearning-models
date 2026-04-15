# Deep Learning CNN Experiments (PyTorch)

## Overview

This repository contains a series of deep learning experiments exploring different Convolutional Neural Network (CNN) architectures across multiple datasets using PyTorch. The goal is to understand model behavior, transfer learning, and fine-tuning strategies through progressive experimentation.

The experiments progress from:
- Training a CNN from scratch  
- Applying transfer learning  
- Advanced fine-tuning  
- Deep fine-tuning on challenging datasets  


## Experiments Included

### LeNet - MNIST

Training a classic CNN architecture from scratch on handwritten digits.

- Model: LeNet  
- Dataset: MNIST  
- Focus: CNN fundamentals and training pipeline  


### AlexNet - Tiny ImageNet

Applying transfer learning using a deeper architecture.

- Model: AlexNet (Pretrained)  
- Dataset: Tiny ImageNet (200 classes)  
- Focus: Transfer learning and classifier fine-tuning
  

### ResNet50 - CIFAR-10

Exploring deeper architectures and advanced fine-tuning strategies.

- Model: ResNet50 (Pretrained)  
- Dataset: CIFAR-10  
- Focus: Layer freezing and optimizer comparison
  

### ResNet50 - Tiny ImageNet

Deep fine-tuning on a challenging multi-class dataset.

- Model: ResNet50 (Pretrained)  
- Dataset: Tiny ImageNet  
- Focus: Deeper fine-tuning and learning rate tuning  


## Contents

Each folder contains:
- Model implementation  
- Training configuration  
- Hyperparameter experiments  
- Results and observations  
- Key learnings  


## Key Concepts Explored

- Convolutional Neural Networks (CNNs)
- Transfer Learning
- Fine-Tuning
- Optimizer Comparison (SGD vs Adam)
- Data Augmentation
- Model Evaluation

## Project Structure

```
deeplearning-models
│
├── LeNet-MNIST
├── AlexNet-TinyImageNet
├── ResNet50
│   ├── CIFAR10
│   └── TinyImageNet
```

## Technologies

- Python  
- PyTorch  
- Torchvision  
- Matplotlib  


## Why This Project

This repository focuses on understanding model behavior through experimentation rather than only training models. Each experiment builds on the previous one, gradually introducing more advanced techniques and harder datasets.
