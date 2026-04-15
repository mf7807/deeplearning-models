# ResNet50-CIFAR10

## Objective

Train and evaluate a fine-tuned ResNet50 using transfer learning for image classification on the CIFAR-10 dataset.


## Model

Architecture: ResNet50 (Pretrained)  
Framework: PyTorch  
Input: 224 × 224 RGB images  
Output: 10-class image classification  

### Architecture Overview

- Pretrained ResNet50 weights from ImageNet  
- Final fully connected layer replaced for CIFAR-10  
- Transfer learning applied for adaptation  
- Selective layer freezing and unfreezing for fine-tuning  

### Fine-Tuning Strategy

- Freeze all layers initially  
- Train final fully connected layer  
- Gradually unfreeze deeper layers (layer4)  
- Compare optimizer and scheduler performance  



## Dataset

Dataset: CIFAR-10  
Training samples: 50,000  
Testing samples: 10,000  
Classes: 10  

### Preprocessing

- Resize to 256  
- RandomCrop to 224  
- RandomHorizontalFlip  
- Normalization using CIFAR-10 statistics  

### Data Augmentation

- Random cropping  
- Horizontal flipping  



## Training Setup

Loss function: CrossEntropyLoss  
Batch size: 64  
Device: CPU / GPU (CUDA if available)



## Baseline Hyperparameters

Optimizer: Adam  
Epochs: 10  
Learning rate: 1e-4  
Frozen Layers: All layers except final FC  
Trainable Layers: Final classifier layer



## Experiments & Results

With the pretrained ResNet50 architecture fixed, experiments were performed by adjusting:

- Optimizer  
- Learning rate  
- Frozen layers  
- Number of epochs  

| Experiment | Epochs | Optimizer | Learning Rate | Frozen Layers | Accuracy |
|------------|--------|-----------|---------------|---------------|----------|
| Baseline | 10 | Adam | 1e-4 | Only FC trainable | 81.18% |
| Tuned-1 | 15 | SGD + Scheduler | 2e-4 | Only FC trainable | 78.14% |
| Tuned-2 | 15 | SGD + Scheduler | 1e-4 | FC + Layer4 trainable | 85.82% |
| Tuned-3 | 15 | SGD + Scheduler | 0.001 | Only FC trainable | 77.73% |
| Tuned-4 | 10 | Adam | 1e-4 | FC + Layer4 trainable | **92.52%** |


## Observations and Analysis

- Baseline transfer learning achieved strong performance (81.18%)
- SGD required careful learning rate tuning compared to Adam
- Unfreezing layer4 significantly improved performance
- High learning rate (0.001) caused training instability
- Scheduler improved convergence stability
- Best performance achieved: **95.52% accuracy**



## Key Learnings

- Transfer learning works very well with pretrained ResNet50
- Gradual unfreezing improves model adaptation
- Smaller learning rates are better when unfreezing deeper layers
- Optimizer choice significantly affects fine-tuning performance



## Future Work

- Unfreeze layer3 for deeper fine-tuning  
- Try lower learning rate (5e-5)  
- Increase epochs (15–20)  
- Try different schedulers  
