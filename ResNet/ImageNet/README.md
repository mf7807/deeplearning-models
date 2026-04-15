# ResNet50-TinyImageNet

## Objective

Train and evaluate a fine-tuned ResNet50 using transfer learning for image classification on the Tiny ImageNet dataset.


## Model

Architecture: ResNet50 (Pretrained)  
Framework: PyTorch  
Input: 224 × 224 RGB images  
Output: 200-class image classification  

### Architecture Overview

- Pretrained ResNet50 weights from ImageNet  
- Final fully connected layer replaced for Tiny ImageNet  
- Transfer learning applied for domain adaptation  
- Gradual unfreezing of deeper layers for advanced fine-tuning  

### Fine-Tuning Strategy

- Freeze all layers initially  
- Unfreeze layer4 and final classifier  
- Compare optimizers and fine-tuning methods  


## Dataset

Dataset: Tiny ImageNet  
Training samples: 100,000 images  
Validation samples: 10,000 images  
Classes: 200  

### Preprocessing

- RandomResizedCrop (224 × 224)  
- RandomHorizontalFlip  
- Normalization using ImageNet statistics  

### Data Augmentation

- Random cropping  
- Horizontal flipping  


## Training Setup

Loss function: CrossEntropyLoss  
Batch size: 64  
Device: CPU / GPU (CUDA if available)


## Baseline Hyperparameters

Optimizer: Adam  
Epochs: 15  
Learning rate: 5e-5  
Frozen Layers: All except layer4 and FC  
Trainable Layers: Layer4 + FC


## Experiments & Results

With the pretrained ResNet50 architecture fixed, experiments were performed by adjusting:

- Optimizer  
- Frozen layers  
- Learning rate  
- Scheduler  

| Experiment | Epochs | Optimizer | Learning Rate | Trainable Layers | Accuracy |
|------------|--------|-----------|---------------|------------------|----------|
| Baseline-1 | 10 | SGD + Scheduler | 1e-4 | FC only | 31.13% |
| Tuned-1 | 10 | Adam | 1e-4 | FC only | 52.35% |
| Tuned-2 | 10 | SGD + Scheduler | 1e-4 | Layer4 + FC | 40.34% |
| Tuned-3 | 15 | Adam | 9e-5 | Layer4 + FC | **65.29%** |


## Observations and Analysis

- Adam consistently performed better than SGD across experiments  
- Fine-tuning only the classifier achieved moderate performance but limited dataset adaptation  
- Unfreezing layer4 allowed the model to learn more dataset-specific features  
- Lower learning rate (9e-5) improved stability during deeper fine-tuning  
- Increasing training epochs from 10 to 15 further improved performance  
- Combining Adam, lower learning rate, longer training, and unfrozen Layer4 + FC produced the best performance: **65.29% accuracy**



## Key Learnings

- Deeper fine-tuning improves performance when enough training time is provided  
- Lower learning rates are important when unfreezing deeper pretrained layers  
- Adam optimizer provided more stable convergence for this task  
- Training duration (epochs) significantly impacts fine-tuning results  
- Tiny ImageNet is significantly harder than CIFAR-10, requiring deeper adaptation



## Future Work

- Reduce learning rate further (5e-5 or 1e-5)  
- Increase epochs (15-20)  
- Try cosine learning rate scheduler  
- Gradually unfreeze layer3 + layer4  
- Compare Adam vs SGD for deeper fine-tuning  