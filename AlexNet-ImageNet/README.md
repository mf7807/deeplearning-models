# AlexNet-TinyImageNet

## Objective

Train and evaluate a fine-tuned AlexNet using transfer learning for image classification on the Tiny ImageNet dataset.


## Model

**Architecture:** AlexNet (Pretrained)  
**Framework:** PyTorch  
**Input:** 224 × 224 RGB images  
**Output:** 200-class image classification  

### Architecture Overview

- Pretrained AlexNet weights from ImageNet  
- Feature layers frozen  
- Final classifier layer replaced  
- Transfer learning applied to adapt model to Tiny ImageNet  

### Fine-Tuning Strategy

- Freeze convolutional feature layers  
- Replace final fully connected layer  
- Train classifier layers only  



## Dataset

**Dataset:** Tiny ImageNet  
**Training samples:** 100,000 images  
**Validation samples:** 10,000 images  
**Classes:** 200  

### Preprocessing

- RandomResizedCrop (224 × 224)  
- RandomHorizontalFlip  
- Normalization using ImageNet statistics  

### Data Augmentation

- Random cropping  
- Horizontal flipping  



## Training Setup

**Loss function:** CrossEntropyLoss  
**Batch size:** 64  
**Device:** CPU / GPU (CUDA if available)  



## Baseline Hyperparameters

**Optimizer:** Adam  
**Epochs:** 10  
**Learning rate:** 1e-4  
**Frozen Layers:** Feature extractor frozen  
**Trainable Layers:** Classifier only  



## Experiments & Results

With the pretrained AlexNet architecture fixed, experiments were performed by adjusting:

- Learning rate  
- Batch size  
- Freezing strategy  

| Experiment | Epochs | Batch Size | Learning Rate | Frozen Layers | Accuracy |
|------------|--------|------------|---------------|---------------|----------|
| Tuned-1 | 10 | 128 | 0.001 | Features frozen | 46.15% |
| Tuned-2 | 10 | 64 | 0.001 | Features frozen | 48.02% |
| Tuned-3 | 10 | 64 | 0.0003 | Features frozen | 44.29% |
| Tuned-4 | 10 | 64 | 0.001 | Unfrozen features | 55.20% |
| Best Model | 10 | 64 | 0.001 | Features frozen | **57.00%** |



## Observations and Analysis

- Learning rate **0.001 performed best** for AlexNet fine-tuning  
- Very high learning rates (0.01, 0.03) caused training instability  
- Reducing batch size from 128 → 64 improved performance  
- Fine-tuning pretrained layers improved performance slightly  
- Best performance achieved: **57.00% accuracy**



## Future Work

- Unfreeze deeper layers for advanced fine-tuning  
- Introduce learning rate scheduler  
- Compare SGD vs Adam optimizers  
- Increase training epochs for better convergence  
- Compare with deeper architectures like ResNet-50