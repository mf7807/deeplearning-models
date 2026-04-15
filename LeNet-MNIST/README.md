# LeNet-MNIST

## Objective
Train and evaluate a LeNet-style Convolutional Neural Network (CNN) for handwritten digit classification on the MNIST dataset.

## Model
- Architecture: LeNet
- Framework: PyTorch
- Input: 28×28 grayscale images
- Output: 10-class digit classification (0–9)
### Architecture Overview:
- 2 convolutional layers with ReLU and max pooling
- 3 fully connected layers
- No batch normalization or data augmentation (baseline model)
  
## Dataset
- Dataset: MNIST
- Training samples: 60,000
- Testing samples: 10,000
- Preprocessing: Images converted to tensors only (no augmentation)

## Training Setup
- Loss function: CrossEntropyLoss
- Batch size: 64
- Device: CPU / GPU (CUDA if available)
### Baseline Hyperparameters
- Optimizer: Adam
- Epochs: 10
- Learning rate: 0.001

## Experiments & Results
With the model architecture fixed, multiple experiments were performed by adjusting the 
- Learning rate
- Number of epochs
- Optimizer

| Experiment | Epochs | Optimizer | LR | Accuracy |
|----------|--------|-------|----------|-----------|
| Baseline | 10     | Adam | 0.001 | 98.90%   |
| Tuned-1 | 15    | Adam  | 0.001 | 98.87%   |
| Tuned-2 | 15    | Adam  | 0.003 | 98.93%   |
| Tuned-3 | 10    | Adam  | 0.003 | 98.96%   |
| Tuned-4 | 20    | Adam  | 0.002 | 98.80%   |
| Tuned-5 | 20    | Adam  | 0.0005 | 98.97%   |
| Tuned-6 | 25     | Adam | 0.0005 | **98.99%**   |
| Tuned-7 | 10     | SGD | 0.001 | 97.74%   |
| Tuned-8 | 25     | SGD | 0.0005 | 98.14%   |

## Observations and Analysis
- The baseline model achieved 98.90% accuracy without data augmentation or batch normalization.
- Lower learning rates with Adam (e.g., 0.0005) produced the best accuracy.
- Adam consistently performed better than SGD, likely due to its faster convergence.
- **Best Performance**: Tuned-6 achieved 98.99% test accuracy with Adam optimizer (LR=0.0005, 25 epochs).

## Future Work
- Introduce batch normalization to improve training stability.
- Apply data augmentation techniques to improve generalization.
- Explore deeper CNN architectures or learning rate schedulers.
