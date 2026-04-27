# Importing the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
import torchvision
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import torch.nn.utils.prune as prune
import matplotlib.pyplot as plt
import os

# Device configuration - deciding whether to run the training on cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 5

# CIFAR-10 dataset
transform_train = transforms.Compose([ 
    transforms.Resize(224),
    transforms.RandomCrop(224, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([  
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])

# Download training and testing data, if not already downloaded inside the 'data' folder
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load train and test data
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# View some of the images from the dataset using matplotlib
# Get a batch
examples = iter(test_loader)
example_data, example_targets = next(examples)

class_names = train_dataset.classes

plt.figure(figsize=(8,5))
for i in range(6):
    plt.subplot(2, 3, i+1)

    mean = torch.tensor([0.4914, 0.4822, 0.4465])
    std  = torch.tensor([0.2023, 0.1994, 0.2010])

    img = example_data[i].permute(1, 2, 0)  # C,H,W → H,W,C
    img = img * std + mean                   # unnormalize
    img = img.clamp(0, 1)
    plt.imshow(img)
    plt.title(class_names[example_targets[i]])
    plt.axis("off")

plt.show()

# Load checkpoint
checkpoint = torch.load("/content/drive/MyDrive/Colab Notebooks/resnet50_cifar10.pth", weights_only=False)

# Initialize model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 10)

# Load state dict
model.load_state_dict(checkpoint['model_state_dict'])

# Move model to device (GPU/CPU)
model = model.to(device)

# Set model to evaluation mode
model.eval()

# Pruning convolutional layers
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.ln_structured(module, name="weight", amount=0.2, n=2, dim=0)

# Make pruning permanent
for module in model.modules():
    if isinstance(module, nn.Conv2d):
        prune.remove(module, "weight")

# Test function
def test(model):
    model.eval()  # Set the model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Turn off gradient computation (no training)
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)  # Move data to same device as model
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total * 100
    print(f'Accuracy of the model on the test set: {accuracy:.2f}%')

# Run test on original model
test(model)
