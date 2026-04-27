# Importing the required libraries
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from sklearn.cluster import KMeans
import os

# Device configuration - deciding whether to run the training on cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
learning_rate = 5e-5
num_epochs = 5

# CIFAR-10 dataset transformations
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

# Load CIFAR-10 datasets
train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

# Load train and test data into DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pre-trained model and modify final layer for CIFAR-10 classification
checkpoint = torch.load("/content/drive/MyDrive/Colab Notebooks/resnet50_cifar10.pth", weights_only=False)
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(2048, 10)  # 10 classes for CIFAR-10
model = model.to(device)

# Load model weights from checkpoint
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# K-means quantization function
def kmeans_quantize_tensor(tensor, k=16):
    shape = tensor.shape
    flat = tensor.view(-1, 1).cpu().numpy()

    kmeans = KMeans(n_clusters=k, n_init=3)
    kmeans.fit(flat)

    centers = kmeans.cluster_centers_
    labels = kmeans.labels_

    quantized = torch.tensor(centers[labels]).view(shape)
    return quantized.to(tensor.device)

# Apply K-means quantization to the model
def apply_kmeans(model, k=16):
    for name, param in model.named_parameters():
        if "weight" in name:
            param.data = kmeans_quantize_tensor(param.data, k)
    return model

model = apply_kmeans(model)

# Test function to evaluate the model
def test(model):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation (for memory efficiency)
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total * 100
    print(f'Accuracy of the quantized model on the test set: {accuracy:.2f}%')

# Run the test
test(model)