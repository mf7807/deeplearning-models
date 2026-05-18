# Importing required libraries
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Device configuration - deciding whether to run on CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
learning_rate = 3e-4
num_epochs = 5

# Download Tiny ImageNet dataset
if not os.path.exists("tiny-imagenet-200"):
    os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    os.system("unzip -q tiny-imagenet-200.zip")

# Preparing validation folder structure
val_dir = "tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

with open(annotations_file) as f:
    lines = f.readlines()

for line in lines:
    img, cls = line.split("\t")[:2]

    cls_dir = os.path.join(val_dir, cls)
    os.makedirs(cls_dir, exist_ok=True)

    src_path = os.path.join(images_dir, img)
    dst_path = os.path.join(cls_dir, img)

    if os.path.exists(src_path):
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)

# Remove empty images folder
if os.path.exists(images_dir) and not os.listdir(images_dir):
    os.rmdir(images_dir)

# Image transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Loading dataset
train_dataset = datasets.ImageFolder("tiny-imagenet-200/train", transform=transform_train)
val_dataset = datasets.ImageFolder("tiny-imagenet-200/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

# Load pre-trained Vision Transformer model
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Freeze all backbone layers
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.heads.head = nn.Linear(model.heads.head.in_features, 200)

# Train only classifier head
for param in model.heads.head.parameters():
    param.requires_grad = True

model = model.to(device)

# Loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
total_steps = len(train_loader)

for epoch in range(num_epochs):
    model.train()

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        # Forward pass
        outputs = model(images.float())
        loss = cost(outputs, labels.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 400 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}')

    # Save checkpoint after each epoch
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }, 'vit_checkpoint.pth')

print("Finished Training")

# Evaluation
model.eval()

with torch.no_grad():
    correct = 0
    total = 0

    for images, labels in val_loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images.float())
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    acc = 100 * correct / total
    print(f'Accuracy of the model: {acc:.3f}')

print("Model evaluation completed")

# Save final model
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'accuracy': acc
}, 'vit_final.pth')

print("Model saved successfully")