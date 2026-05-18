# Importing required libraries
import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torchvision.models import vit_b_16, ViT_B_16_Weights

# Device configuration - deciding whether to run on CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Mount Google Drive (for saving/loading checkpoints)
from google.colab import drive
drive.mount('/content/drive')

# Download Tiny ImageNet dataset if not already present
if not os.path.exists("tiny-imagenet-200"):
    os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    os.system("unzip -q tiny-imagenet-200.zip")

# Prepare validation folder structure
val_dir = "tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
val_annotations = os.path.join(val_dir, "val_annotations.txt")

with open(val_annotations) as f:
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
    else:
        if not os.path.exists(dst_path):
            print(f"Missing image: {img}")

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

# Load dataset
train_dataset = datasets.ImageFolder("tiny-imagenet-200/train", transform=transform_train)
val_dataset = datasets.ImageFolder("tiny-imagenet-200/val", transform=transform_val)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
test_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=2, pin_memory=True)

# Load checkpoint
checkpoint = torch.load("/content/drive/MyDrive/Colab Notebooks/vit.pth", weights_only=False)

# Load pretrained ViT model
model = vit_b_16(weights=ViT_B_16_Weights.DEFAULT)

# Freeze backbone
for param in model.parameters():
    param.requires_grad = False

# Replace classification head
model.heads.head = nn.Linear(model.heads.head.in_features, 200)

# Train only classifier head
for param in model.heads.head.parameters():
    param.requires_grad = True

model = model.to(device)

# Load trained weights
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Move model to CPU for quantization
model = model.to("cpu")

# Quantize model for inference
quantized_model = torch.quantization.quantize_dynamic(
    model,
    {nn.Linear},
    dtype=torch.qint8
)

# Test function
def test(model):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.to("cpu")
            labels = labels.to("cpu")

            outputs = model(data)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of quantized model: {accuracy:.2f}%')

# Run evaluation
test(quantized_model)

# Save quantized model
save_path = "/content/drive/MyDrive/Colab Notebooks/vit_quantized.pth"
torch.save(quantized_model.state_dict(), save_path)

print("Quantized model saved")

# Print model sizes
quantized_model_size = os.path.getsize(save_path) / (1024 * 1024)
print(f"Quantized model size: {quantized_model_size:.2f} MB")

original_model_size = os.path.getsize("/content/drive/MyDrive/Colab Notebooks/vit.pth") / (1024 * 1024)
print(f"Original model size: {original_model_size:.2f} MB")