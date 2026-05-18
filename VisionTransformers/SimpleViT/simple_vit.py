# Importing required libraries
import os
import shutil
import torch
import torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device configuration - deciding whether to run on CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
learning_rate = 3e-4
num_epochs = 10

# Download Tiny-ImageNet dataset
if not os.path.exists("tiny-imagenet-200"):
    os.system("wget http://cs231n.stanford.edu/tiny-imagenet-200.zip")
    os.system("unzip -q tiny-imagenet-200.zip")

# Preparing validation folder structure
val_dir = "tiny-imagenet-200/val"
images_dir = os.path.join(val_dir, "images")
annotations_file = os.path.join(val_dir, "val_annotations.txt")

with open(annotations_file, "r") as f:
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

# Patch Embedding for ViT
class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=256):
        super(PatchEmbedding, self).__init__()

        self.num_patches = (img_size // patch_size) ** 2

        self.projection = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2)
        x = x.transpose(1, 2)
        return x

# Vision Transformer model
class SimpleViT(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_classes=200,
                 embed_dim=256, depth=4, num_heads=4, mlp_dim=512):

        super(SimpleViT, self).__init__()

        self.patch_embed = PatchEmbedding(img_size, patch_size, 3, embed_dim)

        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=mlp_dim,
            batch_first=True
        )

        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        x = self.patch_embed(x)
        batch_size = x.shape[0]

        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embedding
        x = self.transformer(x)
        x = self.fc(x[:, 0])

        return x

# Model initialization
model = SimpleViT(
    img_size=224,
    patch_size=16,
    num_classes=200,
    embed_dim=256
).to(device)

# Loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training the model
total_steps = len(train_loader)

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

print("Model finished evaluation")

# Optional:Save model and optimizer states
# torch.save({
#     'epoch': num_epochs,  
#     'model_state_dict': model.state_dict(),
#     'optimizer_state_dict': optimizer.state_dict(),
#     'loss': cost,  
# }, 'vit.pth')

# print("Model saved successfully")
