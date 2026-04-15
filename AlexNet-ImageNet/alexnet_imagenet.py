# Importing the required libraries
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

# Device configuration - deciding whether to run the training on CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 10

# Download and extract dataset if not exists
if not os.path.exists("tiny-imagenet-200"):
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip -q tiny-imagenet-200.zip

!ls tiny-imagenet-200/val

# Organize validation images into class subfolders
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

    if not os.path.exists(src_path):
        if os.path.exists(dst_path):
            print(f"Image '{img}' already moved to '{cls_dir}'. Skipping.")
        else:
            print(
                f"Warning: Source image '{src_path}' not found for '{img}'. "
                f"It might be missing or the dataset structure is unexpected."
            )
    else:
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Image '{img}' already exists in '{cls_dir}'. Skipping move.")

if os.path.exists(images_dir):
    if not os.listdir(images_dir):
        os.rmdir(images_dir)
    else:
        print(
            f"Warning: '{images_dir}' is not empty after processing. "
            f"Some files might not have been moved or were not part of the annotations."
        )

# Data transformations
transform_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

transform_val = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# Datasets and dataloaders
train_dataset = datasets.ImageFolder(
    "tiny-imagenet-200/train",
    transform=transform_train
)

val_dataset = datasets.ImageFolder(
    "tiny-imagenet-200/val",
    transform=transform_val
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# View some images from dataset
examples = iter(val_loader)
example_data, example_targets = next(examples)
class_names = train_dataset.classes

plt.figure(figsize=(8, 5))
for i in range(6):
    plt.subplot(2, 3, i + 1)
    img = example_data[i].permute(1, 2, 0)  # C,H,W → H,W,C
    plt.imshow(img)
    plt.title(class_names[example_targets[i]])
    plt.axis("off")
plt.show()

# Load pretrained AlexNet model
model = models.alexnet(pretrained=True)

# Replace classifier
model.classifier[6] = nn.Linear(4096, 200)  # 200 classes
model = model.to(device)

# Freeze feature layers
for param in model.features.parameters():
    param.requires_grad = False

# Loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Training
total_steps = len(train_loader)

for current_epoch in range(num_epochs):
    model.train()
    for idx, (train_x, train_label) in enumerate(train_loader):
        train_x = train_x.to(device)
        train_label = train_label.to(device)

        # Forward pass
        predict_y = model(train_x.float())
        loss = cost(predict_y, train_label.long())

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (idx + 1) % 400 == 0:
            print(
                f'Epoch [{current_epoch + 1}/{num_epochs}], '
                f'Step [{idx + 1}/{total_steps}], '
                f'Loss: {loss.item():.4f}'
            )

print('Finished Training')

# Evaluation
model.eval()
with torch.no_grad():
    all_correct_num = 0
    all_sample_num = 0

    for idx, (test_x, test_label) in enumerate(val_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)

        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)

        current_correct_num = predict_y == test_label
        all_correct_num += torch.sum(current_correct_num).item()
        all_sample_num += current_correct_num.shape[0]

    acc = (all_correct_num / all_sample_num) * 100
    print('Accuracy of the baseline model: {:.3f}'.format(acc), flush=True)
    print("Model finished training")
