# Importing the required libraries
import os
import shutil
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms, models
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader

# Device configuration - deciding whether to run training on CPU or GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyperparameters
batch_size = 64
learning_rate = 9e-5
num_epochs = 15

# Download dataset if not exists
if not os.path.exists("tiny-imagenet-200"):
    !wget http://cs231n.stanford.edu/tiny-imagenet-200.zip
    !unzip -q tiny-imagenet-200.zip

!ls tiny-imagenet-200/val

# Organize validation set
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
            print(f"Warning: missing source '{src_path}'")
    else:
        if not os.path.exists(dst_path):
            shutil.move(src_path, dst_path)
        else:
            print(f"Image '{img}' already exists in '{cls_dir}'")

if os.path.exists(images_dir):
    if not os.listdir(images_dir):
        os.rmdir(images_dir)

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

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.485, 0.456, 0.406),
        (0.229, 0.224, 0.225)
    )
])

# Datasets and loaders
train_dataset = datasets.ImageFolder(
    "tiny-imagenet-200/train",
    transform=transform_train
)

test_dataset = datasets.ImageFolder(
    "tiny-imagenet-200/val",
    transform=transform_test
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)

# Visualize samples
examples = iter(test_loader)
example_data, example_targets = next(examples)

class_names = train_dataset.classes

plt.figure(figsize=(8, 5))

mean = torch.tensor([0.485, 0.456, 0.406])
std  = torch.tensor([0.229, 0.224, 0.225])

for i in range(6):
    plt.subplot(2, 3, i + 1)

    img = example_data[i].permute(1, 2, 0)
    img = img * std + mean
    img = img.clamp(0, 1)

    plt.imshow(img)
    plt.title(class_names[example_targets[i]])
    plt.axis("off")

plt.show()

# Model
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

model.fc = nn.Linear(2048, 200)
model = model.to(device)

# Freeze all layers first
for param in model.parameters():
    param.requires_grad = False

for param in model.fc.parameters():
    param.requires_grad = True

# Optional: Fine-tuning layer4
# for param in model.layer4.parameters():
#     param.requires_grad = True


# Loss and optimizer
cost = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Optional hyperparameter tuning
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=learning_rate,
#     momentum=0.9,
#     weight_decay=5e-4
# )

# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max= num_epochs
# )

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

    for idx, (test_x, test_label) in enumerate(test_loader):
        test_x = test_x.to(device)
        test_label = test_label.to(device)

        predict_y = model(test_x.float()).detach()
        predict_y = torch.argmax(predict_y, dim=-1)

        current_correct_num = predict_y == test_label
        all_correct_num += torch.sum(current_correct_num).item()
        all_sample_num += current_correct_num.shape[0]

    acc = (all_correct_num / all_sample_num) * 100
    print('Accuracy of the baseline model: {:.3f}'.format(acc))
    print("Model finished training")