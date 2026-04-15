# Importing the required libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torchvision.models import resnet50, ResNet50_Weights
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# Device configuration - deciding whether to run the training on cpu or gpu
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper-parameters
batch_size = 64
learning_rate = 1e-4
num_epochs = 10


# CIFAR-10 dataset transforms
transform_train = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])

transform_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        (0.4914, 0.4822, 0.4465),
        (0.2023, 0.1994, 0.2010)
    )
])


# Download training and testing data
train_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=True,
    download=True,
    transform=transform_train
)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data',
    train=False,
    download=True,
    transform=transform_test
)

# Load train and test data
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=2
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2
)


# View some images from dataset
examples = iter(test_loader)
example_data, example_targets = next(examples)

class_names = train_dataset.classes

plt.figure(figsize=(8, 5))

mean = torch.tensor([0.4914, 0.4822, 0.4465])
std  = torch.tensor([0.2023, 0.1994, 0.2010])

for i in range(6):
    plt.subplot(2, 3, i + 1)

    img = example_data[i].permute(1, 2, 0)  # C,H,W → H,W,C
    img = img * std + mean                  # unnormalize
    img = img.clamp(0, 1)

    plt.imshow(img)
    plt.title(class_names[example_targets[i]])
    plt.axis("off")

plt.show()


# Load pretrained ResNet50
model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)

# Replace classifier
model.fc = nn.Linear(2048, 10)  # CIFAR-10 classes
model = model.to(device)

# Freeze all layers
for param in model.parameters():
    param.requires_grad = False

# Unfreeze final layer
for param in model.fc.parameters():
    param.requires_grad = True

# Optional fine-tuning (commented)
# for param in model.layer3.parameters():
#     param.requires_grad = True
#
# for param in model.layer4.parameters():
#     param.requires_grad = True


# Loss and optimizer
cost = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=learning_rate
)

# Optional SGD
# optimizer = torch.optim.SGD(
#     model.parameters(),
#     lr=learning_rate,
#     momentum=0.9,
#     weight_decay=5e-4
# )

# Optional scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
#     optimizer,
#     T_max=num_epochs
# )

total_steps = len(train_loader)


# Training loop
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

    # scheduler.step()

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

    print('Accuracy of the baseline model: {:.3f}'.format(acc), flush=True)
    print("Model finished training")