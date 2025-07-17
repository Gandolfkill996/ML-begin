import os
import torch
from torch import nn
from torch.nn import functional as F
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from torchvision.io import read_image
from torch.utils.data import Dataset
import pandas as pd
import matplotlib.pyplot as plt
from d2l import torch as d2l

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class LeafDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data_frame['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_file = os.path.basename(self.data_frame.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_file)
        image = read_image(img_path).float() / 255.0
        label_str = self.data_frame.iloc[idx, 1]
        label = self.class_to_idx[label_str]
        if self.transform:
            image = self.transform(image)
        return image, label


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels,
                 use_1x1conv=False, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels,
                               kernel_size=3, padding=1, stride=strides)
        self.conv2 = nn.Conv2d(num_channels, num_channels,
                               kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels,
                                   kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        if self.conv3:
            X = self.conv3(X)
        Y += X
        return F.relu(Y)


def resnet_block(in_channels, out_channels, num_residuals, first_block=False):
    blk = []
    for i in range(num_residuals):
        if i == 0 and not first_block:
            blk.append(Residual(in_channels, out_channels, use_1x1conv=True, strides=2))
        else:
            blk.append(Residual(out_channels, out_channels))
    return nn.Sequential(*blk)

def build_model(num_classes):
    net = nn.Sequential(
        nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
        nn.BatchNorm2d(64), nn.ReLU(),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        resnet_block(64, 64, 2, first_block=True),
        resnet_block(64, 128, 2),
        resnet_block(128, 256, 2),
        resnet_block(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)  # 4 classes in your dataset
    )
    return net

def plot_training(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Training visualization
def plot_training(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Loss')
    plt.title('Training Loss')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Accuracy')
    plt.title('Training Accuracy')
    plt.legend()
    plt.tight_layout()
    plt.show()

# Main training loop
def train_model(net, train_loader, lr, num_epochs):
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    train_losses, train_accuracies = [], []

    for epoch in range(num_epochs):
        net.train()
        total_loss, correct, total = 0.0, 0, 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            loss = loss_fn(y_hat, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * X.shape[0]
            correct += (y_hat.argmax(dim=1) == y).sum().item()
            total += y.numel()

        avg_loss = total_loss / total
        acc = correct / total
        train_losses.append(avg_loss)
        train_accuracies.append(acc)
        print(f"Epoch {epoch+1}, Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")

    plot_training(train_losses, train_accuracies)
    torch.save(net.state_dict(), "leaf_resnet.pth")
    print("✅ model saved as leaf_resnet.pth")

# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomResizedCrop((224, 224), scale=(0.1, 1.0), ratio=(0.5, 2)),
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.5),
    transforms.ConvertImageDtype(torch.float),  # change to float32 [0,1]
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])


def check_img_info():
    # img info:
    # img shape: torch.Size([3, 224, 224])
    # pixel examples: tensor([254, 254, 254], dtype=torch.uint8)
    img = read_image("classify-leaves/images/0.jpg")
    print(f"img shape: {img.shape}")  # output: num of panels, height, width
    print(f"pixel values examples: {img[:, 0, 0]}")  # print left-top pixel values

if __name__ == "__main__":
    csv_path = "classify-leaves/train.csv"
    img_dir = "classify-leaves/images"
    batch_size, lr, num_epochs = 32, 0.001, 50

    train_dataset = LeafDataset(csv_path, img_dir, transform=train_transforms)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    num_classes = len(train_dataset.classes)

    net = build_model(num_classes)
    train_model(net, train_loader, lr, num_epochs)


