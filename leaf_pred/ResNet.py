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
from sklearn.model_selection import train_test_split

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Custom Dataset
class LeafDataset(Dataset):
    def __init__(self, csv_file_or_df, img_dir, transform=None):
        if isinstance(csv_file_or_df, str):
            self.data_frame = pd.read_csv(csv_file_or_df)
        else:
            self.data_frame = csv_file_or_df  # assume it's a DataFrame

        self.img_dir = img_dir
        self.transform = transform
        self.classes = sorted(self.data_frame['label'].unique())
        self.class_to_idx = {cls: i for i, cls in enumerate(self.classes)}

    def __len__(self):  # 
        return len(self.data_frame)

    def __getitem__(self, idx):
        img_file = os.path.basename(self.data_frame.iloc[idx, 0])
        img_path = os.path.join(self.img_dir, img_file)
        image = read_image(img_path)
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
        nn.Dropout(0.3),
        resnet_block(64, 128, 2),
        nn.Dropout(0.3),
        resnet_block(128, 256, 2),
        nn.Dropout(0.3),
        resnet_block(256, 512, 2),
        nn.AdaptiveAvgPool2d((1, 1)),
        nn.Flatten(),
        nn.Linear(512, num_classes)
    )
    return net


# Training visualization
def plot_training(train_losses, train_accuracies, val_accuracies):
    epochs = range(1, len(train_losses) + 1)
    plt.figure(figsize=(10, 4))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'r-', label='Train Loss')
    plt.title('Loss')
    plt.legend()

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Train Acc')
    plt.plot(epochs, val_accuracies, 'g--', label='Val Acc')
    plt.title('Accuracy')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
# Main training loop
def train_model(net, train_loader, val_loader, lr, num_epochs, patience=7):
    net.to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0

    train_losses, train_accuracies = [], []
    val_accuracies = []

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

        avg_train_loss = total_loss / total
        train_acc = correct / total
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_acc)


        # 验证阶段
        net.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for X_val, y_val in val_loader:
                X_val, y_val = X_val.to(device), y_val.to(device)
                y_hat_val = net(X_val)
                val_loss += loss_fn(y_hat_val, y_val).item() * X_val.shape[0]
                val_correct += (y_hat_val.argmax(1) == y_val).sum().item()
                val_total += y_val.numel()
        avg_val_loss = val_loss / val_total
        val_acc = val_correct / val_total
        val_accuracies.append(val_acc)

        print(f"Epoch {epoch+1}: Train Loss {avg_train_loss:.4f}, Acc {train_acc:.4f} | Val Loss {avg_val_loss:.4f}, Acc {val_acc:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = net.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("⛔ Early stopping triggered")
                break

        scheduler.step()

    # 保存最好的模型
    if best_model_state:
        net.load_state_dict(best_model_state)
        torch.save(net.state_dict(), "leaf_resnet.pth")
        print("✅ Best model saved as leaf_resnet.pth")
    plot_training(train_losses, train_accuracies, val_accuracies)
# Data augmentation and preprocessing
train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ConvertImageDtype(torch.float),
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

    df = pd.read_csv(csv_path)
    train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)
    
    train_dataset = LeafDataset(train_df.reset_index(drop=True), img_dir, transform=train_transforms)
    val_dataset = LeafDataset(val_df.reset_index(drop=True), img_dir, transform=train_transforms)  #  transform
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    num_classes = len(train_dataset.classes)
    net = build_model(num_classes)
    train_model(net, train_loader, val_loader, lr, num_epochs)

    



