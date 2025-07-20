import torch
import numpy as np
import pandas as pd
from torch import nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
#import seaborn as sns
import matplotlib.image as mpimg
import os
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from torch.utils.data import random_split

# Set paths
base_dir = 'classify-leaves'
train_df = pd.read_csv(os.path.join(base_dir, "train.csv"))
val_df = pd.read_csv(os.path.join(base_dir, "test.csv"))
save_path = 'best_model.pth'

# Parameters
num_classes = train_df['label'].nunique()
batch_size = 128

# Label encoding
unique_labels = train_df["label"].unique()
label2idx = {label: idx for idx, label in enumerate(unique_labels)}
idx2label = {v: k for k, v in label2idx.items()}

# Transforms
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
])

# Custom Dataset
class LeaveDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        label_name = self.data.iloc[idx, 1]
        label = label2idx[label_name]
        return image, label

class LeaveValDataset(Dataset):
    def __init__(self, data_df, transform=None):
        self.data = data_df
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = os.path.join(base_dir, self.data.iloc[idx, 0])
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        return image

def show_data(base_path):


    # count each categories num
    label_counts = train_df['label'].value_counts()

    print(train_df.shape, '\n')
    print(train_df.info(), '\n')
    print(train_df.describe(), "\n")
    print(train_df.head(), '\n')
    return label_counts

def show_piechart():
    plt.figure(figsize=(8, 8))
    label_counts = show_data(base_dir)
    plt.pie(

        label_counts.values[:10],
        labels=label_counts.index[:10],
        autopct='%1.1f%%',  # show percentage
        colors=plt.cm.Paired.colors,  # give colors
        startangle=140,
        wedgeprops={'edgecolor': 'black'}  # add frames
    )

    # 设置标题
    plt.title("leaves pie")

    # 显示图像
    plt.show()


# get leaf categorise
unique_labels = train_df["label"].unique()

# create label -> idx
label2idx = {label: idx for idx, label in enumerate(unique_labels)}

# construct inverse relations（id → label）
idx2label = {v: k for k, v in label2idx.items()}

def build_loader():
    train_dataset = LeaveDataset(train_df,transform)

    # split train and test
    total_size = len(train_dataset)
    test_size = int(0.2 * total_size)  # 20% as test
    train_size = total_size - test_size  # 80% as train

    # randomly seperate data
    train_subset, test_subset = random_split(train_dataset, [train_size, test_size])

    # rebuild DataLoader
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("total train data number:", len(train_loader))
    return train_loader, test_loader

# use gpu to train otherwice cpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def model(num_classes):
    # load ResNet-18（can change to resnet34, resnet50, resnet101, resnet152）
    resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

    resnet = resnet.to(device)

    # define loss（for classification）
    criterion = nn.CrossEntropyLoss()

    # define learning rate and momentum
    lr,momentum = 0.01,0.9

    # define optimizer
    optimizer = optim.SGD(resnet.parameters(), lr=lr, momentum=momentum)
    return resnet, criterion, optimizer


# single train
def train(resnet, criterion, optimizer, train_loader):
    resnet.train()
    batch_nums = len(train_loader)
    size = len(train_loader.dataset)
    train_loss, correct = 0.0, 0.0
    p = tqdm(train_loader, desc="Training", unit="batch")

    for X, y in p:
        X, y = X.to(device), y.to(device)
        pred = resnet(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        p.set_postfix(loss=f"{loss.item():.6f}")
        train_loss += loss.item()
        correct += (pred.argmax(1) == y).sum().item()

    train_loss /= batch_nums
    correct /= size
    print(f"Train Accuracy: {(100 * correct):.2f}%, Train Avg loss: {train_loss:.6f}")
    return train_loss, correct


# test
def test(resnet, criterion, test_loader):
    resnet.eval()
    batch_nums = len(test_loader)
    size = len(test_loader.dataset)
    test_loss, correct = 0.0, 0.0

    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            pred = resnet(X)
            loss = criterion(pred, y)
            test_loss += loss.item()
            correct += (pred.argmax(1) == y).sum().item()

    test_loss /= batch_nums
    correct /= size
    print(f"Test Accuracy: {(100 * correct):.2f}%, Test Avg loss: {test_loss:.6f}")
    return test_loss, correct

def get_best_model():
    # get trained model
    resnet, criterion, optimizer = model(num_classes)
    # train outcomes
    train_losses,train_accs = [],[]

    # test outcomes
    test_losses ,test_accs= [],[]

    epochs = 20

    best_acc = 0.0  # get best acc

    # get dataloader
    train_loader, test_loader = build_loader()
    for epoch in range(epochs):
        print(f"Epoch {epoch+1}")
        train_loss, train_acc = train(resnet, criterion, optimizer, train_loader)
        train_losses.append(train_loss)
        train_accs.append(train_acc)

        test_loss,test_acc = test(resnet, criterion, test_loader)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

        # save best model
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(resnet.state_dict(), save_path)
            print(f'New best model saved with accuracy: {best_acc:.4f}')

        print("-"*30)
    return train_losses,train_accs,test_losses,test_accs

def show_training_process():
    train_losses,train_accs,test_losses ,test_accs = get_best_model()
    plt.figure(figsize=(8, 4))
    plt.plot(train_losses, "r-", label="train_losses")
    plt.plot(train_accs, "r--", label="train_accs")
    plt.plot(test_losses, "b-", label="test_losses")
    plt.plot(test_accs, "b--", label="test_accs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss/Accuracy")
    plt.legend()
    plt.show()

def pred_data():

    # redefine model
    resnet = models.resnet18()
    resnet.fc = torch.nn.Linear(resnet.fc.in_features, num_classes)

    # load params doc
    resnet.load_state_dict(torch.load(save_path))

    # change to eval model
    resnet.to(device)
    resnet.eval()


    # single img pred
    img_val_path = os.path.join(base_dir,val_df['image'][0])
    image_val = Image.open(img_val_path)
    image_val_tensor = transform(image_val).unsqueeze(0)  #

    # pred
    with torch.no_grad():
        image_val_tensor = image_val_tensor.to(device)
        output = resnet(image_val_tensor)
        probabilities = F.softmax(output, dim=1)
        pred_class = torch.argmax(probabilities).item()

    print(torch.max(probabilities).item()*100,'%')
    print(idx2label[pred_class])
    return resnet


def batch_pred():
    val_dataset = LeaveValDataset(val_df, transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    print("val batch num:", len(val_loader))

    all_preds = []
    resnet = pred_data()
    with torch.no_grad():
        for inputs in tqdm(val_loader):
            inputs = inputs.to(device)
            outputs = resnet(inputs)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            all_preds.extend(preds.cpu().numpy())

    print("All pred length: ", len(all_preds))



if __name__ == "__main__":
    # run
    show_training_process()
    batch_pred()
