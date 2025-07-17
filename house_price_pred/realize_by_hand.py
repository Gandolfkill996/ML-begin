import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import os
from datetime import datetime


TRAIN_FILE = "train.csv"
TEST_FILE = "test.csv"
SUBMISSION_FILE = "sample_submission.csv"
TARGET = "Sold Price"
DROP_COLS = ["Address", "Summary", "Listed Price", "Last Sold Price", "Zip"]
EPOCHS = 200
BATCH_SIZE = 64
LR = 0.01

# ==== data preprocessing ====
def preprocess(filepath, is_train=True):
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()  # remove columns' names spaces
    ids = df["Id"].values  # keep ID

    # delete not used columns
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # date data processing
    for date_col in ["Listed On", "Last Sold On"]:
        if date_col in df.columns:
            df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
            df[date_col] = (datetime.today() - df[date_col]).dt.days
            df[date_col].fillna(df[date_col].median(), inplace=True)

    df.fillna({
        col: df[col].median() if df[col].dtype != 'object' else 'Unknown'
        for col in df.columns
    }, inplace=True)

    if is_train:
        y = df[TARGET].values
        X = df.drop(columns=[TARGET])
    else:
        y = None
        X = df.drop(columns=[TARGET], errors="ignore")

    # category data processing
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return (
        torch.tensor(X_scaled, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).view(-1, 1) if is_train else None,
        X.shape[1],
        ids
    )


# ==== model structure ====
class PriceModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(32, 1)
        )
    def forward(self, x):
        return self.net(x)



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load train_data
    X_train, y_train, input_dim, _ = preprocess(TRAIN_FILE, is_train=True)
    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)

    # initialize model
    model = PriceModel(input_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    # train process
    losses = []
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        for Xb, yb in train_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            pred = model(Xb)
            loss = loss_fn(pred, yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        losses.append(avg_loss)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {avg_loss:.4f}")

    # draw loss plot
    plt.plot(losses, label='Train Loss')
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Training Loss Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("training_loss.png")
    plt.show()

    # load & predict test data
    X_test, _, _, ids = preprocess(TEST_FILE, is_train=False)
    model.eval()
    with torch.no_grad():
        X_test = X_test.to(device)
        preds = model(X_test).cpu().numpy().flatten()

    # write into submission.csv
    submission = pd.DataFrame({
        "Id": ids,
        "Sold Price": preds.astype(int)
    })
    submission.to_csv(SUBMISSION_FILE, index=False)
    print(f"\n✅ Prediction finished，results saved in {SUBMISSION_FILE}")
    # keep model parameters
    torch.save(model.state_dict(), "house_price_model.pth")
    print("🎯 model's parameters have been saved in house_price_model.pth")


if __name__ == "__main__":
    main()