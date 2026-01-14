import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
from sklearn.preprocessing import StandardScaler

# Load data
df = pd.read_csv("gameplay_data_cleaned.csv")  # Use your actual cleaned CSV file name

# Split features and labels
X = df.iloc[:, 1:16].values  # Game state
y = df.iloc[:, 16:].values   # Button labels

# Normalize input features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Create dataset
dataset = TensorDataset(X_tensor, y_tensor)

# Train-test split
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_set, test_set = random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
test_loader = DataLoader(test_set, batch_size=32)

# Define model
class ImprovedBotModel(nn.Module):
    def __init__(self):
        super(ImprovedBotModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(15, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            nn.Sigmoid()  # for multi-label classification
        )

    def forward(self, x):
        return self.net(x)

model = ImprovedBotModel()

# Calculate class weights for imbalance
pos_weights = torch.tensor([
    (len(y) - y[:, i].sum()) / y[:, i].sum()
    for i in range(y.shape[1])
], dtype=torch.float32)

# Loss & optimizer
criterion = nn.BCELoss()  # BCEWithLogitsLoss can be used if removing final Sigmoid
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Train
EPOCHS = 25
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for xb, yb in train_loader:
        output = model(xb)
        loss = criterion(output, yb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {total_loss / len(train_loader):.4f}")

# Save
torch.save(model.state_dict(), "improved_bot_model.pt")
print("✅ Model saved as 'improved_bot_model.pt'")
