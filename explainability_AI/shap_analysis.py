import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import shap
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Config
SEQ_LEN = 10
EPOCHS = 20
LR = 0.001
HIDDEN_SIZE = 64

# Load data
def load_scaled_data(path, scaler=None, fit=False):
    df = pd.read_csv(path).dropna()
    features = ['Open', 'High', 'Low', 'Close', 'Volume']
    data = df[features].values
    if fit:
        scaler = MinMaxScaler()
        data = scaler.fit_transform(data)
        return data, scaler
    else:
        return scaler.transform(data), scaler

train_data, scaler = load_scaled_data("../data/train_data.csv", fit=True)
test_data, _ = load_scaled_data("../data/test_data.csv", scaler)

# Create sequences
def create_sequences(data, seq_len=10):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][3])  # 'Close' column
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_data, SEQ_LEN)
X_test, y_test = create_sequences(test_data, SEQ_LEN)

X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# LSTM Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Last timestep output
        return self.fc(out)

model = LSTMModel()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# Training
model.train()
loss_history = []
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    preds = model(X_train)
    loss = loss_fn(preds, y_train)
    loss.backward()
    optimizer.step()
    loss_value = loss.item()
    loss_history.append(loss_value)
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss.item():.4f}")

# --- Plot Training Loss ---
plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o', linestyle='-')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("MSE Loss")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/training_loss.png")
plt.show()

# SHAP Explainability
model.eval()
background = X_train[:100]
test_sample = X_test[:10]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_sample)

# Plot SHAP values for the first test sample
shap_vals = shap_values[0][0]  # First sample
plt.figure(figsize=(12, 6))
plt.title("SHAP Value Contributions (Close Price Forecast)")
plt.plot(shap_vals)
plt.xlabel("Time Step")
plt.ylabel("SHAP Value")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/shap_values.png")
plt.show()


shap_sample = shap_values[0][0]
FEATURES = ['Open', 'High', 'Low', 'Close', 'Volume']
shap_df = pd.DataFrame(shap_sample, columns=FEATURES)

# Aggregate SHAP values over time
feature_shap_importance = shap_df.abs().mean()

# Plot bar chart of average SHAP value per feature
plt.figure(figsize=(8, 5))
feature_shap_importance.sort_values().plot(kind='barh', color='skyblue')
plt.title("Average SHAP Value per Feature (Sample 1)")
plt.xlabel("Mean |SHAP Value|")
plt.tight_layout()
plt.grid(True)
plt.show()
