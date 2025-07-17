import pandas as pd
import numpy as np
import shap
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

# --- Hyperparameters ---
SEQ_LEN = 10
BATCH_SIZE = 64
EPOCHS = 20
LR = 0.001

# --- 1. Load and Prepare Data ---
df = pd.read_csv("ohlcv_dummy.csv")  # Expect columns: Open, High, Low, Close, Volume
df = df.dropna()

features = ['Open', 'High', 'Low', 'Close', 'Volume']
scaler = MinMaxScaler()
scaled = scaler.fit_transform(df[features])

# --- 2. Create Sequences ---
def create_sequences(data, seq_len):
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i:i+seq_len])
        y.append(data[i+seq_len][3])  # Close price
    return np.array(X), np.array(y)

X, y = create_sequences(scaled, SEQ_LEN)

X_train = torch.tensor(X[:-200], dtype=torch.float32)
y_train = torch.tensor(y[:-200], dtype=torch.float32)
X_test = torch.tensor(X[-200:], dtype=torch.float32)
y_test = torch.tensor(y[-200:], dtype=torch.float32)

# --- 3. Define LSTM Model ---
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]  # Take last output
        out = self.fc(out)
        return out.squeeze()

model = LSTMModel(input_size=5)
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

# --- 4. Train Model ---
model.train()
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output = model(X_train)
    loss = loss_fn(output, y_train)
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}/{EPOCHS} - Loss: {loss.item():.4f}")

# --- 5. SHAP Explainability (DeepExplainer for LSTM) ---
model.eval()
background = X_train[:100]
test_samples = X_test[:10]

explainer = shap.DeepExplainer(model, background)
shap_values = explainer.shap_values(test_samples)

# --- 6. SHAP Visualization ---
# Plot SHAP values for one test sample
shap_vals = shap_values[0][0]  # First sample, first feature group (LSTM output)
plt.figure(figsize=(10, 6))
plt.title("SHAP Feature Contributions (LSTM Forecast)")
plt.plot(shap_vals)
plt.xlabel("Time Step")
plt.ylabel("SHAP Value")
plt.grid(True)
plt.show()
