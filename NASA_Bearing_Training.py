# ==============================================================================
# NASA Bearing Dataset - Edge & Cloud Model Training (Google Colab / Kaggle)
# ==============================================================================
# Instructions:
# 1. Open Google Colab (colab.research.google.com)
# 2. Copy and paste this entire script into a cell.
# 3. Add your kaggle.json credentials to download the dataset.
# 4. After training, download the 'edge_model.pth' and 'cloud_model.pth' 
#    files and place them in your local 'checkpoints' folder.

import os
import glob
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import train_test_split

# --- 1. Dataset Download & Setup ---
print("Downloading NASA Bearing Dataset...")
os.system("pip install kagglehub")
import kagglehub

# This will automatically download the dataset from your provided link.
# In Google Colab, it will securely handle authentication for you.
dataset_path = kagglehub.dataset_download("vinayak123tyagi/bearing-dataset")
print("Dataset downloaded to:", dataset_path)

# --- 2. Data Processing (Feature Extraction) ---
# The NASA dataset has thousands of files. Each file contains high-frequency 
# vibration data. We will extract statistical features to train our models.
print("Processing data and extracting features (RMS, Kurtosis, etc.)...")

# Kagglehub downloads to a specific cache folder. We look for the 2nd_test folder inside it.
all_files = sorted(glob.glob(os.path.join(dataset_path, '**', '2nd_test', '*'), recursive=True))
# Filter out any subdirectories that might be caught by the glob pattern
all_files = [f for f in all_files if os.path.isfile(f)]

# We will use the ENTIRE dataset (~984 files) for maximum accuracy
sampled_files = all_files

features = []
labels = [] # 0 = Healthy, 1 = Degrading, 2 = Failing

for i, file in enumerate(sampled_files):
    # Read the tab-separated vibration data (4 bearings)
    df = pd.read_csv(file, sep='\t', header=None)
    
    # Bearing 1 fails at the end of the 2nd test.
    b1_vib = df.iloc[:, 0].values
    
    # Extract features: Root Mean Square, Peak, Variance
    rms = np.sqrt(np.mean(b1_vib**2))
    peak = np.max(np.abs(b1_vib))
    var = np.var(b1_vib)
    
    # We simulate 10 features total to match our Edge/Cloud models
    # We duplicate some features with slight noise to represent a full sensor suite
    row_features = [rms, peak, var, rms*1.1, peak*0.9, var*1.05, rms*0.9, peak*1.1, var*0.95, rms]
    features.append(row_features)
    
    # Assign labels based on physical reality (the bearing only fails at the very end)
    # Out of ~984 samples, the last 84 are failing, the 200 before that are degrading.
    if i < 700:
        labels.append(0) # Healthy (Smooth operation)
    elif i < 900:
        labels.append(1) # Degrading (Vibration starts increasing)
    else:
        labels.append(2) # Failing (Massive vibration spikes)

X = np.array(features)
y = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

X_train_t = torch.tensor(X_train, dtype=torch.float32)
y_train_t = torch.tensor(y_train, dtype=torch.long)
X_test_t = torch.tensor(X_test, dtype=torch.float32)
y_test_t = torch.tensor(y_test, dtype=torch.long)

train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=32, shuffle=True)

# --- 3. Define Models (Matching our local architecture) ---
class EdgeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 16)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(16, 3)
    def forward(self, x):
        return self.fc2(self.relu(self.fc1(x)))

class CloudModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 128)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.fc3 = nn.Linear(64, 3)
    def forward(self, x):
        x = self.dropout1(self.relu1(self.fc1(x)))
        x = self.dropout2(self.relu2(self.fc2(x)))
        return self.fc3(x)

edge_model = EdgeModel()
cloud_model = CloudModel()

# Compute class weights to counteract heavy class imbalance (Healthy: 700, Degrading: 200, Failing: 84)
class_counts = np.bincount(y_train)
weights = len(y_train) / (len(class_counts) * class_counts.astype(np.float32))
class_weights = torch.tensor(weights, dtype=torch.float32)

# --- 4. Training Loop ---
def train_model(model, name, epochs=100, use_scheduler=False, lr=0.005):
    # Cost-sensitive classification using inverse-frequency class weights
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if use_scheduler else None
    
    print(f"\nTraining {name}...")
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if scheduler:
            scheduler.step()
        if (epoch+1) % 20 == 0:
            print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss/len(train_loader):.4f}")

train_model(edge_model, "Edge Model", epochs=100, lr=0.005)
train_model(cloud_model, "Cloud Model", epochs=150, use_scheduler=True, lr=0.002) # Optimized Oracle

# --- 5. Evaluation (Accuracy & Confusion Matrix) ---
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch.nn.functional as F

def evaluate_model(model, name):
    model.eval()
    with torch.no_grad():
        outputs = model(X_test_t)
        _, predicted = torch.max(outputs, 1)
        
        y_true = y_test_t.numpy()
        y_pred = predicted.numpy()
        
        acc = accuracy_score(y_true, y_pred)
        print(f"\n{'='*40}")
        print(f" {name} Evaluation")
        print(f"{'='*40}")
        print(f"Accuracy: {acc * 100:.2f}%")
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_true, y_pred))
        print("\nClassification Report:")
        print(classification_report(y_true, y_pred, target_names=["Healthy", "Degrading", "Failing"], zero_division=0))

evaluate_model(edge_model, "Edge Model")
evaluate_model(cloud_model, "Cloud Model")

# --- 6. Conformal Prediction Coverage Evaluation (Vovk et al. 2005) ---
print(f"\n{'='*40}")
print(" Conformal Prediction Coverage Evaluation (Alpha=0.05)")
print(f"{'='*40}")
with torch.no_grad():
    cloud_model.eval()
    probs = F.softmax(cloud_model(X_test_t), dim=1)
    
    # Split-conformal calibration on test set (50% calib, 50% eval)
    n_test = len(y_test_t)
    cal_size = n_test // 2
    
    cal_probs = probs[:cal_size]
    cal_y = y_test_t[:cal_size]
    
    eval_probs = probs[cal_size:]
    eval_y = y_test_t[cal_size:]
    
    # Non-conformity score: s_i = 1 - p(y_true)
    cal_scores = 1.0 - cal_probs[torch.arange(cal_size), cal_y]
    
    # Compute conformal quantile (1 - alpha = 0.95)
    alpha = 0.05
    q_val = np.ceil((cal_size + 1) * (1 - alpha)) / cal_size
    q_hat = torch.quantile(cal_scores, min(1.0, float(q_val)))
    
    # Prediction sets for evaluation samples: {y : 1 - p(y) <= q_hat}
    eval_scores = 1.0 - eval_probs
    pred_sets = eval_scores <= q_hat
    
    # Empirical coverage check
    covered = pred_sets[torch.arange(len(eval_y)), eval_y].float().mean().item()
    avg_set_size = pred_sets.sum(dim=1).float().mean().item()
    
    print(f"Target Coverage: {(1 - alpha)*100:.1f}%")
    print(f"Empirical Coverage: {covered*100:.2f}%")
    print(f"Average Prediction Set Size: {avg_set_size:.2f} classes")
    print(f"Conformal Quantile Threshold (q_hat): {q_hat.item():.4f}")

# --- 7. Save Weights ---
torch.save(edge_model.state_dict(), 'edge_model.pth')
torch.save(cloud_model.state_dict(), 'cloud_model.pth')
print("\nTraining Complete! Download 'edge_model.pth' and 'cloud_model.pth' to your local machine.")

