import os
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler

# --- LSTM MODEL DEFINITION ---
class LoLLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        # Compression layer to handle the 1,818 features before LSTM
        self.bottleneck = nn.Linear(input_dim, 512)
        
        self.lstm = nn.LSTM(
            input_size=512,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x shape: (batch, seq_len, 1818)
        x = self.bottleneck(x) 
        lstm_out, _ = self.lstm(x)
        # Taking the last time step of the bidirectional output
        last_hidden = lstm_out[:, -1, :]
        return self.fc(last_hidden)

# --- 1. DATA PREPARATION ---
print("Loading and Pivoting Data...")
# (Assuming df_wide is already created from your previous pivoting logic)
df_wide = pd.read_csv('df_wide.csv') 

# Define features - exclude metadata
drop_cols = ['winning_team', 'match_id', 'minute']
feature_cols = [c for c in df_wide.columns if c not in drop_cols]
feature_count = len(feature_cols)

# --- 2. SEQUENCE GENERATION FUNCTION ---
def get_sequences(df, target_minute, seq_len=5):
    """
    Grabs the target_minute and the 4 minutes preceding it for every match.
    """
    sequences = []
    labels = []
    match_ids = []

    # Get matches that have at least the required history
    valid_matches = df[df['minute'] == target_minute]['match_id'].unique()
    
    for mid in valid_matches:
        # Extract the window [target - 4, target]
        window = df[(df['match_id'] == mid) & 
                    (df['minute'] <= target_minute) & 
                    (df['minute'] > target_minute - seq_len)]
        
        if len(window) == seq_len:
            # Sort by minute to ensure temporal order
            window = window.sort_values('minute')
            seq_data = window[feature_cols].values.astype('float32')
            sequences.append(seq_data)
            labels.append(window['winning_team'].iloc[0])
            match_ids.append(mid)
            
    return np.array(sequences), np.array(labels), np.array(match_ids)

# --- 3. TRAINING LOOP ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
minutes_to_train = [5, 10, 15, 20, 25]
SEQ_LEN = 5

for target_min in minutes_to_train:
    print(f"\n--- Training Phase: {target_min}m (LSTM True Sequence) ---")
    
    X, y, groups = get_sequences(df_wide, target_min, SEQ_LEN)
    if len(X) < 100: continue

    # Split by Match ID to prevent data leakage
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Scaling: Standardize across the feature dimension
    scaler = StandardScaler()
    # Reshape to 2D for scaler, then back to 3D
    X_train_reshaped = X_train.reshape(-1, feature_count)
    X_train_scaled = scaler.fit_transform(X_train_reshaped).reshape(X_train.shape)
    
    X_test_reshaped = X_test.reshape(-1, feature_count)
    X_test_scaled = scaler.transform(X_test_reshaped).reshape(X_test.shape)
    
    joblib.dump(scaler, f'scaler_lstm_{target_min}m.pkl')

    # Convert to Tensors
    train_ds = TensorDataset(torch.tensor(X_train_scaled), torch.tensor(y_train).reshape(-1,1))
    test_ds = TensorDataset(torch.tensor(X_test_scaled), torch.tensor(y_test).reshape(-1,1))
    
    train_loader = DataLoader(train_ds, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=128)

    model = LoLLSTM(input_dim=feature_count).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    criterion = nn.BCELoss()
    
    # Training
    best_acc = 0
    for epoch in range(50):
        model.train()
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            outputs = model(b_X)
            loss = criterion(outputs, b_y)
            loss.backward()
            torch.nn.utils.clip_grad