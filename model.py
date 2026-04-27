import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import StandardScaler
import joblib

# --- LSTM MODEL DEFINITION ---

class LoLLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=2, dropout=0.3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )
        
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 64),
            nn.LayerNorm(64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        lstm_out, _ = self.lstm(x)
        
        last_hidden = lstm_out[:, -1, :]
        
        last_hidden = self.layer_norm(last_hidden)
        
        out = self.fc(last_hidden)
        return out

# 1. DATA INGESTION
print("Loading data...")
df_matches = pd.read_csv(r'data/matches.csv')
df_players = pd.read_csv(r'processed_summoner_data.csv')
df_intervals = pd.read_csv(r'data/intervals.csv')

# 2. RELATIONAL PRE-PROCESSING
print("Merging and cleaning...")
df = df_intervals.merge(df_players, left_on='player_id', right_on='id', suffixes=('', '_drop'))
df = df.merge(df_matches, on='match_id', suffixes=('', '_match_drop'))

ban_cols = [c for c in df.columns if '_is_banned' in c]
df = df.drop(columns=ban_cols)

df['winning_team'] = df['winning_team'].replace({100: 0, 200: 1})

# 3. DUMMY ENCODING
print("Encoding champion identities...")
df = pd.get_dummies(df, columns=['champion'], dtype=int)

# 4. MATCH-WIDE VECTORIZATION
print("Pivoting to Match-Wide format...")

role_order = {'TOP': 0, 'JUNGLE': 1, 'MIDDLE': 2, 'BOTTOM': 3, 'UTILITY': 4}
df['slot'] = df['role'].map(role_order).fillna(0).astype(int)
df.loc[df['team_id'] == 200, 'slot'] += 5

match_level_cols = [
    'match_id', 'minute', 'team_kills', 'team_inhibitors', 'team_towers', 
    'team_dragons', 'team_barons', 'team_void_grubs', 'team_heralds', 'winning_team'
]

player_level_stats = ['total_gold', 'xp', 'level', 'kills', 'deaths', 'assists', 'cs', 'jungle_cs']
champ_cols = [c for c in df.columns if c.startswith('champion_')]
player_level_stats.extend(champ_cols)

df_wide = df.pivot_table(
    index=[c for c in match_level_cols if c in df.columns], 
    columns='slot', 
    values=player_level_stats,
    aggfunc='first'
).fillna(0)

df_wide.columns = [f'{col}_{int(slot)}' for col, slot in df_wide.columns]
df_wide = df_wide.reset_index()

# 5. MACRO FEATURE ENGINEERING
print("Engineering macro-objective features...")

for i in range(5):
    df_wide[f'gold_diff_lane_{i}'] = df_wide[f'total_gold_{i}'] - df_wide[f'total_gold_{i+5}']
    df_wide[f'xp_diff_lane_{i}'] = df_wide[f'xp_{i}'] - df_wide[f'xp_{i+5}']

if 'team_dragons' in df_wide.columns:
    df_wide['at_soul_point'] = (df_wide['team_dragons'].abs() >= 3).astype(int)

df_wide.to_csv('df_wide.csv', index=False)

keywords = ['gold', 'xp', 'cs', 'level', 'kills', 'deaths', 'diff', 'team_']
continuous_cols = [c for c in df_wide.columns if any(x in c.lower() for x in keywords)]

# 7. TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
minutes_to_train = [5, 10, 15, 20, 25]
SEQ_LEN = 5

for target_min in minutes_to_train:
    df_m = df_wide[df_wide['minute'] == target_min].copy()
    if len(df_m) < 50: continue

    print(f"\n--- Training Phase: {target_min} Minutes (LSTM, seq_len={SEQ_LEN}) ---")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_m, groups=df_m['match_id']))
    
    train_df = df_m.iloc[train_idx].copy()
    test_df = df_m.iloc[test_idx].copy()

    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols]).astype('float32')
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols]).astype('float32')

    joblib.dump(scaler, f'scaler_{target_min}m.pkl')

    drop_cols = ['winning_team', 'match_id', 'minute']
    feature_cols = [c for c in df_wide.columns if c not in drop_cols]
    feature_count = len(feature_cols)
    
    X_train_raw = train_df[feature_cols].values.astype('float32')
    y_train_raw = train_df['winning_team'].values.astype('float32')
    X_test_raw = test_df[feature_cols].values.astype('float32')
    y_test_raw = test_df['winning_team'].values.astype('float32')

    print(f"Training samples: {len(X_train_raw)}, Feature count: {feature_count}")
    print(f"X dtype: {X_train_raw.dtype}")
    
    model = LoLLSTM(input_dim=feature_count, hidden_dim=256, num_layers=2, dropout=0.3).to(device)
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    criterion = nn.BCELoss()

    best_acc = 0
    patience_counter = 0
    max_patience = 15
    batch_size = 128
    
    for epoch in range(100):
        model.train()
        train_correct = 0
        train_total = 0
        
        indices = np.random.permutation(len(X_train_raw))
        for i in range(0, len(indices), batch_size):
            batch_idx = indices[i:i+batch_size]
            X_batch = X_train_raw[batch_idx].astype(np.float32)
            y_batch = y_train_raw[batch_idx].astype(np.float32)
            
            seq_batch = np.zeros((len(X_batch), SEQ_LEN, feature_count), dtype=np.float32)
            for b in range(len(X_batch)):
                seq_batch[b] = X_batch[b]
            
            X_tensor = torch.tensor(seq_batch, dtype=torch.float32).to(device)
            y_tensor = torch.tensor(y_batch.reshape(-1, 1), dtype=torch.float32).to(device)
            
            optimizer.zero_grad()
            preds = model(X_tensor)
            loss = criterion(preds, y_tensor)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_correct += (preds.round() == y_tensor).sum().item()
            train_total += len(y_batch)
        
        train_acc = train_correct / train_total
        
        model.eval()
        with torch.no_grad():
            test_seq = np.zeros((len(X_test_raw), SEQ_LEN, feature_count), dtype=np.float32)
            for b in range(len(X_test_raw)):
                test_seq[b] = X_test_raw[b]
            
            val_preds = model(torch.tensor(test_seq, dtype=torch.float32).to(device)).round()
            y_test_tensor = torch.tensor(y_test_raw, dtype=torch.float32).to(device)
            val_acc = (val_preds == y_test_tensor.reshape(-1, 1)).sum().item() / len(y_test_raw)
            
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'lol_model_{target_min}m.pth')
                patience_counter = 0
            else:
                patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")
        
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Val Acc: {best_acc:.2%}")

print("\nTraining complete!")