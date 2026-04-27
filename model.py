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

# Improved Model with Modern Techniques

class ResidualBlock(nn.Module):
    """Residual block with skip connection"""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.act = nn.GELU()
        
    def forward(self, x):
        return self.act(x + self.net(x))

class ImprovedLoLNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, 256), # Reduced from 512
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5) # Increased dropout
        )
        # Residual blocks
        self.res1 = ResidualBlock(256)
        self.res2 = ResidualBlock(256)
        self.res3 = ResidualBlock(256)
        
        # Output head
        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = self.input_proj(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.res3(x)
        return self.head(x)

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

# target encoding: 0 = Blue Win, 1 = Red Win
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

# Feature Selection for Scaling
keywords = ['gold', 'xp', 'cs', 'level', 'kills', 'deaths', 'diff', 'team_']
continuous_cols = [c for c in df_wide.columns if any(x in c.lower() for x in keywords)]

# 7. TRAINING LOOP
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
minutes_to_train = [5, 10, 15, 20, 25]

for target_min in minutes_to_train:
    df_m = df_wide[df_wide['minute'] == target_min].copy()
    if len(df_m) < 50: continue

    print(f"\n--- Training Phase: {target_min} Minutes ---")
    
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(df_m, groups=df_m['match_id']))
    
    train_df = df_m.iloc[train_idx].copy()
    test_df = df_m.iloc[test_idx].copy()

    scaler = StandardScaler()
    train_df[continuous_cols] = scaler.fit_transform(train_df[continuous_cols])
    test_df[continuous_cols] = scaler.transform(test_df[continuous_cols])
    joblib.dump(scaler, f'scaler_{target_min}m.pkl')

    drop_cols = ['winning_team', 'match_id', 'minute']
    X_train_raw = train_df.drop(columns=drop_cols).values.astype('float32')
    y_train_raw = train_df['winning_team'].values.astype('float32').reshape(-1, 1)
    X_test_raw = test_df.drop(columns=drop_cols).values.astype('float32')
    y_test_raw = test_df['winning_team'].values.astype('float32').reshape(-1, 1)

    X_train, y_train = torch.tensor(X_train_raw), torch.tensor(y_train_raw)
    X_test, y_test = torch.tensor(X_test_raw), torch.tensor(y_test_raw)

    # Improved model
    model = ImprovedLoLNet(X_train.shape[1]).to(device)
    
    # Better optimizer with weight decay
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=0.01)
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5, verbose=True
    )
    
    # Standard BCE loss
    criterion = nn.BCELoss()

    train_loader = DataLoader(TensorDataset(X_train, y_train), batch_size=256, shuffle=True)

    best_acc = 0
    patience_counter = 0
    max_patience = 15
    
    for epoch in range(100):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        for b_X, b_y in train_loader:
            b_X, b_y = b_X.to(device), b_y.to(device)
            optimizer.zero_grad()
            preds = model(b_X)
            loss = criterion(preds, b_y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_correct += (preds.round() == b_y).sum().item()
            train_total += len(b_y)
        
        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total
        
        model.eval()
        with torch.no_grad():
            val_preds = model(X_test.to(device)).round()
            val_acc = (val_preds == y_test.to(device)).sum().item() / len(y_test)
            
            # Update scheduler
            scheduler.step(val_acc)
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), f'lol_model_{target_min}m.pth')
                patience_counter = 0
            else:
                patience_counter += 1
        
        if epoch % 5 == 0:
            print(f"Epoch {epoch} | Loss: {avg_train_loss:.4f} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")
        
        # Early stopping
        if patience_counter >= max_patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print(f"Best Val Acc: {best_acc:.2%}")