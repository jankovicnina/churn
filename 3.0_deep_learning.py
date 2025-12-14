import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# pre processing
data = pd.read_parquet('train.parquet', engine='pyarrow')
data = data[data['registration'] <= '2018-11-10']

last_date = pd.to_datetime('2018-11-20')
data['D-X'] = (last_date - data['time']).dt.days
data['days_since_registration'] = (data['time'] - data['registration']).dt.days

# Keep users with >3 days of activity
user_day_counts = data.groupby('userId')['D-X'].nunique()
users_cleaned = user_day_counts[user_day_counts > 3].index.tolist()
data = data[data['userId'].isin(users_cleaned)]

# Identify churners
churners = data.loc[data['page'] == 'Cancellation Confirmation', 'userId'].unique()
data['churned'] = data['userId'].isin(churners).astype(int)

# edit this ciganka
def create_daily_features(df):
    bad_pages = ['Downgrade', 'Thumbs Down', 'Submit Downgrade', 'Roll Advert', 'Error']
    good_pages = ['Thumbs Up', 'Add to Playlist', 'Add Friend', 'Upgrade', 'Submit Upgrade']
    help_pages = ['Help', 'About', 'Settings', 'Save Settings']

    daily_features = (
        df.groupby(['userId', 'days_since_registration'])
          .agg({
              'length': 'sum',
              'sessionId': 'nunique',
              'page': lambda x: list(x),
              'level': 'first'
          })
          .rename(columns={'length': 'total_length', 'sessionId': 'num_sessions'})
          .reset_index()
    )

    def count_pages(pages, page_list):
        return sum([1 for p in pages if p in page_list])

    daily_features['bad_pages_count'] = daily_features['page'].apply(lambda x: count_pages(x, bad_pages))
    daily_features['good_pages_count'] = daily_features['page'].apply(lambda x: count_pages(x, good_pages))
    daily_features['help_pages_count'] = daily_features['page'].apply(lambda x: count_pages(x, help_pages))
    daily_features['num_likes'] = daily_features['page'].apply(lambda x: x.count('Thumbs Up'))
    daily_features['num_events'] = daily_features['page'].apply(len)
    daily_features['status_paid'] = daily_features['level'].apply(lambda x: 1 if x == 'paid' else 0)

    daily_features = daily_features.drop(columns=['page', 'level'])

    # Fill missing days
    all_users = daily_features['userId'].unique()
    full_daily = []

    for user in all_users:
        user_df = daily_features[daily_features['userId'] == user].set_index('days_since_registration')
        idx = range(user_df.index.min(), user_df.index.max() + 1)
        user_df = user_df.reindex(idx, fill_value=0).reset_index()
        user_df['userId'] = user
        # Add mask for actual vs. padded
        user_df['mask'] = (user_df['total_length'] != 0).astype(float)
        full_daily.append(user_df)

    full_daily = pd.concat(full_daily, ignore_index=True)
    full_daily.rename(columns={'index': 'days_since_registration'}, inplace=True)
    return full_daily

daily = create_daily_features(data)

# Merge churn labels
churn_labels = data.groupby('userId')['churned'].max().reset_index()
daily = daily.merge(churn_labels, on='userId', how='left')

# here we create the sequences of 20 days
seq_len = 20
feature_cols = ['total_length', 'num_sessions', 'mask']  # include mask

X_list, y_list = [], []

for user, df_user in daily.groupby('userId'):
    df_user = df_user.sort_values('days_since_registration')
    seq = df_user[feature_cols].values[-seq_len:]
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], len(feature_cols)), dtype='float32')
        pad[:, -1] = 0  # mask=0 for padded days
        seq = np.vstack([pad, seq])
    X_list.append(seq)
    y_list.append(df_user['churned'].max())

X = np.array(X_list, dtype=np.float32)
y = np.array(y_list, dtype=np.float32)

# scaling 
scaler = StandardScaler()
X_reshaped = X[:, :, :-1].reshape(-1, len(feature_cols) - 1)  # scale only real features
X_scaled = scaler.fit_transform(X_reshaped)
X[:, :, :-1] = X_scaled.reshape(X.shape[0], seq_len, len(feature_cols) - 1)

# Convert to tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)

# -split 
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# creating the data
class ChurnDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    def __len__(self):
        return len(self.X)
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_loader = DataLoader(ChurnDataset(X_train, y_train), batch_size=32, shuffle=True)
val_loader = DataLoader(ChurnDataset(X_val, y_val), batch_size=32)

# model
class LSTMChurn(nn.Module):
    def __init__(self, input_dim, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers>1 else 0)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 32)
        self.fc2 = nn.Linear(32,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        out, (h, c) = self.lstm(x)
        h_last = h[-1]
        x = self.dropout(h_last)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # raw logits

model = LSTMChurn(X.shape[2])

# -------------------- LOSS + OPTIMIZER --------------------
pos_weight = torch.tensor([(y_train==0).sum().item() / (y_train==1).sum().item()])
criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)

# -------------------- TRAINING --------------------
for epoch in range(20):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        preds = model(batch_X).squeeze()
        loss = criterion(preds, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            preds = model(batch_X).squeeze()
            val_loss += criterion(preds, batch_y).item()
            predicted = (torch.sigmoid(preds) > 0.5).float()
            correct += (predicted == batch_y).sum().item()
            total += len(batch_y)
    print(f"Epoch {epoch+1} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {correct/total:.4f}")

# prediction
test_data = pd.read_parquet('test.parquet', engine='pyarrow')
test_data['days_since_registration'] = (test_data['time'] - test_data['registration']).dt.days
daily_test_features = create_daily_features(test_data)

test_user_ids = daily_test_features['userId'].unique()
num_test_users = len(test_user_ids)
X_test = np.zeros((num_test_users, seq_len, len(feature_cols)), dtype=np.float32)

for i, user_id in enumerate(test_user_ids):
    df_user = daily_test_features[daily_test_features['userId']==user_id].sort_values('days_since_registration')
    seq = df_user[feature_cols].values[-seq_len:]
    if seq.shape[0] < seq_len:
        pad = np.zeros((seq_len - seq.shape[0], len(feature_cols)), dtype=np.float32)
        pad[:, -1] = 0
        seq = np.vstack([pad, seq])
    X_test[i] = seq

# Scale features
X_test_reshaped = X_test[:, :, :-1].reshape(-1, len(feature_cols)-1)
X_scaled = scaler.transform(X_test_reshaped)
X_test[:, :, :-1] = X_scaled.reshape(X_test.shape[0], seq_len, len(feature_cols)-1)

X_test = torch.tensor(X_test, dtype=torch.float32)

# Predict
model.eval()
predictions = []
with torch.no_grad():
    for i in range(len(X_test)):
        logit = model(X_test[i].unsqueeze(0)).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        predictions.append(1 if prob>0.5 else 0)

submission = pd.DataFrame({'userId': test_user_ids, 'target': predictions})
submission.to_csv('pytorch_fixed.csv', index=False)
print("Submission saved!")