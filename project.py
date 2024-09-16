
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from sklearn.metrics import f1_score, accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

X = pd.read_csv('data/X_train.csv')
y = pd.read_csv('data/y_train.csv')

X.columns
y['surface'].value_counts().plot(kind='barh', figsize=(8,4))
plt.tight_layout()
plt.savefig('target_distribution.png')

X['series_id'].value_counts()

X.columns




rows = X.groupby('series_id').get_group(12)
rows.loc[:,'orientation_X':].plot(subplots=True, layout=(3,4), figsize=(15, 8))
plt.tight_layout()
plt.savefig('example_carpet.png')

y[y['surface'] == 'carpet'].value_counts()
merged = X.merge(y)

plt.figure(figsize=(20,15))
for i, col in enumerate(X.loc[:,'orientation_X':].columns):
    ax = plt.subplot(3,4,i+1)
    ax = plt.title(col)
    for name, rows in merged.groupby('surface'):
        sns.kdeplot(rows[col], label=name)
    plt.legend()
plt.tight_layout()
plt.savefig('s.png')

acc_z = merged[['surface', 'linear_acceleration_Z']].melt(id_vars='surface')
plt.figure()
sns.boxenplot(x='surface', y='value', data=acc_z)
plt.xticks(rotation=90)
plt.tight_layout()
plt.savefig('lin_z.png')

X_agg = X.drop(columns='row_id').groupby('series_id').agg(['mean', 'std', 'median', 'min', 'max']).reset_index()
X_agg.columns = [f'{a}_{b}' for a,b in X_agg.columns]

X_data = X_agg.drop(columns='series_id_')


X_train, X_test, y_train, y_test = train_test_split(X_data,y['surface'], test_size=0.2, random_state=123, stratify=y['surface'] )

lr = Pipeline([('scaler', StandardScaler()), ('model', LogisticRegression(max_iter=100))])
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(classification_report(y_test, y_pred))

final = []
final.append(['LogisticRegression', accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')])


cm = confusion_matrix(y_test, y_pred)
conf = ConfusionMatrixDisplay(cm, display_labels=lr.classes_)
fig, ax = plt.subplots(figsize=(8,7))
conf.plot(ax=ax)
plt.xticks(rotation=90)
plt.title('LogisticRegression')
plt.tight_layout()
plt.savefig('lr.png')

rf = RandomForestClassifier(max_depth=8)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))


cm = confusion_matrix(y_test, y_pred)
conf = ConfusionMatrixDisplay(cm, display_labels=rf.classes_)
fig, ax = plt.subplots(figsize=(8,7))
conf.plot(ax=ax)
plt.xticks(rotation=90)
plt.title('RandomForestClassifier')
plt.tight_layout()
plt.savefig('rf.png')

final.append(['RandomForestClassifier', accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')])

# CNN
cols = X.loc[:,'orientation_X':].columns
X_2d = X.drop(columns='row_id').groupby('series_id').apply( lambda x: x[cols].values).tolist()
X_2d = np.stack(X_2d, axis=0)


X_train, X_test, y_train, y_test = train_test_split(X_2d,y['surface'].astype('category').cat.codes.values, test_size=0.2, random_state=123, stratify=y['surface'] )

X_train.shape

n_tr = X_train.shape[0]
n_te = X_test.shape[0]


sc = StandardScaler()
X_train = sc.fit_transform(X_train.reshape(n_tr, -1)).reshape(X_train.shape)
X_test = sc.transform(X_test.reshape(n_te, -1)).reshape(X_test.shape)

nclasses = len(set(y_train))

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=X_train.shape[1:]),
    tf.keras.layers.Conv1D(24, 10, activation='relu'),
    tf.keras.layers.AvgPool1D(5),
    tf.keras.layers.Conv1D(24, 5, activation='relu'),
    tf.keras.layers.GlobalAvgPool1D(),
    tf.keras.layers.Dense(nclasses)
])

model.summary()

#tf.keras.utils.plot_model(model)

loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
model.compile('adam', loss, metrics=['accuracy'])

model.fit(X_train, y_train, epochs=100, validation_data=(X_test, y_test))

y_pred = model.predict(X_test)
y_pred = np.argmax(y_pred, 1)

y['surface'].astype('category').cat.codes.values
labels = y['surface'].astype('category').cat.categories

cm = confusion_matrix(y_test, y_pred)
conf = ConfusionMatrixDisplay(cm, display_labels=labels)
fig, ax = plt.subplots(figsize=(8,7))
conf.plot(ax=ax)
plt.xticks(rotation=90)
plt.title('Conv1D')
plt.tight_layout()
plt.savefig('Conv1D.png')

final.append(['NeuralNetworks', accuracy_score(y_test, y_pred), f1_score(y_test, y_pred, average='macro')])

print(pd.DataFrame(final, columns=['Model', 'Accuracy', 'F1-Score']))

##CNN with attention

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
from sklearn.metrics import ConfusionMatrixDisplay
import torch
import torch.nn as nn
import torch.optim as optim

# Load your data
df_X = pd.read_csv('data/X_train.csv')
df_y = pd.read_csv('data/y_train.csv')

# Extract features and target variable
X = df_X.drop(['row_id', 'series_id', 'measurement_number'], axis=1)
y = df_y['surface']

# Reshape features for 1D CNN
n_series = df_X['series_id'].nunique()
n_timesteps = df_X['measurement_number'].nunique()
n_features = X.shape[1]

X_reshaped = X.values.reshape((n_series, n_timesteps, n_features))

# Convert target variable to numerical labels
y_labels, class_labels = pd.factorize(y)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y_labels, test_size=0.2, random_state=123, stratify=y_labels)

# Standardize the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(X_train.shape[0], -1)).reshape(X_train.shape)
X_test_scaled = scaler.transform(X_test.reshape(X_test.shape[0], -1)).reshape(X_test.shape)

# Build 1D CNN model with Attention using PyTorch
nclasses = len(class_labels)

class CNNWithAttention(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super(CNNWithAttention, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=n_features, out_channels=64, kernel_size=10, stride=1)
        self.batch_norm1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.pool1 = nn.AvgPool1d(kernel_size=5)
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=5, stride=1)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.attention = nn.MultiheadAttention(embed_dim=128, num_heads=2)
        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, nclasses)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = torch.relu(self.conv1(x))
        x = self.batch_norm1(x)
        x = self.dropout1(x)
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.batch_norm2(x)
        x = self.dropout2(x)
        x = x.permute(2, 0, 1)
        x, _ = self.attention(x, x, x)
        x = x.permute(1, 2, 0)
        x = self.global_avg_pool(x)
        x = x.view(-1, 128)
        x = torch.softmax(self.fc(x), dim=1)
        return x

# Initialize the model with a lower dropout rate
model_with_attention = CNNWithAttention(dropout_rate=0.3)

# Use GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_with_attention.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_with_attention.parameters(), lr=0.001)

# Convert data to PyTorch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
y_train_tensor = torch.tensor(y_train, dtype=torch.long).to(device)

X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)
y_test_tensor = torch.tensor(y_test, dtype=torch.long).to(device)

# Training loop
epochs = 50
for epoch in range(epochs):
    model_with_attention.train()
    optimizer.zero_grad()
    outputs = model_with_attention(X_train_tensor)
    loss = criterion(outputs, y_train_tensor)
    loss.backward()
    optimizer.step()

    # Print training loss for monitoring
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item()}")

# Testing
model_with_attention.eval()
with torch.no_grad():
    outputs = model_with_attention(X_test_tensor)
    _, predicted = torch.max(outputs, 1)
    y_pred_labels = predicted.cpu().numpy()

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred_labels)
f1_macro = f1_score(y_test, y_pred_labels, average='macro')

print(f'Accuracy: {accuracy}')
print(f'F1 Score (Macro): {f1_macro}')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred_labels)
conf = ConfusionMatrixDisplay(cm, display_labels=class_labels)
conf.plot()
plt.show()