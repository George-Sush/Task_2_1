import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight

# ------------------------------------------------------------
# 1. Загрузка данных
# ------------------------------------------------------------

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test_features.csv')

feature_cols = [f't_{i:03d}' for i in range(140)]
X = train_df[feature_cols].values
y = train_df['prediction'].values
test_X = test_df[feature_cols].values
test_ids = test_df['heartbeat_id'].values

# ------------------------------------------------------------
# 2. Стратифицированное разбиение (80/20)
# ------------------------------------------------------------

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# ------------------------------------------------------------
# 3. Нормализация
# ------------------------------------------------------------

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
test_X_scaled = scaler.transform(test_X)

# ------------------------------------------------------------
# 4. Подготовка данных для PyTorch
# ------------------------------------------------------------

def to_tensor(x, y=None):
    # Добавляем размерность канала для 1D CNN: (batch, channels, length) -> (batch, 1, 140)
    x = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    if y is not None:
        y = torch.tensor(y, dtype=torch.long)
        return TensorDataset(x, y)
    return x


train_dataset = to_tensor(X_train_scaled, y_train)
val_dataset = to_tensor(X_val_scaled, y_val)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# ------------------------------------------------------------
# 5. Архитектура 1D CNN
# ------------------------------------------------------------
class ECG1DCNN(nn.Module):
    def __init__(self, num_classes=5):
        super(ECG1DCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.MaxPool1d(2)
        self.dropout = nn.Dropout(0.3)
        # После двух пулингов длина 140 -> 70 -> 35. 35 * 32 = 1120
        self.fc1 = nn.Linear(32 * 35, 64)
        self.fc2 = nn.Linear(64, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = ECG1DCNN(num_classes=5).to(device)

# ------------------------------------------------------------
# 6. Взвешенная функция потерь для работы с дисбалансом
# ------------------------------------------------------------

class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ------------------------------------------------------------
# 7. Обучение с ранней остановкой по Macro F1
# ------------------------------------------------------------

best_val_f1 = 0.0
best_model_state = None
patience = 15
patience_counter = 0
epochs = 100

train_losses, val_f1s = [], []

print("Начало обучения...")
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)

        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # Оценка на валидации
    model.eval()
    val_preds = []
    val_true = []
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x = batch_x.to(device)
            outputs = model(batch_x)
            _, predicted = torch.max(outputs, 1)
            val_preds.extend(predicted.cpu().numpy())
            val_true.extend(batch_y.numpy())

    val_f1 = f1_score(val_true, val_preds, average='macro')
    avg_loss = epoch_loss / len(train_loader)

    train_losses.append(avg_loss)
    val_f1s.append(val_f1)

    print(f"Epoch {epoch + 1}/{epochs} | Loss: {avg_loss:.4f} | Val Macro F1: {val_f1:.4f}")

    # Ранняя остановка
    if val_f1 > best_val_f1:
        best_val_f1 = val_f1
        best_model_state = model.state_dict()
        patience_counter = 0
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Ранняя остановка на эпохе {epoch + 1}")
            break

# Загрузка лучшей модели
model.load_state_dict(best_model_state)

# ------------------------------------------------------------
# 8. Итоговая оценка на валидации
# ------------------------------------------------------------

model.eval()
final_preds = []
with torch.no_grad():
    for batch_x, _ in val_loader:
        batch_x = batch_x.to(device)
        outputs = model(batch_x)
        _, predicted = torch.max(outputs, 1)
        final_preds.extend(predicted.cpu().numpy())

print("\n--- Итоговые метрики на валидации ---")
print(f"Accuracy: {accuracy_score(y_val, final_preds):.4f}")
print(f"Macro F1: {f1_score(y_val, final_preds, average='macro'):.4f}")
print("\nClassification Report:")
print(classification_report(y_val, final_preds))

# ------------------------------------------------------------
# 9. Визуализации
# ------------------------------------------------------------

os.makedirs('outputs', exist_ok=True)

# ------------------------------------------------------------
# 9.1 Матрица ошибок
# ------------------------------------------------------------

plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_val, final_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=range(5), yticklabels=range(5))
plt.title('Confusion Matrix (Validation)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.savefig('outputs/confusion_matrix.png')
plt.close()

# ------------------------------------------------------------
# 9.2 Графики потерь и F1
# ------------------------------------------------------------

fig, ax1 = plt.subplots(figsize=(10, 5))
ax1.plot(train_losses, label='Train Loss', color='red')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss', color='red')
ax2 = ax1.twinx()
ax2.plot(val_f1s, label='Val Macro F1', color='blue')
ax2.set_ylabel('Macro F1', color='blue')
plt.title('Train Loss and Validation Macro F1')
fig.tight_layout()
plt.savefig('outputs/loss_f1_plot.png')
plt.close()

# ------------------------------------------------------------
# 9.3 Представительные сигналы всех 5 классов
# ------------------------------------------------------------

plt.figure(figsize=(15, 10))
for cls in range(5):
    idx = np.where(y_train == cls)[0][0]  # Первый пример класса
    plt.subplot(3, 2, cls + 1)
    plt.plot(X_train_scaled[idx], label=f'Class {cls}')
    plt.title(f'Representative Signal: Class {cls} (n={np.sum(y == cls)})')
    plt.xlabel('Time steps')
    plt.ylabel('Normalized Amplitude')
    plt.legend()
plt.tight_layout()
plt.savefig('outputs/representative_signals.png')
plt.close()

# ------------------------------------------------------------
# 10. Предсказание для тестовой выборки и сохранение
# ------------------------------------------------------------

print("\nГенерация предсказаний для тестовой выборки...")
test_tensor = to_tensor(test_X_scaled).to(device)
model.eval()
with torch.no_grad():
    test_outputs = model(test_tensor)
    _, test_predictions = torch.max(test_outputs, 1)
    test_predictions = test_predictions.cpu().numpy()

submission = pd.DataFrame({
    'heartbeat_id': test_ids,
    'prediction': test_predictions
})

submission.to_csv('outputs/task2_1_predictions.csv', index=False)
print("Прогнозы успешно сохранены в outputs/task2_1_predictions.csv")
print("Распределение предсказанных классов:\n", submission['prediction'].value_counts().sort_index())