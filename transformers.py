import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, average_precision_score
from sklearn.preprocessing import LabelEncoder
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torch.optim import AdamW
import warnings
warnings.filterwarnings('ignore')

# Проверка доступности GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Используемое устройство: {device}")

# Загрузка данных
df = pd.read_csv('train_m.csv')
text_col = df.columns[1]
label_col = df.columns[2]

print("=== АНАЛИЗ ДАТАСЕТА ===")
print(f"Размер датасета: {len(df)}")
print(f"Распределение классов:\n{df[label_col].value_counts().sort_index()}")

# Удаляем классы с малым количеством примеров
min_samples = 2
class_counts = df[label_col].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
df = df[df[label_col].isin(valid_classes)]

print(f"После фильтрации: {len(df)} примеров")

# Кодирование меток
le = LabelEncoder()
df['label_encoded'] = le.fit_transform(df[label_col])

# Разделение данных
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['label_encoded'])
train_df, val_df = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label_encoded'])

print(f"\nРазделение: Train {len(train_df)}, Val {len(val_df)}, Test {len(test_df)}")

# Модель с исправлением pooler_output
class ToxicClassifier(nn.Module):
    def __init__(self, model_name, num_classes):
        super(ToxicClassifier, self).__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Используем первый токен [CLS] для классификации
        pooled_output = outputs.last_hidden_state[:, 0, :]
        output = self.dropout(pooled_output)
        return self.classifier(output)

# Датасет
class CommentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts.iloc[idx])
        label = self.labels.iloc[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Инициализация
model_name = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
num_classes = len(le.classes_)

model = ToxicClassifier(model_name, num_classes).to(device)

# Датасеты и даталоадеры
train_dataset = CommentDataset(train_df[text_col], train_df['label_encoded'], tokenizer)
val_dataset = CommentDataset(val_df[text_col], val_df['label_encoded'], tokenizer)
test_dataset = CommentDataset(test_df[text_col], test_df['label_encoded'], tokenizer)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=8)
test_loader = DataLoader(test_dataset, batch_size=8)

# Веса классов
class_counts = np.bincount(df['label_encoded'])
class_weights = torch.tensor(
    [1.0 / count if count > 0 else 1.0 for count in class_counts],
    dtype=torch.float32
).to(device)
class_weights = class_weights / class_weights.sum() * num_classes

print(f"\nВеса классов: {class_weights.cpu().numpy()}")

# Оптимизатор и функция потерь
optimizer = AdamW(model.parameters(), lr=2e-5)
criterion = nn.CrossEntropyLoss(weight=class_weights)

# Обучение
def train_epoch(model, dataloader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, preds = torch.max(outputs, dim=1)
        correct_predictions += torch.sum(preds == labels)
    
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy

# Валидация
def eval_model(model, dataloader, criterion):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            probs = torch.softmax(outputs, dim=1)
            
            total_loss += loss.item()
            _, preds = torch.max(outputs, dim=1)
            correct_predictions += torch.sum(preds == labels)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    accuracy = correct_predictions.double() / len(dataloader.dataset)
    avg_loss = total_loss / len(dataloader)
    
    return avg_loss, accuracy, all_preds, all_labels, all_probs

print("\n=== НАЧАЛО ОБУЧЕНИЯ ===")
best_val_loss = float('inf')
patience = 3
patience_counter = 0

for epoch in range(5):
    print(f"\nЭпоха {epoch + 1}/5")
    
    # Обучение
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion)
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    
    # Валидация
    val_loss, val_acc, val_preds, val_labels, val_probs = eval_model(model, val_loader, criterion)
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Ранняя остановка
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), 'best_model.pt')
        print("✅ Сохранена лучшая модель")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print("🛑 Ранняя остановка")
            break

# Загрузка лучшей модели
model.load_state_dict(torch.load('best_model.pt'))

# Тестирование
print("\n=== ТЕСТИРОВАНИЕ ===")
test_loss, test_acc, test_preds, test_labels, test_probs = eval_model(model, test_loader, criterion)
test_probs = np.array(test_probs)

# МЕТРИКИ КАЧЕСТВА
print("\n" + "="*50)
print("МЕТРИКИ КАЧЕСТВА НА ТЕСТЕ")
print("="*50)

from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

accuracy = accuracy_score(test_labels, test_preds)
macro_f1 = f1_score(test_labels, test_preds, average='macro')
weighted_f1 = f1_score(test_labels, test_preds, average='weighted')
precision = precision_score(test_labels, test_preds, average='macro')
recall = recall_score(test_labels, test_preds, average='macro')

print(f"\n🎯 ОСНОВНЫЕ МЕТРИКИ:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Macro F1: {macro_f1:.4f}")
print(f"Weighted F1: {weighted_f1:.4f}")
print(f"Macro Precision: {precision:.4f}")
print(f"Macro Recall: {recall:.4f}")

# Classification Report
print(f"\n📊 CLASSIFICATION REPORT:")
print(classification_report(test_labels, test_preds, target_names=[f'Class {i}' for i in le.classes_]))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(test_labels, test_preds)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=[f'Class {i}' for i in le.classes_],
            yticklabels=[f'Class {i}' for i in le.classes_])
plt.title('Confusion Matrix - Test Set')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.show()

# ROC-AUC и PR-AUC
try:
    if num_classes == 2:
        roc_auc = roc_auc_score(test_labels, test_probs[:, 1])
        pr_auc = average_precision_score(test_labels, test_probs[:, 1])
        print(f"ROC-AUC: {roc_auc:.4f}")
        print(f"PR-AUC: {pr_auc:.4f}")
    else:
        roc_auc = roc_auc_score(test_labels, test_probs, multi_class='ovr', average='macro')
        pr_auc = average_precision_score(test_labels, test_probs, average='macro')
        print(f"Macro ROC-AUC: {roc_auc:.4f}")
        print(f"Macro PR-AUC: {pr_auc:.4f}")
except Exception as e:
    print(f"AUC метрики: недоступно ({e})")

# АНАЛИЗ РЕЗУЛЬТАТОВ
print("\n" + "="*50)
print("АНАЛИЗ РЕЗУЛЬТАТОВ")
print("="*50)

results_df = test_df.copy()
results_df['predicted'] = test_preds
results_df['correct'] = (results_df['label_encoded'] == results_df['predicted'])
results_df['confidence'] = np.max(test_probs, axis=1)
results_df['text_length'] = results_df[text_col].apply(len)

# Анализ по классам
print("\n📈 ТОЧНОСТЬ ПО КЛАССАМ:")
for class_id in le.classes_:
    class_data = results_df[results_df[label_col] == class_id]
    if len(class_data) > 0:
        accuracy_class = class_data['correct'].mean()
        print(f"Класс {class_id}: {accuracy_class:.1%} ({class_data['correct'].sum()}/{len(class_data)})")

# Визуализация результатов
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# 1. Accuracy по классам
class_accuracies = []
for class_id in le.classes_:
    class_data = results_df[results_df[label_col] == class_id]
    accuracy = class_data['correct'].mean() if len(class_data) > 0 else 0
    class_accuracies.append(accuracy)

colors = ['green', 'orange', 'red', 'blue', 'purple']
axes[0, 0].bar(range(len(le.classes_)), class_accuracies, 
                color=colors[:len(le.classes_)], alpha=0.7)
axes[0, 0].set_xlabel('Класс')
axes[0, 0].set_ylabel('Accuracy')
axes[0, 0].set_title('Accuracy по классам')
axes[0, 0].set_xticks(range(len(le.classes_)))
axes[0, 0].set_xticklabels([f'Class {i}' for i in le.classes_])
axes[0, 0].set_ylim(0, 1)

# 2. Уверенность модели
correct_conf = results_df[results_df['correct']]['confidence'].mean()
incorrect_conf = results_df[~results_df['correct']]['confidence'].mean()

axes[0, 1].bar(['Правильные', 'Ошибки'], [correct_conf, incorrect_conf], 
                color=['green', 'red'], alpha=0.7)
axes[0, 1].set_ylabel('Средняя уверенность')
axes[0, 1].set_title('Уверенность модели')
axes[0, 1].set_ylim(0, 1)

# 3. Распределение длины текста
axes[1, 0].hist([results_df[results_df['correct']]['text_length'], 
                 results_df[~results_df['correct']]['text_length']],
                bins=20, alpha=0.7, label=['Правильные', 'Ошибки'], 
                color=['green', 'red'])
axes[1, 0].set_xlabel('Длина текста')
axes[1, 0].set_ylabel('Количество')
axes[1, 0].set_title('Распределение длины текста')
axes[1, 0].legend()

# 4. Матрица ошибок (проценты)
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='Blues', ax=axes[1, 1],
            xticklabels=[f'Class {i}' for i in le.classes_],
            yticklabels=[f'Class {i}' for i in le.classes_])
axes[1, 1].set_title('Confusion Matrix (%)')
axes[1, 1].set_ylabel('True Label')
axes[1, 1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК
print("\n" + "="*50)
print("ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК")
print("="*50)

# Анализ наиболее проблемных пар классов
error_pairs = []
for true_class in range(len(le.classes_)):
    for pred_class in range(len(le.classes_)):
        if true_class != pred_class:
            count = cm[true_class, pred_class]
            if count > 0:
                error_pairs.append((true_class, pred_class, count))

error_pairs.sort(key=lambda x: x[2], reverse=True)

print("\n🧩 НАИБОЛЕЕ ЧАСТЫЕ ОШИБКИ:")
for true_class, pred_class, count in error_pairs[:5]:
    print(f"True: Class {true_class} → Pred: Class {pred_class}: {count} случаев")

# Примеры ошибок
print(f"\n📝 ПРИМЕРЫ ОШИБОЧНЫХ ПРЕДСКАЗАНИЙ:")
errors_df = results_df[~results_df['correct']].head(3)
for idx, row in errors_df.iterrows():
    print(f"\nТекст: {row[text_col][:150]}...")
    print(f"Истинный класс: {row[label_col]}, Предсказанный: {row['predicted']}")
    print(f"Уверенность: {row['confidence']:.3f}")

# ВЫВОДЫ И РЕКОМЕНДАЦИИ
print("\n" + "="*50)
print("ВЫВОДЫ И РЕКОМЕНДАЦИИ")
print("="*50)

# Оценка качества
if macro_f1 >= 0.75:
    rating = "✅ ОТЛИЧНО"
    recommendation = "Модель готова к использованию"
elif macro_f1 >= 0.65:
    rating = "⚠️  ХОРОШО" 
    recommendation = "Модель можно использовать с некоторыми ограничениями"
elif macro_f1 >= 0.55:
    rating = "⚠️  УДОВЛЕТВОРИТЕЛЬНО"
    recommendation = "Требует доработки перед использованием"
else:
    rating = "❌  ПЛОХО"
    recommendation = "Не пригодна для практического использования"

print(f"Общая оценка: {rating}")
print(f"Рекомендация: {recommendation}")
print(f"Ключевая метрика (Macro F1): {macro_f1:.4f}")

# Анализ причин
print(f"\n🔍 КЛЮЧЕВЫЕ НАБЛЮДЕНИЯ:")

if weighted_f1 - macro_f1 > 0.1:
    print("• Значительный дисбаланс классов влияет на качество")
    
if incorrect_conf > 0.7:
    print("• Модель слишком уверена в ошибках - возможен overfitting")
    
if len(results_df[results_df['text_length'] < 10]) > len(results_df) * 0.3:
    print("• Много коротких текстов - сложно для классификации")

# Конкретные рекомендации
print(f"\n💡 КОНКРЕТНЫЕ РЕКОМЕНДАЦИИ:")
if macro_f1 < 0.7:
    print("1. Увеличить объем обучающих данных")
    print("2. Попробовать другие модели (RoBERTa, DeBERTa)")
    print("3. Добавить текстовую предобработку")
    print("4. Использовать аугментацию данных")
    
if len(error_pairs) > 0 and error_pairs[0][2] > len(test_df) * 0.1:
    print(f"5. Обратить внимание на confusion между классами {error_pairs[0][0]} и {error_pairs[0][1]}")
    
print(f"\n🎯 ФИНАЛЬНЫЕ МЕТРИКИ:")
print(f"• Macro F1: {macro_f1:.4f}")
print(f"• Accuracy: {accuracy:.4f}")
print(f"• Weighted F1: {weighted_f1:.4f}")
if 'roc_auc' in locals():
    print(f"• ROC-AUC: {roc_auc:.4f}")
if 'pr_auc' in locals():
    print(f"• PR-AUC: {pr_auc:.4f}")
