import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import nltk
import re
from collections import Counter
import warnings
from tqdm import tqdm
warnings.filterwarnings('ignore')

# Скачиваем стоп-слова если нужно
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

class TextPreprocessor:
    def __init__(self, language='russian'):
        self.language = language
        self.stop_words = self._get_stop_words()
        
    def _get_stop_words(self):
        """Получение стоп-слов из NLTK"""
        try:
            from nltk.corpus import stopwords
            if self.language == 'russian':
                return set(stopwords.words('russian'))
            elif self.language == 'english':
                return set(stopwords.words('english'))
            else:
                return set()
        except:
            print("Предупреждение: не удалось загрузить стоп-слова из NLTK, используются базовые")
            if self.language == 'russian':
                return {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она'}
            elif self.language == 'english':
                return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            else:
                return set()
    
    def clean_text(self, text):
        """Очистка и предобработка текста"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        text = text.lower()
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?]', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_dataframe(self, df, text_column, label_column):
        """Предобработка всего датафрейма"""
        df_clean = df.copy()
        print("Начата предобработка текстов для нейросетей...")
        
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        final_len = len(df_clean)
        
        print(f"Предобработка завершена. Удалено {initial_len - final_len} пустых текстов.")
        return df_clean

class TextTokenizer:
    """Токенизатор для нейросетевых моделей"""
    def __init__(self, max_features=20000, max_len=200):
        self.max_features = max_features
        self.max_len = max_len
        self.word2idx = {}
        self.idx2word = {}
        self.vocab_size = 0
        
    def build_vocab(self, texts):
        """Построение словаря на основе текстов"""
        print("Построение словаря...")
        counter = Counter()
        
        for text in texts:
            words = str(text).split()
            counter.update(words)
        
        # Берем самые частые слова
        most_common = counter.most_common(self.max_features - 2)
        
        # Создаем словарь
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        for idx, (word, count) in enumerate(most_common):
            self.word2idx[word] = idx + 2
            
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"Словарь построен. Размер: {self.vocab_size} слов")
        return self.word2idx, self.idx2word
    
    def texts_to_sequences(self, texts):
        """Преобразование текстов в последовательности индексов"""
        sequences = []
        for text in texts:
            words = str(text).split()[:self.max_len]
            sequence = [self.word2idx.get(word, 1) for word in words]  # 1 = <UNK>
            
            # Паддинг
            if len(sequence) < self.max_len:
                sequence += [0] * (self.max_len - len(sequence))  # 0 = <PAD>
            else:
                sequence = sequence[:self.max_len]
                
            sequences.append(sequence)
        
        return np.array(sequences)

class TextDataset(Dataset):
    """Датасет для нейросетевых моделей"""
    def __init__(self, texts, labels, tokenizer):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts.iloc[idx] if hasattr(self.texts, 'iloc') else self.texts[idx]
        label = self.labels.iloc[idx] if hasattr(self.labels, 'iloc') else self.labels[idx]
        
        sequence = self.tokenizer.texts_to_sequences([text])[0]
        
        return {
            'input_ids': torch.tensor(sequence, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# =============================================================================
# МОДЕЛЬ 1: HYBRID CNN + BiLSTM + ATTENTION
# =============================================================================

class HybridCNNBiLSTM(nn.Module):
    """
    Гибридная модель: CNN для локальных паттернов + BiLSTM для контекста + Attention для важных частей
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, n_classes=3, 
                 dropout_rate=0.5, num_filters=100, kernel_sizes=[3, 4, 5]):
        super(HybridCNNBiLSTM, self).__init__()
        
        # Embedding слой
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        
        # CNN часть для захвата n-грамм
        self.convs = nn.ModuleList([
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k, padding=k//2)
            for k in kernel_sizes
        ])
        self.cnn_dropout = nn.Dropout(dropout_rate)
        self.cnn_pool = nn.AdaptiveMaxPool1d(1)
        
        # BiLSTM часть для контекстной информации
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, 
                           bidirectional=True, dropout=dropout_rate)
        self.lstm_dropout = nn.Dropout(dropout_rate)
        
        # Attention механизм
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim * 2,  # bidirectional
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Классификатор
        cnn_output_size = num_filters * len(kernel_sizes)
        lstm_output_size = hidden_dim * 2  # bidirectional
        
        self.classifier = nn.Sequential(
            nn.Linear(cnn_output_size + lstm_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(dropout_rate),
            
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(dropout_rate),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            
            nn.Linear(128, n_classes)
        )
        
        # Инициализация весов
        self._init_weights()
    
    def _init_weights(self):
        """Инициализация весов"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LSTM):
                for name, param in module.named_parameters():
                    if 'weight' in name:
                        nn.init.orthogonal_(param)
                    elif 'bias' in name:
                        nn.init.zeros_(param)
    
    def forward(self, input_ids):
        # Embedding
        x_embed = self.embedding(input_ids)  # [batch_size, seq_len, embedding_dim]
        
        # CNN branch
        x_cnn = x_embed.transpose(1, 2)  # [batch_size, embedding_dim, seq_len]
        cnn_outputs = []
        for conv in self.convs:
            conv_out = torch.relu(conv(x_cnn))
            pooled = self.cnn_pool(conv_out).squeeze(-1)  # [batch_size, num_filters]
            cnn_outputs.append(pooled)
        
        x_cnn_combined = torch.cat(cnn_outputs, dim=1)  # [batch_size, num_filters * len(kernel_sizes)]
        x_cnn_combined = self.cnn_dropout(x_cnn_combined)
        
        # BiLSTM branch
        lstm_out, (hidden, cell) = self.lstm(x_embed)  # [batch_size, seq_len, hidden_dim*2]
        
        # Attention
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )  # [batch_size, seq_len, hidden_dim*2]
        
        # Берем взвешенную сумму
        x_lstm = attn_out.mean(dim=1)  # [batch_size, hidden_dim*2]
        x_lstm = self.lstm_dropout(x_lstm)
        
        # Объединяем CNN и LSTM выходы
        x_combined = torch.cat([x_cnn_combined, x_lstm], dim=1)
        
        # Классификация
        return self.classifier(x_combined)

# =============================================================================
# МОДЕЛЬ 2: DEEP BiLSTM WITH ATTENTION
# =============================================================================

class DeepBiLSTMAttention(nn.Module):
    """
    Глубокая BiLSTM сеть с механизмом внимания
    Простая но эффективная архитектура для текстовой классификации
    """
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, 
                 n_layers=3, n_classes=3, dropout_rate=0.5):
        super(DeepBiLSTMAttention, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.embedding_dropout = nn.Dropout(dropout_rate)
        
        # Многослойный BiLSTM
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, n_layers, 
            batch_first=True, bidirectional=True, dropout=dropout_rate
        )
        
        # Attention mechanism
        self.attention = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, n_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, input_ids):
        # Embedding
        x = self.embedding(input_ids)
        x = self.embedding_dropout(x)
        
        # LSTM
        lstm_out, (hidden, cell) = self.lstm(x)
        
        # Attention
        attention_weights = torch.softmax(self.attention(lstm_out), dim=1)
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)
        
        # Classification
        return self.classifier(context_vector)

class NeuralTextClassifier:
    def __init__(self, language='russian', device=None):
        self.language = language
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = {}
        self.metrics = {}
        self.preprocessor = TextPreprocessor(language)
        self.tokenizer = None
        self.error_analysis = {}
        self.X_train = None
        self.y_train = None
        
        print(f"Используется устройство: {self.device}")
        if self.device == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
    
    def load_and_prepare_data(self, file_path, text_column='cleaned_comment', label_column='label'):
        """Загрузка и подготовка данных"""
        print("Загрузка данных для нейросетей...")
        df = pd.read_csv(file_path)
        
        # Предобработка
        df_processed = self.preprocessor.preprocess_dataframe(df, text_column, label_column)
        
        X = df_processed['cleaned_text']
        y = df_processed[label_column]
        
        # Сохраняем для доступа позже
        self.X_train_full = X
        self.y_train_full = y
        
        # Стратифицированное разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Инициализация и построение словаря
        self.tokenizer = TextTokenizer(max_features=30000, max_len=200)
        self.tokenizer.build_vocab(X_train)
        
        # Создание датасетов
        train_dataset = TextDataset(X_train, y_train, self.tokenizer)
        test_dataset = TextDataset(X_test, y_test, self.tokenizer)
        
        # DataLoader'ы
        train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=2)
        test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)
        
        print(f"\nДанные подготовлены:")
        print(f"Тренировочная выборка: {len(X_train)} samples")
        print(f"Тестовая выборка: {len(X_test)} samples")
        print(f"Размер словаря: {self.tokenizer.vocab_size}")
        print(f"Размер батча: 64")
        
        return train_loader, test_loader, X_test, y_test
    
    def calculate_class_weights(self, y_train):
        """Расчет весов классов для функции потерь"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = torch.tensor(weights, dtype=torch.float32).to(self.device)
        
        print(f"Веса классов: {dict(zip(classes, weights))}")
        return class_weights
    
    def train_model(self, model, model_name, train_loader, val_loader, class_weights, epochs=10):
        """Обучение нейросетевой модели"""
        print(f"\n🎯 Начало обучения {model_name}...")
        
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        train_losses = []
        val_accuracies = []
        best_accuracy = 0
        patience = 3
        patience_counter = 0
        
        for epoch in range(epochs):
            # Training
            model.train()
            total_loss = 0
            train_correct = 0
            train_total = 0
            
            for batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                outputs = model(input_ids)
                loss = criterion(outputs, labels)
                loss.backward()
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                total_loss += loss.item()
                
                # Calculate training accuracy
                _, predicted = torch.max(outputs.data, 1)
                train_total += labels.size(0)
                train_correct += (predicted == labels).sum().item()
            
            scheduler.step()
            
            # Validation
            val_accuracy, val_predictions, val_probabilities, val_labels = self.evaluate_model(model, val_loader)
            train_accuracy = train_correct / train_total
            avg_loss = total_loss / len(train_loader)
            
            train_losses.append(avg_loss)
            val_accuracies.append(val_accuracy)
            
            print(f'Epoch {epoch+1}:')
            print(f'  Loss: {avg_loss:.4f}, Train Acc: {train_accuracy:.4f}, Val Acc: {val_accuracy:.4f}')
            print(f'  Learning Rate: {scheduler.get_last_lr()[0]:.2e}')
            
            # Early stopping
            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                patience_counter = 0
                torch.save(model.state_dict(), f'best_{model_name.replace(" ", "_").lower()}.pth')
                print(f"  🎯 Новый лучший результат! Сохранена модель.")
            else:
                patience_counter += 1
                print(f"  ⏳ Early stopping: {patience_counter}/{patience}")
                
            if patience_counter >= patience:
                print(f"  🛑 Early stopping на эпохе {epoch+1}")
                break
        
        print(f"\n✅ Обучение {model_name} завершено! Лучшая точность: {best_accuracy:.4f}")
        
        # Load best model
        model.load_state_dict(torch.load(f'best_{model_name.replace(" ", "_").lower()}.pth'))
        
        return train_losses, val_accuracies
    
    def evaluate_model(self, model, data_loader):
        """Оценка модели"""
        model.eval()
        correct = 0
        total = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                input_ids = batch['input_ids'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = model(input_ids)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs.data, 1)
                
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        accuracy = correct / total
        return accuracy, all_predictions, all_probabilities, all_labels
    
    def train_all_models(self, train_loader, val_loader, y_train):
        """Обучение всех нейросетевых моделей"""
        class_weights = self.calculate_class_weights(y_train)
        
        # Модель 1: Hybrid CNN + BiLSTM + Attention
        model1 = HybridCNNBiLSTM(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            n_classes=3,
            dropout_rate=0.5
        ).to(self.device)
        
        print(f"\n🔄 МОДЕЛЬ 1: Hybrid CNN-BiLSTM-Attention")
        print(f"Параметры: {sum(p.numel() for p in model1.parameters()):,}")
        
        train_losses1, val_accuracies1 = self.train_model(
            model1, "Hybrid CNN-BiLSTM", train_loader, val_loader, class_weights, epochs=15
        )
        
        # Модель 2: Deep BiLSTM with Attention
        model2 = DeepBiLSTMAttention(
            vocab_size=self.tokenizer.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            n_layers=3,
            n_classes=3,
            dropout_rate=0.5
        ).to(self.device)
        
        print(f"\n🔄 МОДЕЛЬ 2: Deep BiLSTM with Attention")
        print(f"Параметры: {sum(p.numel() for p in model2.parameters()):,}")
        
        train_losses2, val_accuracies2 = self.train_model(
            model2, "Deep BiLSTM", train_loader, val_loader, class_weights, epochs=15
        )
        
        self.models = {
            'Hybrid CNN-BiLSTM': model1,
            'Deep BiLSTM': model2
        }
        
        return {
            'Hybrid CNN-BiLSTM': (train_losses1, val_accuracies1),
            'Deep BiLSTM': (train_losses2, val_accuracies2)
        }
    
    def comprehensive_evaluation(self, test_loader, X_test, y_test):
        """Комплексная оценка всех моделей"""
        for model_name, model in self.models.items():
            print(f"\n{'='*60}")
            print(f"📊 ОЦЕНКА {model_name.upper()}")
            print(f"{'='*60}")
            
            accuracy, predictions, probabilities, true_labels = self.evaluate_model(model, test_loader)
            
            y_true = np.array(true_labels)
            y_pred = np.array(predictions)
            y_proba = np.array(probabilities)
            
            # Расчет метрик
            metrics = {
                'accuracy': accuracy_score(y_true, y_pred),
                'macro_f1': f1_score(y_true, y_pred, average='macro'),
                'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
                'confusion_matrix': confusion_matrix(y_true, y_pred),
                'roc_auc_ovr': roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro'),
                'roc_auc_ovo': roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro'),
                'pr_auc': average_precision_score(y_true, y_proba, average='macro'),
                'classification_report': classification_report(y_true, y_pred, output_dict=True)
            }
            
            self.metrics[model_name] = metrics
            
            # Вывод результатов
            print(f"Accuracy: {metrics['accuracy']:.4f}")
            print(f"Macro F1: {metrics['macro_f1']:.4f}")
            print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
            print(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
            print(f"PR-AUC: {metrics['pr_auc']:.4f}")
            
            print(f"\nДетальный отчет по классам:")
            print(classification_report(y_true, y_pred, target_names=['Нейтральный', 'Оскорбительный', 'Токсичный']))
            
            # Вывод confusion matrix
            self._print_confusion_matrix(model_name, metrics['confusion_matrix'])
            
            # Анализ ошибок
            self._analyze_errors(model_name, y_true, y_pred, X_test)
    
    def _print_confusion_matrix(self, model_name, cm):
        """Красивый вывод confusion matrix"""
        print(f"\n📋 CONFUSION MATRIX - {model_name.upper()}:")
        print(" " * 15 + "Predicted →")
        print(" " * 12 + "Нейтр  Оскорб  Токсич")
        
        class_names = ['Нейтральный', 'Оскорбительный', 'Токсичный']
        for i, true_class in enumerate(class_names):
            row_label = f"True {true_class[:6]}"
            print(f"{row_label:12}", end="")
            for j in range(3):
                print(f"{cm[i, j]:6}  ", end="")
            print()
        
        # Анализ матрицы
        total = cm.sum()
        accuracy = np.trace(cm) / total
        print(f"\nОбщая точность: {accuracy:.3f}")
        
        # Анализ по классам
        for i, class_name in enumerate(class_names):
            precision = cm[i,i] / cm[:,i].sum() if cm[:,i].sum() > 0 else 0
            recall = cm[i,i] / cm[i,:].sum() if cm[i,:].sum() > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}, F1={f1:.3f}")
    
    def _analyze_errors(self, model_name, y_true, y_pred, X_test):
        """Анализ ошибок классификации"""
        errors = {
            'false_positives': [],
            'false_negatives': [], 
            'confusions': []
        }
        
        for i, (true, pred, text) in enumerate(zip(y_true, y_pred, X_test)):
            text_length = len(str(text).split())
            
            if true == 0 and pred != 0:
                errors['false_positives'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
            elif true != 0 and pred == 0:
                errors['false_negatives'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
            elif true != pred and true != 0 and pred != 0:
                errors['confusions'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
        
        self.error_analysis[model_name] = errors
        
        print(f"\n🔍 АНАЛИЗ ОШИБОК ДЛЯ {model_name}:")
        print(f"False Positives (нейтральные → опасные): {len(errors['false_positives'])}")
        print(f"False Negatives (опасные → нейтральные): {len(errors['false_negatives'])}")
        print(f"Confusions (ошибки между классами): {len(errors['confusions'])}")
        
        # Примеры ошибок
        if errors['false_positives']:
            print(f"\n📝 Пример False Positive:")
            sample = errors['false_positives'][0]
            print(f"  Текст: {sample['text'][:100]}...")
            print(f"  Истинный: {sample['true_class']}, Предсказанный: {sample['pred_class']}")
    
    def plot_training_history(self, training_history):
        """Визуализация процесса обучения"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        colors = ['blue', 'red', 'green']
        model_names = list(training_history.keys())
        
        for idx, model_name in enumerate(model_names):
            train_losses, val_accuracies = training_history[model_name]
            
            # Loss
            axes[0, 0].plot(train_losses, label=model_name, color=colors[idx], linewidth=2)
            axes[0, 0].set_title('Функция потерь на тренировке')
            axes[0, 0].set_xlabel('Эпоха')
            axes[0, 0].set_ylabel('Loss')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # Accuracy
            axes[0, 1].plot(val_accuracies, label=model_name, color=colors[idx], linewidth=2)
            axes[0, 1].set_title('Точность на валидации')
            axes[0, 1].set_xlabel('Эпоха')
            axes[0, 1].set_ylabel('Accuracy')
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # Сравнение финальных метрик
        metrics_comparison = []
        for model_name, metrics in self.metrics.items():
            metrics_comparison.append({
                'Model': model_name,
                'Accuracy': metrics['accuracy'],
                'Macro F1': metrics['macro_f1'],
                'ROC-AUC': metrics['roc_auc_ovr']
            })
        
        df_metrics = pd.DataFrame(metrics_comparison)
        
        # Bar plot для сравнения метрик
        x = np.arange(len(df_metrics))
        width = 0.25
        
        metrics_to_plot = ['Accuracy', 'Macro F1', 'ROC-AUC']
        for i, metric in enumerate(metrics_to_plot):
            axes[1, 0].bar(x + i*width, df_metrics[metric], width, label=metric, alpha=0.8)
        
        axes[1, 0].set_title('Сравнение метрик моделей')
        axes[1, 0].set_xlabel('Модели')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].set_xticks(x + width)
        axes[1, 0].set_xticklabels(df_metrics['Model'], rotation=45)
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Confusion matrices
        for idx, (model_name, metrics) in enumerate(self.metrics.items()):
            cm = metrics['confusion_matrix']
            row = 1
            col = 1
            im = axes[row, col].imshow(cm, cmap='Blues', aspect='auto')
            axes[row, col].set_title(f'Confusion Matrix - {model_name}')
            axes[row, col].set_xticks([0, 1, 2])
            axes[row, col].set_yticks([0, 1, 2])
            axes[row, col].set_xticklabels(['Нейтр', 'Оскорб', 'Токсич'])
            axes[row, col].set_yticklabels(['Нейтр', 'Оскорб', 'Токсич'])
            
            # Добавляем аннотации
            for i in range(3):
                for j in range(3):
                    axes[row, col].text(j, i, f'{cm[i, j]}', 
                                      ha='center', va='center', 
                                      color='white' if cm[i, j] > cm.max()/2 else 'black')
        
        plt.tight_layout()
        plt.show()
    
    def plot_metrics_comparison(self):
        """Сравнение метрик всех моделей"""
        metrics_df = pd.DataFrame({
            model: {
                'Accuracy': metrics['accuracy'],
                'Macro F1': metrics['macro_f1'],
                'Weighted F1': metrics['weighted_f1'],
                'ROC-AUC': metrics['roc_auc_ovr'],
                'PR-AUC': metrics['pr_auc']
            }
            for model, metrics in self.metrics.items()
        }).T
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        axes = axes.ravel()
        
        metrics_to_plot = ['Accuracy', 'Macro F1', 'Weighted F1', 'ROC-AUC', 'PR-AUC']
        
        for idx, metric in enumerate(metrics_to_plot):
            ax = axes[idx]
            colors = ['lightblue', 'lightcoral']
            bars = ax.bar(range(len(metrics_df)), metrics_df[metric], color=colors, alpha=0.8)
            ax.set_title(f'Сравнение {metric}', fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.set_xticks(range(len(metrics_df)))
            ax.set_xticklabels(metrics_df.index, rotation=45)
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
        
        axes[5].set_visible(False)
        plt.tight_layout()
        plt.show()
        
        return metrics_df

def run_neural_networks(file_path, text_column='cleaned_comment', label_column='label', language='russian'):
    """
    Полный запуск нейросетевых моделей для классификации текстов
    """
    print("🧠 ЗАПУСК НЕЙРОСЕТЕВЫХ МОДЕЛЕЙ (PyTorch)")
    print("="*70)
    print(f"Язык: {language}")
    print(f"Файл данных: {file_path}")
    print("="*70)
    
    # Инициализация классификатора
    classifier = NeuralTextClassifier(language=language)
    
    # Загрузка и подготовка данных
    train_loader, test_loader, X_test, y_test = classifier.load_and_prepare_data(
        file_path, text_column, label_column
    )
    
    # Разделение на train/validation
    dataset_size = len(train_loader.dataset)
    train_size = int(0.9 * dataset_size)
    val_size = dataset_size - train_size
    
    train_subset, val_subset = torch.utils.data.random_split(
        train_loader.dataset, [train_size, val_size]
    )
    
    train_loader_sub = DataLoader(train_subset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=64, shuffle=False)
    
    # Получаем y_train для расчета весов классов
    y_train = [classifier.y_train_full.iloc[i] for i in train_subset.indices]
    
    # Обучение моделей
    training_history = classifier.train_all_models(train_loader_sub, val_loader, y_train)
    
    # Комплексная оценка
    classifier.comprehensive_evaluation(test_loader, X_test, y_test)
    
    # Визуализация результатов
    classifier.plot_training_history(training_history)
    metrics_df = classifier.plot_metrics_comparison()
    
    # Вывод итоговой таблицы
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ - НЕЙРОСЕТИ")
    print("="*80)
    print(metrics_df.round(4))
    
    # Анализ лучшей модели
    best_model = metrics_df['Macro F1'].idxmax()
    best_score = metrics_df.loc[best_model, 'Macro F1']
    print(f"\n🏆 ЛУЧШАЯ НЕЙРОСЕТЕВАЯ МОДЕЛЬ: {best_model} (Macro F1: {best_score:.4f})")
    
    return classifier

if __name__ == "__main__":
    # Пример использования
    try:
        results = run_neural_networks(
            file_path='train_m.csv',
            text_column='comment_text', 
            label_column='label',
            language='russian'
        )
        
