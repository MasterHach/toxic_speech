import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Скачиваем необходимые ресурсы
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Загрузка данных
df = pd.read_csv('train.csv')

# Создаем 3 класса на основе оригинальных меток
def define_three_classes(row):
    if row['severe_toxic'] == 1 or row['threat'] == 1:
        return 2  # Токсичный (самый серьезный)
    elif row['toxic'] == 1 or row['obscene'] == 1 or row['insult'] == 1 or row['identity_hate'] == 1:
        return 1  # Оскорбительный
    else:
        return 0  # Нейтральный

df['label'] = df.apply(define_three_classes, axis=1)

# Проверяем распределение классов
print("Распределение классов:")
print(df['label'].value_counts())
print(f"Всего комментариев: {len(df)}")

# -----------------------------------------

def preprocess_text(text):
    """
    Функция для предобработки текста для 3-х классовной классификации
    """
    # 1. Приведение к нижнему регистру
    text = text.lower()
    
    # 2. Удаление символов переноса строки
    text = re.sub(r'\n', ' ', text)
    
    # 3. Замена HTML-сущностей
    text = re.sub(r'&amp;', '&', text)
    text = re.sub(r'&lt;', '<', text)
    text = re.sub(r'&gt;', '>', text)
    
    # 4. Удаление URL-ссылок
    text = re.sub(r'http\S+', '', text)
    
    # 5. Удаление всего, кроме букв и пробелов
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # 6. Удаление лишних пробелов
    text = re.sub(r'\s+', ' ', text).strip()
    
    # 7. Токенизация и лемматизация
    tokens = text.split()
    if len(tokens) == 0:
        return ""
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    # Удаляем стоп-слова и проводим лемматизацию
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    
    return ' '.join(tokens)

# Применяем предобработку
df['cleaned_comment'] = df['comment_text'].apply(preprocess_text)

# Удаляем пустые комментарии после очистки
df = df[df['cleaned_comment'].str.len() > 0]

# -----------------------------------------------

from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv1D, GlobalMaxPooling1D, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

# Параметры
MAX_FEATURES = 20000
MAX_LEN = 200
EMBEDDING_DIM = 100

# Разделяем данные
X = df['cleaned_comment']
y = df['label']

# Стратифицированное разделение для сохранения распределения классов
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Токенизация
tokenizer = Tokenizer(num_words=MAX_FEATURES, oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

# Паддинг
X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

# Преобразуем метки в one-hot encoding для 3 классов
y_train_categorical = to_categorical(y_train, num_classes=3)
y_test_categorical = to_categorical(y_test, num_classes=3)

# Рассчитываем веса классов для борьбы с дисбалансом
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = dict(enumerate(class_weights))
print("Веса классов:", class_weight_dict)

# Создание ансамблевой модели для 3 классов
def create_ensemble_3class_model(vocab_size, embedding_dim, max_len):
    # Входной слой
    input_layer = Input(shape=(max_len,))
    
    # Слой эмбеддинга
    embedding = Embedding(vocab_size, embedding_dim)(input_layer)
    
    # --- Ветка CNN ---
    conv1 = Conv1D(128, 5, activation='relu')(embedding)
    pool1 = GlobalMaxPooling1D()(conv1)
    cnn_branch = Dense(64, activation='relu')(pool1)
    cnn_branch = Dropout(0.5)(cnn_branch)
    
    # --- Ветка LSTM ---
    lstm_branch = LSTM(64, return_sequences=False, dropout=0.2, recurrent_dropout=0.2)(embedding)
    lstm_branch = Dense(64, activation='relu')(lstm_branch)
    lstm_branch = Dropout(0.5)(lstm_branch)
    
    # Объединяем ветки
    concatenated = Concatenate()([cnn_branch, lstm_branch])
    
    # Добавляем дополнительный полносвязный слой
    hidden = Dense(64, activation='relu')(concatenated)
    hidden = Dropout(0.3)(hidden)
    
    # Выходной слой для 3 классов
    output_layer = Dense(3, activation='softmax')(hidden)
    
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(
        optimizer=Adam(learning_rate=0.001), 
        loss='categorical_crossentropy', 
        metrics=['accuracy']
    )
    return model

# Создаем и обучаем модель
vocab_size = min(MAX_FEATURES, len(tokenizer.word_index)) + 1
model = create_ensemble_3class_model(vocab_size, EMBEDDING_DIM, MAX_LEN)

print(model.summary())

# Обучение с ранней остановкой
early_stop = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train_categorical,
    epochs=15,
    batch_size=32,
    validation_split=0.1,
    class_weight=class_weight_dict,
    callbacks=[early_stop],
    verbose=1
)

# -------------------------------------------------------

# Предсказания на тестовой выборке
y_pred_proba = model.predict(X_test_pad)
y_pred = np.argmax(y_pred_proba, axis=1)

# Отчет о классификации
print("Детальный отчет по классификации:")
print(classification_report(y_test, y_pred, 
                          target_names=['Нейтральный', 'Оскорбительный', 'Токсичный']))

# Матрица ошибок
conf_matrix = confusion_matrix(y_test, y_pred)
print("Матрица ошибок:")
print(conf_matrix)

# Анализ неправильно классифицированных объектов
error_df = pd.DataFrame({
    'original_text': X_test,
    'cleaned_text': X_test,
    'true_label': y_test,
    'predicted_label': y_pred,
    'prediction_probabilities': list(y_pred_proba)
})

# Добавляем названия классов для удобства
label_names = {0: 'Нейтральный', 1: 'Оскорбительный', 2: 'Токсичный'}
error_df['true_class'] = error_df['true_label'].map(label_names)
error_df['predicted_class'] = error_df['predicted_label'].map(label_names)

# Находим различные типы ошибок
misclassified = error_df[error_df['true_label'] != error_df['predicted_label']]

print(f"\nВсего неправильно классифицировано: {len(misclassified)} из {len(X_test)} ({len(misclassified)/len(X_test)*100:.2f}%)")

# Анализируем конкретные типы ошибок
print("\n=== АНАЛИЗ ОШИБОК ===")

# 1. Токсичные, классифицированные как оскорбительные (серьезная ошибка)
toxic_as_offensive = misclassified[
    (misclassified['true_label'] == 2) & (misclassified['predicted_label'] == 1)
]
print(f"\nТоксичные, классифицированные как оскорбительные: {len(toxic_as_offensive)}")

# 2. Оскорбительные, классифицированные как токсичные (менее серьезная ошибка)
offensive_as_toxic = misclassified[
    (misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 2)
]
print(f"Оскорбительные, классифицированные как токсичные: {len(offensive_as_toxic)}")

# 3. Любые токсичные/оскорбительные, классифицированные как нейтральные (самая опасная ошибка)
toxic_as_neutral = misclassified[
    (misclassified['true_label'].isin([1, 2])) & (misclassified['predicted_label'] == 0)
]
print(f"Токсичные/оскорбительные, пропущенные (классифицированные как нейтральные): {len(toxic_as_neutral)}")

# Показываем примеры самых опасных ошибок
print("\n--- САМЫЕ ОПАСНЫЕ ОШИБКИ: Пропущенные токсичные комментарии ---")
for i, row in toxic_as_neutral.head(5).iterrows():
    print(f"\nИсходный текст: {row['original_text']}")
    print(f"Истинный класс: {row['true_class']}")
    print(f"Предсказанный класс: {row['predicted_class']}")
    probs = row['prediction_probabilities']
    print(f"Вероятности: Нейтральный={probs[0]:.3f}, Оскорбительный={probs[1]:.3f}, Токсичный={probs[2]:.3f}")

# Анализ сложных случаев
print("\n--- СЛОЖНЫЕ СЛУЧАИ: Токсичные vs Оскорбительные ---")
borderline_cases = misclassified[
    ((misclassified['true_label'] == 1) & (misclassified['predicted_label'] == 2)) |
    ((misclassified['true_label'] == 2) & (misclassified['predicted_label'] == 1))
].head(3)

for i, row in borderline_cases.iterrows():
    print(f"\nТекст: {row['original_text'][:200]}...")
    print(f"Истинный: {row['true_class']}, Предсказанный: {row['predicted_class']}")

