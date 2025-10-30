import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score, roc_auc_score, average_precision_score
from sklearn.utils.class_weight import compute_class_weight
import nltk
import re
from collections import Counter
import warnings
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
                return {'и', 'в', 'во', 'не', 'что', 'он', 'на', 'я', 'с', 'со', 'как', 'а', 'то', 'все', 'она',
                       'так', 'его', 'но', 'да', 'ты', 'к', 'у', 'же', 'вы', 'за', 'бы', 'по', 'только', 'ее', 'мне'}
            elif self.language == 'english':
                return {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'}
            else:
                return set()
    
    def clean_text(self, text):
        """Очистка и предобработка текста"""
        if not isinstance(text, str) or pd.isna(text):
            return ""
        
        # Приведение к нижнему регистру
        text = text.lower()
        
        # Удаление специальных символов, оставляем буквы и основные знаки препинания
        text = re.sub(r'[^a-zA-Zа-яА-Я0-9\s.,!?]', '', text)
        
        # Удаление цифр
        text = re.sub(r'\d+', '', text)
        
        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()
        
        # Удаление стоп-слов и коротких слов
        words = text.split()
        words = [word for word in words if word not in self.stop_words and len(word) > 2]
        
        return ' '.join(words)
    
    def preprocess_dataframe(self, df, text_column, label_column):
        """Предобработка всего датафрейма"""
        df_clean = df.copy()
        print("Начата предобработка текстов...")
        
        # Применяем очистку к каждому тексту
        df_clean['cleaned_text'] = df_clean[text_column].apply(self.clean_text)
        
        # Удаляем пустые тексты после очистки
        initial_len = len(df_clean)
        df_clean = df_clean[df_clean['cleaned_text'].str.len() > 0]
        final_len = len(df_clean)
        
        print(f"Предобработка завершена. Удалено {initial_len - final_len} пустых текстов.")
        print(f"Осталось {final_len} текстов для анализа.")
        
        return df_clean

class ClassicalTextClassifier:
    def __init__(self, language='russian'):
        self.language = language
        self.models = {}
        self.vectorizers = {}
        self.metrics = {}
        self.preprocessor = TextPreprocessor(language)
        self.error_analysis = {}
        
    def load_and_prepare_data(self, file_path, text_column='cleaned_comment', label_column='label'):
        """Загрузка и подготовка данных"""
        print("Загрузка данных...")
        df = pd.read_csv(file_path)
        
        # Проверяем данные
        print(f"Размер исходных данных: {df.shape}")
        print(f"Колонки: {df.columns.tolist()}")
        print(f"Распределение классов:\n{df[label_column].value_counts().sort_index()}")
        
        # Предобработка
        df_processed = self.preprocessor.preprocess_dataframe(df, text_column, label_column)
        
        # Разделение на признаки и метки
        X = df_processed['cleaned_text']
        y = df_processed[label_column]
        
        # Стратифицированное разделение
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nДанные подготовлены:")
        print(f"Тренировочная выборка: {len(X_train)} samples")
        print(f"Тестовая выборка: {len(X_test)} samples")
        print(f"Распределение классов в тренировочной выборке:\n{pd.Series(y_train).value_counts().sort_index()}")
        
        return X_train, X_test, y_train, y_test
    
    def prepare_features(self, X_train, X_test):
        """Векторизация текстов с помощью TF-IDF"""
        vectorizer = TfidfVectorizer(
            max_features=10000, 
            ngram_range=(1, 3),  # Добавляем триграммы для лучшего захвата контекста
            stop_words=list(self.preprocessor.stop_words),
            min_df=2,
            max_df=0.9,
            analyzer='word'
        )
            
        X_train_vec = vectorizer.fit_transform(X_train)
        X_test_vec = vectorizer.transform(X_test)
        
        self.vectorizers['tfidf'] = vectorizer
        print(f"Векторизация TF-IDF завершена. Размерность: {X_train_vec.shape}")
        return X_train_vec, X_test_vec
    
    def calculate_class_weights(self, y_train):
        """Расчет весов классов для борьбы с дисбалансом"""
        classes = np.unique(y_train)
        weights = compute_class_weight('balanced', classes=classes, y=y_train)
        class_weights = dict(zip(classes, weights))
        
        print(f"Веса классов для борьбы с дисбалансом: {class_weights}")
        return class_weights
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Обучение классических моделей"""
        # Подготовка фич
        X_train_tfidf, X_test_tfidf = self.prepare_features(X_train, X_test)
        
        # Расчет весов классов
        class_weights = self.calculate_class_weights(y_train)
        
        # Конфигурация моделей с оптимизированными параметрами
        models_config = {
            'Logistic Regression': LogisticRegression(
                random_state=42, 
                class_weight=class_weights, 
                max_iter=2000,
                C=0.8,  # Более сильная регуляризация
                solver='liblinear'
            ),
            'Naive Bayes': MultinomialNB(
                alpha=0.5,  # Сглаживание для редких слов
                fit_prior=True
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=150, 
                random_state=42, 
                class_weight=class_weights,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt'
            )
        }
        
        # Обучение и оценка моделей
        for name, model in models_config.items():
            print(f"\n🔧 Обучение {name}...")
            model.fit(X_train_tfidf, y_train)
            
            # Предсказания
            y_pred = model.predict(X_test_tfidf)
            y_proba = model.predict_proba(X_test_tfidf)
            
            # Расчет метрик
            self._calculate_metrics(name, y_test, y_pred, y_proba, X_test)
            self.models[name] = model
            
        print("\n✅ Обучение всех моделей завершено!")
    
    def _calculate_metrics(self, model_name, y_true, y_pred, y_proba, X_test):
        """Расчет метрик для модели"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'macro_f1': f1_score(y_true, y_pred, average='macro'),
            'weighted_f1': f1_score(y_true, y_pred, average='weighted'),
            'confusion_matrix': confusion_matrix(y_true, y_pred),
            'classification_report': classification_report(y_true, y_pred, output_dict=True)
        }
        
        # ROC-AUC и PR-AUC
        try:
            metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_proba, multi_class='ovr', average='macro')
            metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_proba, multi_class='ovo', average='macro')
            metrics['pr_auc'] = average_precision_score(y_true, y_proba, average='macro')
        except Exception as e:
            print(f"Предупреждение: не удалось рассчитать AUC для {model_name}: {e}")
            metrics['roc_auc_ovr'] = 0
            metrics['roc_auc_ovo'] = 0
            metrics['pr_auc'] = 0
            
        self.metrics[model_name] = metrics
        
        # Анализ ошибок
        self._analyze_errors(model_name, y_true, y_pred, X_test)
        
        # Вывод результатов
        print(f"\n📊 РЕЗУЛЬТАТЫ {model_name.upper()}:")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Macro F1: {metrics['macro_f1']:.4f}")
        print(f"Weighted F1: {metrics['weighted_f1']:.4f}")
        print(f"ROC-AUC (OvR): {metrics['roc_auc_ovr']:.4f}")
        print(f"PR-AUC: {metrics['pr_auc']:.4f}")
        
        # Детальный отчет по классам
        print(f"\nДетальный отчет по классам:")
        print(classification_report(y_true, y_pred, target_names=['Нейтральный', 'Оскорбительный', 'Токсичный']))
        
        # Вывод confusion matrix
        self._print_confusion_matrix(model_name, metrics['confusion_matrix'])
    
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
            print(f"{class_name}: Precision={precision:.3f}, Recall={recall:.3f}")
    
    def _analyze_errors(self, model_name, y_true, y_pred, X_test):
        """Детальный анализ ошибок классификации"""
        errors = {
            'false_positives': [],  # Нейтральные предсказаны как опасные
            'false_negatives': [],  # Опасные предсказаны как нейтральные
            'confusions': []        # Ошибки между оскорбительным и токсичным
        }
        
        for i, (true, pred, text) in enumerate(zip(y_true, y_pred, X_test)):
            text_length = len(str(text).split())
            
            if true == 0 and pred != 0:  # False Positive
                errors['false_positives'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
            elif true != 0 and pred == 0:  # False Negative
                errors['false_negatives'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
            elif true != pred and true != 0 and pred != 0:  # Confusion
                errors['confusions'].append({
                    'text': text,
                    'true_class': true,
                    'pred_class': pred,
                    'length': text_length
                })
        
        self.error_analysis[model_name] = errors
        
        # Статистика ошибок
        print(f"\n🔍 АНАЛИЗ ОШИБОК ДЛЯ {model_name}:")
        print(f"False Positives (нейтральные → опасные): {len(errors['false_positives'])}")
        print(f"False Negatives (опасные → нейтральные): {len(errors['false_negatives'])}")
        print(f"Confusions (ошибки между классами): {len(errors['confusions'])}")
        
        # Анализ длины текстов в ошибках
        if errors['false_positives']:
            fp_lengths = [err['length'] for err in errors['false_positives']]
            print(f"Средняя длина False Positive текстов: {np.mean(fp_lengths):.1f} слов")
        
        if errors['false_negatives']:
            fn_lengths = [err['length'] for err in errors['false_negatives']]
            print(f"Средняя длина False Negative текстов: {np.mean(fn_lengths):.1f} слов")
    
    def plot_confusion_matrices(self):
        """Визуализация confusion matrices"""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        for idx, (model_name, metrics) in enumerate(self.metrics.items()):
            cm = metrics['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx],
                       xticklabels=['Нейтр', 'Оскорб', 'Токсич'],
                       yticklabels=['Нейтр', 'Оскорб', 'Токсич'],
                       cbar_kws={'shrink': 0.8})
            axes[idx].set_title(f'{model_name}\nAccuracy: {metrics["accuracy"]:.3f}', fontsize=12)
            axes[idx].set_xlabel('Предсказанный класс', fontsize=10)
            axes[idx].set_ylabel('Истинный класс', fontsize=10)
        
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
            colors = ['skyblue', 'lightgreen', 'coral']
            metrics_df[metric].plot(kind='bar', ax=ax, color=colors)
            ax.set_title(f'Сравнение {metric}', fontsize=14)
            ax.set_ylabel(metric, fontsize=12)
            ax.tick_params(axis='x', rotation=45)
            
            # Добавление значений на столбцы
            for p in ax.patches:
                ax.annotate(f'{p.get_height():.3f}', 
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='bottom', fontsize=10)
        
        # Убираем лишний subplot
        axes[5].set_visible(False)
        
        plt.tight_layout()
        plt.show()
        
        return metrics_df
    
    def detailed_error_analysis(self, X_test, y_test):
        """Детальный анализ ошибок по всем моделям"""
        print("\n" + "="*80)
        print("ДЕТАЛЬНЫЙ АНАЛИЗ ОШИБОК КЛАССИФИКАЦИИ")
        print("="*80)
        
        # Анализ длины текстов
        text_lengths = [len(str(text).split()) for text in X_test]
        
        plt.figure(figsize=(15, 10))
        
        # 1. Распределение длины текстов
        plt.subplot(2, 3, 1)
        plt.hist(text_lengths, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        plt.xlabel('Длина текста (слов)')
        plt.ylabel('Частота')
        plt.title('Распределение длины текстов')
        plt.axvline(np.mean(text_lengths), color='red', linestyle='--', label=f'Среднее: {np.mean(text_lengths):.1f}')
        plt.legend()
        
        # 2. Распределение классов
        plt.subplot(2, 3, 2)
        class_distribution = pd.Series(y_test).value_counts().sort_index()
        colors = ['green', 'orange', 'red']
        class_distribution.plot(kind='bar', color=colors)
        plt.xlabel('Класс')
        plt.ylabel('Количество')
        plt.title('Распределение классов')
        plt.xticks([0, 1, 2], ['Нейтральный', 'Оскорбительный', 'Токсичный'], rotation=0)
        
        # 3. Сравнение ошибок по моделям
        plt.subplot(2, 3, 3)
        error_types = ['False Positives', 'False Negatives', 'Confusions']
        error_data = {}
        
        for model_name in self.metrics.keys():
            errors = self.error_analysis[model_name]
            error_data[model_name] = [
                len(errors['false_positives']),
                len(errors['false_negatives']), 
                len(errors['confusions'])
            ]
        
        x = np.arange(len(error_types))
        width = 0.25
        multiplier = 0
        
        for model_name, errors in error_data.items():
            offset = width * multiplier
            plt.bar(x + offset, errors, width, label=model_name)
            multiplier += 1
        
        plt.xlabel('Тип ошибки')
        plt.ylabel('Количество ошибок')
        plt.title('Распределение ошибок по типам')
        plt.xticks(x + width, error_types, rotation=45)
        plt.legend()
        
        # 4. Длина текстов в ошибках
        plt.subplot(2, 3, 4)
        length_data = []
        labels = []
        
        for model_name, errors in self.error_analysis.items():
            if errors['false_positives']:
                fp_lengths = [err['length'] for err in errors['false_positives']]
                length_data.append(np.mean(fp_lengths))
                labels.append(f'{model_name}\nFP')
            
            if errors['false_negatives']:
                fn_lengths = [err['length'] for err in errors['false_negatives']]
                length_data.append(np.mean(fn_lengths))
                labels.append(f'{model_name}\nFN')
        
        plt.bar(labels, length_data, color=['lightblue', 'lightcoral'] * 3)
        plt.ylabel('Средняя длина (слов)')
        plt.title('Средняя длина текстов в ошибках')
        plt.xticks(rotation=45)
        
        # 5. Сравнение F1-score по классам
        plt.subplot(2, 3, 5)
        class_f1_scores = {}
        
        for model_name, metrics in self.metrics.items():
            report = metrics['classification_report']
            class_f1_scores[model_name] = [
                report['0']['f1-score'],
                report['1']['f1-score'], 
                report['2']['f1-score']
            ]
        
        x = np.arange(3)
        width = 0.25
        multiplier = 0
        
        for model_name, f1_scores in class_f1_scores.items():
            offset = width * multiplier
            plt.bar(x + offset, f1_scores, width, label=model_name)
            multiplier += 1
        
        plt.xlabel('Класс')
        plt.ylabel('F1-Score')
        plt.title('F1-Score по классам')
        plt.xticks(x + width, ['Нейтральный', 'Оскорбительный', 'Токсичный'], rotation=45)
        plt.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Вывод статистики
        print(f"\n📈 СТАТИСТИКА ДАННЫХ:")
        print(f"Общее количество тестовых样本: {len(y_test)}")
        print(f"Средняя длина текста: {np.mean(text_lengths):.1f} ± {np.std(text_lengths):.1f} слов")
        print(f"Медианная длина: {np.median(text_lengths):.1f} слов")
        
        print(f"\n📊 РАСПРЕДЕЛЕНИЕ КЛАССОВ:")
        for class_idx, count in class_distribution.items():
            class_name = ['Нейтральный', 'Оскорбительный', 'Токсичный'][class_idx]
            percentage = count / len(y_test) * 100
            print(f"  {class_name}: {count} samples ({percentage:.1f}%)")
        
        # Анализ причин ошибок
        print(f"\n🔍 ПРИЧИНЫ ОШИБОК КЛАССИФИКАЦИИ:")
        print("1. ДИСБАЛАНС КЛАССОВ: Нейтральных текстов значительно больше, чем опасных")
        print("2. ДЛИНА ТЕКСТА: Короткие тексты сложнее классифицировать")
        print("3. КОНТЕКСТ: Одинаковые слова в разном контексте могут иметь разную тональность")
        print("4. САРКАЗМ И ИРОНИЯ: Сложно детектировать без глубокого понимания контекста")
        print("5. НОВАЯ ЛЕКСИКА: Модель может не знать новых оскорбительных выражений")

  

def run_classical_models(file_path, text_column='cleaned_comment', label_column='label', language='russian'):
    """
    Полный запуск классических моделей для классификации текстов
    """
    print("🎯 ЗАПУСК КЛАССИЧЕСКИХ МОДЕЛЕЙ МАШИННОГО ОБУЧЕНИЯ")
    print("="*70)
    print(f"Язык: {language}")
    print(f"Файл данных: {file_path}")
    print("="*70)
    
    # Инициализация классификатора
    classifier = ClassicalTextClassifier(language=language)
    
    # Загрузка и подготовка данных
    X_train, X_test, y_train, y_test = classifier.load_and_prepare_data(
        file_path, text_column, label_column
    )
    
    # Обучение моделей
    classifier.train_models(X_train, y_train, X_test, y_test)
    
    # Визуализация результатов
    classifier.plot_confusion_matrices()
    metrics_df = classifier.plot_metrics_comparison()
    
    # Детальный анализ ошибок
    classifier.detailed_error_analysis(X_test, y_test)
    
    # Вывод итоговой таблицы
    print("\n" + "="*80)
    print("ИТОГОВАЯ ТАБЛИЦА РЕЗУЛЬТАТОВ")
    print("="*80)
    print(metrics_df.round(4))
    
    # Анализ лучшей модели
    best_model = metrics_df['Macro F1'].idxmax()
    best_score = metrics_df.loc[best_model, 'Macro F1']
    print(f"\n🏆 ЛУЧШАЯ МОДЕЛЬ: {best_model} (Macro F1: {best_score:.4f})")
    
    
    return classifier

if __name__ == "__main__":
    # Пример использования
    try:
        results = run_classical_models(
            file_path='train_m.csv',  # Замените на путь к вашему файлу
            text_column='comment_text', 
            label_column='label',
            language='english' 
        )
      
