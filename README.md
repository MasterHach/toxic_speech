# toxic_speech

Датасет: Jigsaw Toxic Comment Classification Challenge с переработанной разметкой для 3-х классовной классификации:

Класс 0 (Нейтральный): Комментарии без каких-либо токсичных меток

Класс 1 (Оскорбительный): Комментарии с метками toxic, obscene, insult

Класс 2 (Токсичный): Комментарии с метками severe_toxic, threat, identity_hate (наиболее токсичные)

Язык: английский

Размер: 159571 строк, 2 колонки
Пропуски: 0
Дубликаты: 0

=== РАСПРЕДЕЛЕНИЕ КЛАССОВ ===
Нейтральный: 143346 (89.8%)
Оскорбительный: 13238 (8.3%)
Токсичный: 2987 (1.9%)

Наблюдается сильный дисбаланс классов, будет исправлено весами

Методы классификации: 
1) Logistic Regression, Naive Bayes, Random Forest
2) Deep BiLSTM и Hybrid CNN-BiLSTM
3) BERT + Finetuning

Результаты обучения

Logistic Regression  (лучшее из классики)
Accuracy: 0.9380
Macro F1: 0.6841

Naive Bayes
Accuracy: 0.9345
Macro F1: 0.5945

Random Forest
Accuracy: 0.7400
Macro F1: 0.5111

--------------------------

Hybrid CNN-BiLSTM
Лучшая точность: 0.9305

Deep BiLSTM (лучшая)
Лучшая точность: 0.9403

-----------------------

BERT + Finetuning
Лучшая точность: 0.9398




