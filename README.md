# Investment Prediction - Решение для хакатона Сбер

**Random Seed:** 42
**F1-Score (Validation):** 0.8340
**Лучшая модель:** Random Forest

---

## Структура решения

```
sber_investment_prediction/
├── train_FIXED.ipynb           # Код обучения модели
├── predict_FIXED.py            # Код для генерации предсказаний
├── requirements.txt            # Зависимости
├── best_model_seed_42.pkl      # Обученная модель
├── feature_names.pkl           # Список признаков
└── submission_seed_42.csv      # Результат для submission
```

---

## Быстрый запуск

### 1. Установка зависимостей

```bash
pip install -r requirements.txt
```

### 2. Обучение модели

```bash
jupyter notebook train_FIXED.ipynb
```

Запустите все ячейки (Kernel → Restart & Run All)

### 3. Генерация предсказаний

```bash
python predict_FIXED.py --model_path best_model_seed_42.pkl \
                        --test_path invest_test_public.csv \
                        --output_path submission_seed_42.csv
```

---

## Данные

**Признаки:**
- customer_id - ID клиента
- age - возраст
- balance - баланс счета
- risk_profile - профиль риска (low/medium/high)
- marketing_channel - канал маркетинга (email/phone/sms/in_branch)
- offer_amount - размер предложения
- previous_investments - предыдущие инвестиции (0/1)
- responded_before - отвечал ранее (0/1)
- membership_tier - уровень привилегий (standard/gold/platinum)
- accepted - целевая переменная (0/1, только в train)

**Размеры:**
- Train: 6000 строк
- Test: 1000 строк

---

## Feature Engineering

Созданные признаки:
1. offer_to_balance_ratio - отношение предложения к балансу
2. experienced_investor - опытный инвестор
3. age_group - группы по возрасту
4. balance_group - группы по балансу
5. offer_size - размер предложения
6. high_value_customer - клиент с высокой ценностью
7. is_active - активный клиент
8. One-Hot Encoding для категориальных признаков

Итого: 30 признаков после обработки

---

## Модели

Протестированы 4 модели:
- Random Forest (выбрана как финальная) - F1: 0.8340
- CatBoost
- XGBoost
- LightGBM

Все модели обучены с:
- random_state=42
- Cross-validation (5 фолдов)
- Балансировка классов (class_weight='balanced')
- Стратифицированное разбиение данных

---

## Результаты

**Validation Metrics:**
- F1-Score: 0.8340
- Model: Random Forest
- Features: 30
- Training samples: 6000

**Submission:**
- Файл: submission_seed_42.csv
- Формат: customer_id,accepted
- Размер: 1000 предсказаний

---

## Воспроизводимость

**Зафиксированные параметры:**
- RANDOM_SEED = 42
- np.random.seed(42)
- Все модели используют random_state=42

**Seed в названиях файлов:**
- best_model_seed_42.pkl
- submission_seed_42.csv

---

## Зависимости

```
pandas>=2.1.0
numpy>=1.26.0
scikit-learn>=1.3.0
xgboost>=2.0.0
lightgbm>=4.0.0
catboost>=1.2
matplotlib>=3.8.0
seaborn>=0.13.0
joblib>=1.3.0
jupyter>=1.0.0
notebook>=7.0.0
```

Python: 3.8+

---

### Обучение новой модели

```python
# В train_FIXED.ipynb измените RANDOM_SEED
RANDOM_SEED = 123

# Запустите все ячейки
# Будут созданы:
# - best_model_seed_123.pkl
# - submission_seed_123.csv
```

### Генерация предсказаний на новых данных

```bash
python predict_FIXED.py \
    --model_path best_model_seed_42.pkl \
    --test_path new_test_data.csv \
    --output_path new_submission.csv
```

---

## Submission

Готовый файл: submission_seed_42.csv

Формат:
```csv
customer_id,accepted
6000,1
6002,1
6004,0
...
```

---

## Техническая информация

**Время выполнения:**
- Обучение: 2-3 минуты
- Предсказания: <1 секунда

**Требования к памяти:**
- Обучение: ~500 MB RAM
- Inference: ~100 MB RAM

---

**Дата создания:** Октябрь 2025
**Хакатон:** Сбер - Investment Prediction
**Random Seed:** 42
**Validation F1-Score:** 0.8340
