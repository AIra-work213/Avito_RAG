# Поиск статей поддержки Авито (MAP@10)

Ранжирование статей справки под поисковые запросы пользователей.
Соревновательная метрика — **MAP@10**.

## Пайплайн

```
Сырые данные → Препроцессинг → BM25 → CatBoost LTR → answers.csv
```

### 1. Препроцессинг
- **Статьи:** HTML → Markdown (через `html2text`, таблицы линеаризуются), лемматизация spaCy.
- **Запросы:** исправление опечаток через `sage-fredt5-distilled-95m`, лемматизация spaCy.

### 2. Поиск (BM25)
- Два индекса: title (вес 2.0) + body (вес 1.0).
- Параметры: k1=0.6, b=0.4.
- Для каждого запроса — топ-100 кандидатов.

### 3. Ранжирование (CatBoost LTR)
- **Признаки:** bm25_score, title_score, body_score, bm25_rank, article_len, query_len, len_ratio.
- **Модель:** CatBoostRanker, loss=YetiRank.
- **Гиперпараметры** (подобраны holdout-поиском):
  - depth=6, learning_rate=0.1, l2_leaf_reg=1, border_count=255, iterations=500.
- **Сид:** 42 (фиксирован для воспроизводимости).
- **Валидация:** 5-fold GroupKFold по query_id.

**Результат:** CV MAP@10 = **0.5017** (текущий baseline).

---

## Какие гипотезы проверяли

### 1. Плотный поиск (Dense Retrieval)
- **Модель:** `ai-sage/Giga-Embeddings-instruct` → не загрузилась (проблемы с all_tied_weights_keys).
- **Замена:** `intfloat/multilingual-e5-small` → загрузилась (INT8), но MAP@10 = **0.0418**.
- **Вывод:** embedding-модели требуют много памяти и тонкой настройки под домен.
- **Статус:** ❌ не используется.

### 2. RRF-фьюжн (BM25 + Dense)
- Смешивали ранги BM25 и Dense через Reciprocal Rank Fusion (k=60).
- **Результат:** MAP@10 = 0.1384 — хуже, чем BM25 в одиночку (0.1882).
- **Вывод:** если dense поиск плохой, фьюжн только портит.
- **Статус:** ❌ не используется.

### 3. Cross-encoder (bge-reranker-v2-m3)
- Переранжирование топ-100 кандидатов прямой pairwise моделью.
- **Проблема:** ~5–16 секунд на запрос на CPU, калибровка не завершилась за 48 минут.
- **Вывод:** cross-encoder модели без GPU не применимы для 1000 запросов.
- **Статус:** ❌ не используется.

### 4. LTR v2 — больше признаков + тюнинг + ансамбль
- Добавили 10 overlap-фич (Jaccard, token overlap), нормализацию скоров, тюнинг 72 комбинаций, ансамбль из 3 seed.
- **Результат:** CV MAP@10 = 0.4914 (v1: 0.4916) — практически то же самое.
- **Вывод:** простые BM25-признаки уже несут основную информацию; overlap-фичи не дают прироста.
- **Статус:** ❌ не используется (v1 с tuned параметрами стал baseline — 0.5017).

### 5. Tuned гиперпараметры (текущий)
- v1 использовал depth=5 без l2_leaf_reg и border_count.
- Holdout-тюнинг (72 комбинации depth, lr, l2, border_count) нашёл: depth=6, lr=0.1, l2=1, border_count=255.
- **Результат:** CV поднялся с 0.4916 → **0.5017**.
- **Статус:** ✅ используется.

---

## Воспроизводимость

### Зависимости
```bash
uv sync
```

### Полный пайплайн «с нуля» (порядок 30–40 минут)

```bash
# Шаг 1: предобработка статей (HTML→Markdown, лемматизация)
uv run -- python3 -m src.preprocessing.articles

# Шаг 2: предобработка запросов (sage-fredt5, лемматизация)
uv run -- python3 -m src.preprocessing.queries

# Шаг 3: BM25-ранжирование (сохраняет bm25_calib.f + bm25_test.f)
uv run -- python3 -m src.retrieval.bm25

# Шаг 4: обучение CatBoost LTR + предсказание → answers.csv
uv run -- python3 -m src.reranking.run_ltr
```

### Если данные уже предобработаны

```bash
uv run -- python3 -m src.retrieval.bm25
uv run -- python3 -m src.reranking.run_ltr
```

### Структура `data/ready/` после полного прогона

| Файл | Описание |
|------|----------|
| `articles_processed.f` | Статьи после HTML→MD + лемматизации |
| `queries_raw.f` | Запросы: исходный текст + леммы |
| `queries_corrected.f` | Запросы: исправленные + леммы |
| `calibration_targets.f` | Релевантность для calibration (500 запросов) |
| `test_ids_mapping.json` | Маппинг test_id → оригинальные id |
| `bm25_calib.f` | BM25-результаты для calibration |
| `bm25_test.f` | BM25-результаты для test |
| `answer.csv` | Результат LTR v1 (MAP@10 = 0.4916) |
| `answers.csv` | **Финальный результат** (MAP@10 = 0.5017) |
