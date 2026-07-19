# Тестовое задание Авито

## Пайплайн

```
Сырые данные → Обработка → BM25 + dense кодирование заголовка и тела запроса через intfloat/multilingual-e5-small с INT8-квантизацией + dense и для BM25→ LTR → answers.csv
```

### 1. Обработка данных
- **Статьи:** HTML → Markdown (через `html2text`, таблицы линеаризуются), лемматизация spaCy.
- **Запросы:** исправление опечаток через `sage-fredt5-distilled-95m`, лемматизация spaCy (проверял оба варианта и с исправлением опечаток и без)

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

**Результат:** CV MAP@10 = **0.5017** (текущий результат среднее по фолдам целевой метрики)

---



### Подобранные гиперпараметры
- v1 использовал depth=5 без l2_leaf_reg и border_count.
- Holdout-тюнинг (72 комбинации depth, lr, l2, border_count) нашёл: depth=6, lr=0.1, l2=1, border_count=255.
- **Результат:** CV поднялся с 0.4916 → **0.5017**.

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

# Шаг 4: обучение CatBoost LTR + предсказание вместе с dense скорами- answers.csv
uv run -- python3 -m src.reranking.run_ltr_dense_v2
```

### Если данные уже предобработаны

```bash
uv run -- python3 -m src.retrieval.bm25
uv run -- python3 -m src.reranking.run_ltr_dense_v2
```

В папке data/ready 
