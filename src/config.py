"""Конфигурационные параметры пайплайна поиска статей Авито.

Содержит пути к данным, гиперпараметры BM25,
параметры CatBoost-ранжировщика и прочие настройки.
"""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "candidate_data"
READY_DATA_DIR = PROJECT_ROOT / "data" / "ready"

ARTICLES_PATH = RAW_DATA_DIR / "articles.f"
CALIBRATION_PATH = RAW_DATA_DIR / "calibration.f"
TEST_PATH = RAW_DATA_DIR / "test.f"

ARTICLES_PROCESSED_PATH = READY_DATA_DIR / "articles_processed.f"
QUERIES_CORRECTED_PATH = READY_DATA_DIR / "queries_corrected.f"
QUERIES_RAW_PATH = READY_DATA_DIR / "queries_raw.f"

# Параметры BM25
BM25_PARAMS = {"k1": 0.6, "b": 0.4}
FIELD_WEIGHTS = {"title": 2.0, "body": 1.0}

# Параметры ранжирования
TOP_K_RETRIEVAL = 100
TOP_K_FINAL = 10

# Параметры CatBoost, подобранные через holdout-тюнинг
LTR_PARAMS = {
    "iterations": 500,
    "learning_rate": 0.1,
    "depth": 6,
    "l2_leaf_reg": 1,
    "border_count": 255,
    "loss_function": "YetiRank",
    "random_seed": 42,
    "verbose": 50,
}

# Параметры кросс-валидации
N_CALIBRATION_FOLDS = 5
RANDOM_STATE = 42

# Модель для лемматизации
SPACY_MODEL_NAME = "ru_core_news_lg"
