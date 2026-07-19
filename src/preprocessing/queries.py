"""Препроцессинг запросов пользователей.

Сначала исправляет опечатки с помощью sage-fredt5-distilled-95m,
затем лемматизирует как исходные, так и исправленные запросы.
"""

import pandas as pd
import spacy
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from src.config import (
    CALIBRATION_PATH,
    TEST_PATH,
    QUERIES_CORRECTED_PATH,
    QUERIES_RAW_PATH,
    READY_DATA_DIR,
    TYP0_MODEL_NAME,
    SPACY_MODEL_NAME,
)


def _init_spacy():
    """Загружает модель spaCy для русского языка с отключёнными NER и parser."""
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["ner", "parser", "textcat"])
    nlp.max_length = 5_000_000
    return nlp


def _init_typo_corrector():
    """Загружает токенизатор и модель sage-fredt5 для исправления опечаток.

    Возвращает:
        кортеж (tokenizer, model) из transformers.
    """
    tokenizer = AutoTokenizer.from_pretrained(TYP0_MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(TYP0_MODEL_NAME)
    model.eval()
    return tokenizer, model


def correct_text(text: str, tokenizer, model, max_length: int = 512) -> str:
    """Исправляет опечатки в тексте с помощью sage-fredt5.

    Аргументы:
        text: исходный текст с возможными опечатками.
        tokenizer: токенизатор модели.
        model: модель seq2seq для исправления.
        max_length: максимальная длина последовательности.

    Возвращает:
        исправленный текст.
    """
    inputs = tokenizer(
        text, return_tensors="pt", truncation=True, max_length=max_length
    )
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=max_length)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def lemmatize_text(text: str, nlp) -> str:
    """Лемматизирует текст с помощью spaCy.

    Аргументы:
        text: исходный текст.
        nlp: загруженная модель spaCy.

    Возвращает:
        строка из лемм, разделённых пробелами.
    """
    doc = nlp(text)
    return " ".join(token.lemma_ for token in doc)


def process_queries():
    """Загружает запросы, исправляет опечатки и лемматизирует.

    Сохраняет два датасета:
      - QUERIES_RAW_PATH: исходные запросы + леммы (baseline).
      - QUERIES_CORRECTED_PATH: исправленные запросы + леммы.
    """
    print("Загрузка запросов...")
    calibration = pd.read_feather(CALIBRATION_PATH)
    test = pd.read_feather(TEST_PATH)

    test_ids = test["query_id"].copy()
    test["query_id"] = test["query_id"] + 500

    queries = pd.concat(
        [
            calibration[["query_id", "query_text"]],
            test[["query_id", "query_text"]],
        ],
        ignore_index=True,
    )

    query_ids = queries["query_id"].tolist()
    raw_texts = queries["query_text"].tolist()

    test_ids_mapping = dict(zip(test["query_id"], test_ids))

    print("Загрузка spaCy...")
    nlp = _init_spacy()

    print("Лемматизация исходных запросов (baseline)...")
    raw_lemmas = []
    for text in tqdm(raw_texts, desc="Лемматизация исходных"):
        raw_lemmas.append(lemmatize_text(text, nlp))

    print("Загрузка корректора опечаток sage-fredt5...")
    tokenizer, model = _init_typo_corrector()

    print("Исправление опечаток...")
    corrected_texts = []
    for text in tqdm(raw_texts, desc="Исправление опечаток"):
        corrected_texts.append(correct_text(text, tokenizer, model))

    print("Лемматизация исправленных запросов...")
    corrected_lemmas = []
    for ctext in tqdm(corrected_texts, desc="Лемматизация исправленных"):
        corrected_lemmas.append(lemmatize_text(ctext, nlp))

    raw_df = pd.DataFrame({
        "query_id": query_ids,
        "query_text": raw_texts,
        "query_lemma": raw_lemmas,
    })

    corrected_df = pd.DataFrame({
        "query_id": query_ids,
        "query_text_corrected": corrected_texts,
        "query_lemma": corrected_lemmas,
    })

    print(f"Сохранение исходных запросов в {QUERIES_RAW_PATH}...")
    raw_df.to_feather(QUERIES_RAW_PATH)

    print(f"Сохранение исправленных запросов в {QUERIES_CORRECTED_PATH}...")
    corrected_df.to_feather(QUERIES_CORRECTED_PATH)

    print(f"Сохранение маппинга оригинальных test_id...")
    import json
    with open(READY_DATA_DIR / "test_ids_mapping.json", "w") as f:
        json.dump({str(k): v for k, v in test_ids_mapping.items()}, f)

    print(f"Готово. Обработано {len(raw_df)} запросов.")
    print(f"  Первый исходный:      {raw_texts[0][:80]}")
    print(f"  Первый исправленный: {corrected_texts[0][:80]}")
    return raw_df, corrected_df


if __name__ == "__main__":
    process_queries()
