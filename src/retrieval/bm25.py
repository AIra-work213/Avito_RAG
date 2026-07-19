"""BM25-ранжирование с field boosting.

Строит отдельные индексы для заголовка и тела статьи,
поддерживает field boosting (Title=2.0, Body=1.0).
"""

from collections import Counter

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.config import (
    ARTICLES_PROCESSED_PATH,
    QUERIES_RAW_PATH,
    BM25_PARAMS,
    FIELD_WEIGHTS,
    TOP_K_RETRIEVAL,
)


def tokenize(text: str) -> list[str]:
    """Разбивает текст на токены по пробелам.

    Аргументы:
        text: входной текст.

    Возвращает:
        список токенов.
    """
    return text.lower().split()


def compute_idf(corpus: list[list[str]]) -> dict[str, float]:
    """Вычисляет IDF для каждого термина в корпусе.

    IDF = ln(1 + (N - df + 0.5) / (df + 0.5))

    Аргументы:
        corpus: список документов, каждый — список токенов.

    Возвращает:
        словарь {термин: idf}.
    """
    n_docs = len(corpus)
    df = Counter()
    for doc in corpus:
        df.update(set(doc))
    idf = {}
    for term, freq in df.items():
        idf[term] = np.log(1 + (n_docs - freq + 0.5) / (freq + 0.5))
    return idf


class BM25Index:
    """BM25-индекс для одного поля (title или body).

    Аргументы:
        corpus: список документов в виде строк.
        k1: параметр насыщения частоты термина.
        b: параметр штрафа за длину.
    """

    def __init__(self, corpus: list[str], k1: float = 1.2, b: float = 0.75):
        self.tokenized = [tokenize(doc) for doc in corpus]
        self.doc_lengths = np.array([len(doc) for doc in self.tokenized])
        self.avg_dl = self.doc_lengths.mean() if len(self.doc_lengths) > 0 else 1.0
        self.k1 = k1
        self.b = b
        self.idf = compute_idf(self.tokenized)
        self.n_docs = len(corpus)
        self.doc_tf = [Counter(doc) for doc in self.tokenized]

    def score_all(self, query_tokens: list[str]) -> np.ndarray:
        """Вычисляет BM25-скор для запроса по всем документам.

        Использует векторизованные операции для скорости.

        Аргументы:
            query_tokens: токены запроса.

        Возвращает:
            массив скоров длины n_docs.
        """
        scores = np.zeros(self.n_docs, dtype=np.float64)
        for term in set(query_tokens):
            if term not in self.idf:
                continue
            idf_val = self.idf[term]
            for doc_idx in range(self.n_docs):
                tf = self.doc_tf[doc_idx].get(term, 0)
                if tf == 0:
                    continue
                doc_len = self.doc_lengths[doc_idx]
                numerator = tf * (self.k1 + 1)
                denominator = tf + self.k1 * (1 - self.b + self.b * doc_len / self.avg_dl)
                scores[doc_idx] += idf_val * numerator / denominator
        return scores


def build_bm25_scores_for_queries(
    articles: pd.DataFrame,
    queries: pd.DataFrame,
    bm25_title: BM25Index,
    bm25_body: BM25Index,
) -> dict[int, dict]:
    """Вычисляет BM25-скоры для набора запросов.

    Аргументы:
        articles: датасет статей.
        queries: датасет запросов с колонкой query_lemma.
        bm25_title: BM25-индекс для заголовков.
        bm25_body: BM25-индекс для тел статей.

    Возвращает:
        словарь {query_id: {article_ids, scores, ranks, ...}}.
    """
    all_scores = {}
    for _, row in tqdm(queries.iterrows(), total=len(queries), desc="BM25-ранжирование"):
        q_tokens = tokenize(row["query_lemma"])
        title_scores = bm25_title.score_all(q_tokens) * FIELD_WEIGHTS["title"]
        body_scores = bm25_body.score_all(q_tokens) * FIELD_WEIGHTS["body"]
        combined = title_scores + body_scores
        article_ids = articles["article_id"].values
        top_k = min(TOP_K_RETRIEVAL, len(combined))
        top_indices = np.argsort(combined)[::-1][:top_k]
        all_scores[int(row["query_id"])] = {
            "article_ids": article_ids[top_indices].tolist(),
            "scores": combined[top_indices].tolist(),
            "ranks": list(range(1, top_k + 1)),
            "title_scores": title_scores[top_indices].tolist(),
            "body_scores": body_scores[top_indices].tolist(),
        }
    return all_scores


def build_bm25_scores():
    """Строит BM25-индексы, ранжирует calibration и test запросы.

    Загружает обработанные статьи, строит BM25-индексы,
    вычисляет скоры отдельно для calibration и test.
    """
    print("Загрузка обработанных статей...")
    articles = pd.read_feather(ARTICLES_PROCESSED_PATH)
    print(f"Загружено {len(articles)} статей.")

    print("Загрузка запросов...")
    queries = pd.read_feather(QUERIES_RAW_PATH)

    print("Построение BM25-индекса для заголовков...")
    bm25_title = BM25Index(
        articles["title_lemma"].tolist(),
        k1=BM25_PARAMS["k1"],
        b=BM25_PARAMS["b"],
    )

    print("Построение BM25-индекса для тел статей...")
    bm25_body = BM25Index(
        articles["body_lemma"].tolist(),
        k1=BM25_PARAMS["k1"],
        b=BM25_PARAMS["b"],
    )

    calib_queries = queries[queries["query_id"] <= 500].copy()
    test_queries = queries[queries["query_id"] > 500].copy()

    print("Ранжирование calibration-запросов...")
    calib_scores = build_bm25_scores_for_queries(articles, calib_queries, bm25_title, bm25_body)

    print("Ранжирование test-запросов...")
    test_scores = build_bm25_scores_for_queries(articles, test_queries, bm25_title, bm25_body)

    print(f"Готово. Calibration: {len(calib_scores)} запросов, Test: {len(test_scores)} запросов.")
    return calib_scores, test_scores


def save_results(results: dict, path, extra_fields: list | None = None) -> None:
    """Сохраняет результаты ранжирования в формате Feather.

    Конвертирует словарь результатов в плоский DataFrame
    с колонками query_id, article_id, score, rank.

    Аргументы:
        results: словарь {query_id: {article_ids, scores, ranks}}.
        path: путь к выходному файлу .f.
        extra_fields: список дополнительных полей для сохранения
                     (например, title_scores, body_scores).
    """
    data = []
    for qid, res in results.items():
        for i, (aid, score, rank) in enumerate(zip(res["article_ids"], res["scores"], res["ranks"])):
            row = {"query_id": qid, "article_id": aid, "score": score, "rank": rank}
            if extra_fields:
                for field in extra_fields:
                    row[field] = res.get(field, [0] * len(res["article_ids"]))[i]
            data.append(row)
    pd.DataFrame(data).to_feather(path)


def load_retrieval_results(calib_path, test_path):
    """Загружает результаты ранжирования из Feather-файлов.

    Восстанавливает словарный формат {query_id: {article_ids, scores, ranks}}
    из плоского DataFrame с колонками query_id, article_id, score, rank.
    Дополнительные колонки (title_scores, body_scores и т.д.)
    также загружаются, если присутствуют в файле.

    Аргументы:
        calib_path: путь к файлу с результатами для calibration.
        test_path: путь к файлу с результатами для test.

    Возвращает:
        кортеж (calib_dict, test_dict), каждый словарь содержит
        article_ids, scores, ranks и дополнительные поля.
    """
    def _load(path):
        df = pd.read_feather(path)
        extra_cols = [c for c in df.columns if c not in ("query_id", "article_id", "score", "rank")]
        results = {}
        for qid, group in df.groupby("query_id"):
            group = group.sort_values("rank")
            result = {
                "article_ids": group["article_id"].tolist(),
                "scores": group["score"].tolist(),
                "ranks": group["rank"].tolist(),
            }
            for col in extra_cols:
                result[col] = group[col].tolist()
            results[int(qid)] = result
        return results
    return _load(calib_path), _load(test_path)


if __name__ == "__main__":
    from src.config import READY_DATA_DIR
    calib, test = build_bm25_scores()
    print("Сохранение результатов BM25...")
    save_results(calib, READY_DATA_DIR / "bm25_calib.f", extra_fields=["title_scores", "body_scores"])
    save_results(test, READY_DATA_DIR / "bm25_test.f", extra_fields=["title_scores", "body_scores"])
    first_qid = list(calib.keys())[0]
    print(f"Пример calibration-запроса {first_qid}:")
    print(f"  Топ-3 article_id: {calib[first_qid]['article_ids'][:3]}")
    print(f"  Топ-3 скоры: {[round(s, 4) for s in calib[first_qid]['scores'][:3]]}")
