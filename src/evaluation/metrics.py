"""Метрики оценки качества ранжирования.

Содержит реализацию MAP@10, Precision@10 и Recall@10
для оценки результатов поиска статей Авито.
"""

import numpy as np
import pandas as pd


def average_precision(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Вычисляет Average Precision для одного запроса.

    Для каждой релевантной позиции в топ-k вычисляется точность
    на текущей позиции, результат усредняется по всем релевантным документам.

    Аргументы:
        ranked_ids: упорядоченный список идентификаторов статей.
        relevant_ids: множество релевантных идентификаторов.
        k: глубина просмотра топа.

    Возвращает:
        значение Average Precision (0.0, если нет релевантных).
    """
    hits = 0
    sum_precisions = 0.0
    for pos, aid in enumerate(ranked_ids[:k], 1):
        if aid in relevant_ids:
            hits += 1
            sum_precisions += hits / pos
    if not relevant_ids:
        return 0.0
    return sum_precisions / min(k, len(relevant_ids))


def mean_average_precision(
    predictions: dict[int, dict],
    relevance: pd.DataFrame,
    k: int = 10,
) -> float:
    """Вычисляет Mean Average Precision по всем запросам.

    Аргументы:
        predictions: словарь {query_id: {article_ids, scores, ranks}}.
        relevance: датасет с колонками query_id, article_id, target.
        k: глубина просмотра топа.

    Возвращает:
        усреднённое значение AP по всем запросам.
    """
    query_relevant = relevance.groupby("query_id")["article_id"].apply(set).to_dict()
    aps = []
    for qid, pred in predictions.items():
        relevant = query_relevant.get(qid, set())
        if not relevant:
            continue
        aps.append(average_precision(pred["article_ids"], relevant, k))
    return float(np.mean(aps)) if aps else 0.0


def precision_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Вычисляет Precision@k для одного запроса.

    Аргументы:
        ranked_ids: упорядоченный список идентификаторов статей.
        relevant_ids: множество релевантных идентификаторов.
        k: глубина просмотра топа.

    Возвращает:
        долю релевантных среди первых k документов.
    """
    if not ranked_ids or k == 0:
        return 0.0
    hits = sum(1 for aid in ranked_ids[:k] if aid in relevant_ids)
    return hits / k


def recall_at_k(ranked_ids: list[int], relevant_ids: set[int], k: int = 10) -> float:
    """Вычисляет Recall@k для одного запроса.

    Аргументы:
        ranked_ids: упорядоченный список идентификаторов статей.
        relevant_ids: множество релевантных идентификаторов.
        k: глубина просмотра топа.

    Возвращает:
        долю найденных релевантных среди всех релевантных.
    """
    if not relevant_ids:
        return 0.0
    hits = sum(1 for aid in ranked_ids[:k] if aid in relevant_ids)
    return hits / len(relevant_ids)


def compute_metrics(
    predictions: dict[int, dict],
    relevance: pd.DataFrame,
    k: int = 10,
) -> dict[str, float]:
    """Вычисляет MAP@k, Precision@k и Recall@k по всем запросам.

    Аргументы:
        predictions: словарь {query_id: {article_ids, scores, ranks}}.
        relevance: датасет с колонками query_id, article_id, target.
        k: глубина просмотра топа.

    Возвращает:
        словарь с ключами MAP, P@{k}, R@{k}.
    """
    query_relevant = relevance.groupby("query_id")["article_id"].apply(set).to_dict()
    precisions, recalls, aps = [], [], []
    for qid, pred in predictions.items():
        relevant = query_relevant.get(qid, set())
        precisions.append(precision_at_k(pred["article_ids"], relevant, k))
        recalls.append(recall_at_k(pred["article_ids"], relevant, k))
        aps.append(average_precision(pred["article_ids"], relevant, k))
    return {
        "MAP": float(np.mean(aps)),
        f"P@{k}": float(np.mean(precisions)),
        f"R@{k}": float(np.mean(recalls)),
    }
