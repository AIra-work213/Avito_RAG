"""Финальное обучение CatBoost-ранжировщика для поиска статей Авито.

Пайплайн:
  1. Загрузка BM25-результатов, статей, запросов, релевантности.
  2. Построение признаков (BM25-скор, длина, отношение длин).
  3. Подготовка целевой переменной (1 — релевантен, 0 — нет).
  4. Кросс-валидация (5-fold GroupKFold) для контроля качества.
  5. Обучение финальной модели на всех calibration-данных.
  6. Предсказание на test-запросах.
  7. Сохранение результата в answers.csv.

Гиперпараметры CatBoost зафиксированы в config.py (LTR_PARAMS).
Все случайные сиды зафиксированы для воспроизводимости.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold

from src.config import READY_DATA_DIR, LTR_PARAMS, TOP_K_FINAL
from src.retrieval.bm25 import load_retrieval_results
from src.evaluation.metrics import compute_metrics


def _load_articles() -> pd.DataFrame:
    """Загружает обработанные статьи и вычисляет длину каждой статьи.

    Длина считается как сумма длин лемматизированных заголовка и тела.

    Возвращает:
        DataFrame с колонками article_id, title_lemma, body_lemma, article_len.
    """
    df = pd.read_feather(READY_DATA_DIR / "articles_processed.f")
    df["article_len"] = (
        df["title_lemma"].fillna("").str.len()
        + df["body_lemma"].fillna("").str.len()
    )
    return df


def _build_features(
    bm25_results: dict,
    articles_df: pd.DataFrame,
    queries_df: pd.DataFrame,
) -> pd.DataFrame:
    """Строит матрицу признаков для обучения ранжировщика.

    Для каждой пары (запрос, статья) вычисляются:
      - bm25_score: комбинированный BM25-скор (title + body).
      - title_score: BM25-скор по заголовку.
      - body_score: BM25-скор по телу.
      - bm25_rank: позиция статьи в BM25-выдаче.
      - article_len: общая длина статьи.
      - query_len: длина запроса.
      - len_ratio: отношение длины статьи к длине запроса.

    Аргументы:
        bm25_results: словарь {query_id: {article_ids, scores, ...}}.
        articles_df: датасет статей с колонкой article_len.
        queries_df: датасет запросов с колонкой query_lemma.

    Возвращает:
        DataFrame с колонками query_id, article_id и признаками.
    """
    article_map = articles_df.set_index("article_id")[["article_len"]].to_dict("index")

    query_map = {}
    for _, row in queries_df.iterrows():
        qid = int(row["query_id"])
        query_map[qid] = len(str(row.get("query_lemma", "")))

    rows = []
    for qid, pred in bm25_results.items():
        for i, aid in enumerate(pred["article_ids"]):
            alen = article_map.get(aid, {}).get("article_len", 0)
            qlen = query_map.get(qid, 1)
            rows.append({
                "query_id": qid,
                "article_id": aid,
                "bm25_score": pred["scores"][i],
                "title_score": pred.get("title_scores", [0])[i],
                "body_score": pred.get("body_scores", [0])[i],
                "bm25_rank": pred["ranks"][i],
                "article_len": alen,
                "query_len": qlen,
                "len_ratio": alen / max(qlen, 1),
            })
    return pd.DataFrame(rows)


def _add_targets(features_df: pd.DataFrame, relevance: pd.DataFrame) -> pd.DataFrame:
    """Добавляет целевую переменную для обучения.

    Статья считается релевантной запросу, если target > 0
    в датасете calibration_targets.

    Аргументы:
        features_df: DataFrame с признаками.
        relevance: датасет с колонками query_id, article_id, target.

    Возвращает:
        features_df с добавленной колонкой target (1 — релевантна, 0 — нет).
    """
    qrel = relevance.groupby("query_id")["article_id"].apply(set).to_dict()
    targets = []
    for _, row in features_df.iterrows():
        relevant = qrel.get(int(row["query_id"]), set())
        targets.append(1 if int(row["article_id"]) in relevant else 0)
    features_df["target"] = targets
    return features_df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Возвращает список колонок-признаков для обучения.

    Исключает служебные колонки query_id, article_id, target,
    а также нечисловые колонки.

    Аргументы:
        df: DataFrame с признаками.

    Возвращает:
        список имён числовых колонок-признаков.
    """
    return [
        c for c in df.columns
        if c not in ("query_id", "article_id", "target")
        and df[c].dtype in (int, float)
    ]


def cross_validate(features_df: pd.DataFrame, n_folds: int = 5) -> list[float]:
    """Выполняет кросс-валидацию модели с GroupKFold.

    Запросы одного пользователя не должны попадать одновременно
    в train и val, поэтому используется GroupKFold по query_id.

    Аргументы:
        features_df: DataFrame с признаками и целевой переменной.
        n_folds: количество фолдов.

    Возвращает:
        список значений MAP@10 на валидации по фолдам.
    """
    gkf = GroupKFold(n_splits=n_folds)
    feature_cols = _get_feature_cols(features_df)
    groups = features_df["query_id"].values

    cv_scores = []
    base_params = {k: v for k, v in LTR_PARAMS.items() if k != "verbose"}
    base_params["verbose"] = 0

    for fold, (train_idx, val_idx) in enumerate(gkf.split(features_df, groups=groups)):
        train = features_df.iloc[train_idx]
        val = features_df.iloc[val_idx]

        model = CatBoostRanker(**base_params)
        train_pool = Pool(
            train[feature_cols].values,
            label=train["target"].values,
            group_id=train["query_id"].values,
        )
        model.fit(train_pool)

        val_copy = val.copy()
        val_copy["ltr_score"] = model.predict(val[feature_cols].values)
        val_preds = {}
        for qid, group in val_copy.groupby("query_id"):
            group = group.sort_values("ltr_score", ascending=False)
            val_preds[int(qid)] = {
                "article_ids": group["article_id"].tolist(),
                "scores": group["ltr_score"].tolist(),
                "ranks": list(range(1, len(group) + 1)),
            }
        relevance = pd.read_feather(READY_DATA_DIR / "calibration_targets.f")
        m = compute_metrics(val_preds, relevance)
        cv_scores.append(m["MAP"])
        print(f"  Фолд {fold + 1}: MAP={m['MAP']:.4f}")

    print(f"CV MAP@10: {np.mean(cv_scores):.4f} +/- {np.std(cv_scores):.4f}")
    return cv_scores


def train_ltr(features_df: pd.DataFrame):
    """Обучает финальную CatBoost-модель на всех calibration-данных.

    Использует гиперпараметры из config.py (LTR_PARAMS).
    Сид зафиксирован (random_seed=42) для воспроизводимости.

    Аргументы:
        features_df: DataFrame с признаками и целевой переменной.

    Возвращает:
        кортеж (обученная модель, список колонок-признаков).
    """
    feature_cols = _get_feature_cols(features_df)
    X = features_df[feature_cols].values
    y = features_df["target"].values
    groups = features_df["query_id"].values

    model = CatBoostRanker(**LTR_PARAMS)
    model.fit(Pool(X, label=y, group_id=groups))
    return model, feature_cols


def predict_ltr(
    model: CatBoostRanker,
    feature_cols: list[str],
    test_df: pd.DataFrame,
) -> dict:
    """Выполняет предсказание для test-запросов.

    Для каждого запроса выбирает TOP_K_FINAL статей
    с наибольшим скором от ранжировщика.

    Аргументы:
        model: обученная CatBoost-модель.
        feature_cols: список колонок-признаков.
        test_df: DataFrame с признаками для test-запросов.

    Возвращает:
        словарь {query_id: {article_ids, scores, ranks}}.
    """
    X = test_df[feature_cols].values
    test_copy = test_df.copy()
    test_copy["ltr_score"] = model.predict(X)
    results = {}
    for qid, group in test_copy.groupby("query_id"):
        group = group.sort_values("ltr_score", ascending=False).head(TOP_K_FINAL)
        results[int(qid)] = {
            "article_ids": group["article_id"].tolist(),
            "scores": group["ltr_score"].tolist(),
            "ranks": list(range(1, len(group) + 1)),
        }
    return results


def main() -> None:
    """Основная функция пайплайна обучения и предсказания.

    Последовательность:
      1. Загрузка BM25-результатов для calibration и test.
      2. Загрузка статей и запросов.
      3. Построение признаков для calibration.
      4. Кросс-валидация (5-fold).
      5. Обучение финальной модели.
      6. Построение признаков для test.
      7. Предсказание и сохранение answers.csv.
    """
    print("Загрузка BM25-результатов...")
    bm25_calib, bm25_test = load_retrieval_results(
        READY_DATA_DIR / "bm25_calib.f",
        READY_DATA_DIR / "bm25_test.f",
    )

    print("Загрузка статей...")
    articles = _load_articles()

    print("Загрузка запросов...")
    queries = pd.read_feather(READY_DATA_DIR / "queries_raw.f")

    print("Загрузка релевантности...")
    relevance = pd.read_feather(READY_DATA_DIR / "calibration_targets.f")

    print("Построение признаков для calibration...")
    calib_queries = queries[queries["query_id"] <= 500].copy()
    calib_features = _build_features(bm25_calib, articles, calib_queries)
    calib_features = _add_targets(calib_features, relevance)
    print(f"  {len(calib_features)} строк, {calib_features['query_id'].nunique()} запросов")

    print("Кросс-валидация (5-fold GroupKFold)...")
    cross_validate(calib_features, n_folds=5)

    print("Обучение финальной модели на всех данных calibration...")
    model, feature_cols = train_ltr(calib_features)
    print(f"  Признаки: {feature_cols}")

    print("Построение признаков для test и предсказание...")
    test_queries = queries[queries["query_id"] > 500].copy()
    test_features = _build_features(bm25_test, articles, test_queries)
    test_preds = predict_ltr(model, feature_cols, test_features)

    print("Сохранение результата в answer.csv...")
    with open(READY_DATA_DIR / "test_ids_mapping.json") as f:
        id_map = json.load(f)

    rows = []
    for qid, pred in test_preds.items():
        orig_qid = id_map.get(str(qid), qid)
        row = {"query_id": int(orig_qid), "answer": " ".join(str(a) for a in pred["article_ids"])}
        rows.append(row)

    result_df = pd.DataFrame(rows).sort_values("query_id")
    result_df.to_csv(READY_DATA_DIR / "answer.csv", index=False)
    print(f"Сохранён answer.csv: {len(result_df)} запросов")


if __name__ == "__main__":
    main()
