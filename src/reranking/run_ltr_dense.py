"""Финальное обучение CatBoost-ранжировщика с dense-эмбеддингами (e5-small).

Добавляет к BM25-признакам:
  - dense_score — косинусная близость эмбеддингов запроса и статьи.
  - dense_rank — позиция статьи по dense_score среди всех статей.

Пайплайн: BM25 → эмбеддинги e5-small → CatBoost (+dense_score, +dense_rank) → answer.csv.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.model_selection import GroupKFold

from src.config import READY_DATA_DIR, LTR_PARAMS, TOP_K_FINAL
from src.retrieval.bm25 import load_retrieval_results
from src.evaluation.metrics import compute_metrics
from src.embeddings.embedder import build_embeddings


def _build_features_dense(
    bm25_results: dict,
    articles_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    query_emb: np.ndarray,
    article_emb: np.ndarray,
    article_map: dict,
) -> pd.DataFrame:
    """Строит признаковое описание с dense_score и dense_rank.

    Для каждого BM25-кандидата вычисляется косинусная близость между
    эмбеддингом запроса и статьи (dense_score), а также ранг статьи
    по этой близости среди всех статей (dense_rank).

    Аргументы:
        bm25_results: словарь {query_id: {article_ids, scores, ...}}.
        articles_df: датасет статей с article_len.
        queries_df: датасет запросов с query_lemma.
        query_emb: эмбеддинги запросов (1000, 384).
        article_emb: эмбеддинги статей (793, 384).
        article_map: {article_id: {"idx": int}}.

    Возвращает:
        DataFrame с колонками query_id, article_id и признаками.
    """
    article_map_full = articles_df.set_index("article_id")[["article_len"]].to_dict("index")

    query_len_map = {}
    for _, row in queries_df.iterrows():
        query_len_map[int(row["query_id"])] = len(str(row.get("query_lemma", "")))

    query_emb_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    article_emb_norm = article_emb / (np.linalg.norm(article_emb, axis=1, keepdims=True) + 1e-9)

    sim_matrix = query_emb_norm @ article_emb_norm.T
    n_articles = sim_matrix.shape[1]

    rows = []
    for qid, pred in bm25_results.items():
        qidx = qid - 1
        sorted_by_dense = np.argsort(-sim_matrix[qidx])
        dense_rank_map = {int(aidx): r + 1 for r, aidx in enumerate(sorted_by_dense)}

        for i, aid in enumerate(pred["article_ids"]):
            alen = article_map_full.get(aid, {}).get("article_len", 0)
            qlen = query_len_map.get(qid, 1)
            aidx = article_map.get(aid, {}).get("idx", 0)

            if aidx < n_articles:
                dense_score = float(sim_matrix[qidx, aidx])
                dense_rank = dense_rank_map.get(aidx, 100)
            else:
                dense_score = 0.0
                dense_rank = 100

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
                "dense_score": dense_score,
                "dense_rank": dense_rank,
            })
    return pd.DataFrame(rows)


def _add_targets(features_df: pd.DataFrame, relevance: pd.DataFrame) -> pd.DataFrame:
    """Добавляет целевую переменную (1 — релевантна, 0 — нет).

    Аргументы:
        features_df: DataFrame с признаками.
        relevance: датасет с колонками query_id, article_id, target.

    Возвращает:
        features_df с колонкой target.
    """
    qrel = relevance.groupby("query_id")["article_id"].apply(set).to_dict()
    targets = []
    for _, row in features_df.iterrows():
        relevant = qrel.get(int(row["query_id"]), set())
        targets.append(1 if int(row["article_id"]) in relevant else 0)
    features_df["target"] = targets
    return features_df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """Возвращает список колонок-признаков, исключая служебные.

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


def cross_validate(features_df: pd.DataFrame, n_folds: int = 5) -> float:
    """Выполняет кросс-валидацию с GroupKFold.

    Аргументы:
        features_df: DataFrame с признаками и target.
        n_folds: количество фолдов.

    Возвращает:
        средний MAP@10 по фолдам.
    """
    gkf = GroupKFold(n_splits=n_folds)
    feature_cols = _get_feature_cols(features_df)
    groups = features_df["query_id"].values

    base_params = {k: v for k, v in LTR_PARAMS.items() if k != "verbose"}
    base_params["verbose"] = 0

    cv_scores = []
    for fold, (train_idx, val_idx) in enumerate(gkf.split(features_df, groups=groups)):
        train = features_df.iloc[train_idx]
        val = features_df.iloc[val_idx]

        model = CatBoostRanker(**base_params)
        model.fit(Pool(
            train[feature_cols].values,
            label=train["target"].values,
            group_id=train["query_id"].values,
        ))

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

    mean_cv = float(np.mean(cv_scores))
    print(f"CV MAP@10: {mean_cv:.4f} +/- {np.std(cv_scores):.4f}")
    return mean_cv


def main() -> None:
    """Основная функция пайплайна с dense-признаками.

    1. Загружает BM25-результаты, статьи, запросы.
    2. Строит или загружает эмбеддинги e5-small.
    3. Вычисляет dense_score и dense_rank для BM25-кандидатов.
    4. Кросс-валидация с плотными признаками.
    5. Обучение финальной модели и сохранение answer.csv.
    """
    print("Загрузка BM25-результатов...")
    bm25_calib, bm25_test = load_retrieval_results(
        READY_DATA_DIR / "bm25_calib.f",
        READY_DATA_DIR / "bm25_test.f",
    )

    print("Загрузка статей...")
    articles = pd.read_feather(READY_DATA_DIR / "articles_processed.f")
    articles["article_len"] = (
        articles["title_lemma"].fillna("").str.len()
        + articles["body_lemma"].fillna("").str.len()
    )

    print("Загрузка запросов...")
    queries = pd.read_feather(READY_DATA_DIR / "queries_raw.f")

    print("Загрузка релевантности...")
    relevance = pd.read_feather(READY_DATA_DIR / "calibration_targets.f")

    print("Построение или загрузка эмбеддингов e5-small...")
    article_emb, query_emb = build_embeddings(force=False)

    print("Построение карты статей (article_id → позиция в embedding-матрице)...")
    article_map = {}
    for idx, aid in enumerate(articles["article_id"]):
        article_map[int(aid)] = {"idx": idx}

    print("Построение признаков для calibration...")
    calib_queries = queries[queries["query_id"] <= 500].copy()
    calib_features = _build_features_dense(
        bm25_calib, articles, calib_queries,
        query_emb, article_emb, article_map,
    )
    calib_features = _add_targets(calib_features, relevance)
    feature_cols = _get_feature_cols(calib_features)
    print(f"  {len(calib_features)} строк, "
          f"{calib_features['query_id'].nunique()} запросов, "
          f"{len(feature_cols)} признаков: {feature_cols}")

    print("Кросс-валидация (5-fold GroupKFold)...")
    cross_validate(calib_features, n_folds=5)

    print("Обучение финальной модели на всех calibration-данных...")
    model = CatBoostRanker(**LTR_PARAMS)
    model.fit(Pool(
        calib_features[feature_cols].values,
        label=calib_features["target"].values,
        group_id=calib_features["query_id"].values,
    ))

    print("Построение признаков для test и предсказание...")
    test_queries = queries[queries["query_id"] > 500].copy()
    test_features = _build_features_dense(
        bm25_test, articles, test_queries,
        query_emb, article_emb, article_map,
    )

    print("Сохранение результата в answer.csv...")
    with open(READY_DATA_DIR / "test_ids_mapping.json") as f:
        id_map = json.load(f)

    rows = []
    for qid in sorted(test_features["query_id"].unique()):
        group = test_features[test_features["query_id"] == qid].copy()
        X = group[feature_cols].values
        group["ltr_score"] = model.predict(X)
        group = group.sort_values("ltr_score", ascending=False).head(TOP_K_FINAL)
        orig_qid = id_map.get(str(int(qid)), qid)
        answer_str = " ".join(str(a) for a in group["article_id"].tolist())
        rows.append({"query_id": int(orig_qid), "answer": answer_str})

    result_df = pd.DataFrame(rows).sort_values("query_id")
    result_df.to_csv(READY_DATA_DIR / "answer.csv", index=False)
    print(f"Сохранён answer.csv: {len(result_df)} запросов")


if __name__ == "__main__":
    main()
