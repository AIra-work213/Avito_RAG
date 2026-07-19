"""Финальное обучение CatBoost с отдельными dense-признаками заголовка и тела.

Добавляет к BM25-признакам:
  - dense_title_score — косинус похожести запроса с заголовком статьи.
  - dense_body_score — косинус похожести запроса с телом статьи.

Пайплайн: BM25 → title/body эмбеддинги e5-small → CatBoost → answer.csv.
"""

import json

import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool

from src.config import READY_DATA_DIR, LTR_PARAMS, TOP_K_FINAL
from src.retrieval.bm25 import load_retrieval_results
from src.embeddings.embedder import build_embeddings, build_title_body_embeddings


def _build_features(
    bm25_results: dict,
    articles_df: pd.DataFrame,
    queries_df: pd.DataFrame,
    query_emb: np.ndarray,
    title_emb: np.ndarray,
    body_emb: np.ndarray,
    article_map: dict,
) -> pd.DataFrame:
    """Строит признаки с отдельными dense_score для заголовка и тела.

    Для каждого BM25-кандидата вычисляет два косинуса:
      - между эмбеддингом запроса и эмбеддингом заголовка.
      - между эмбеддингом запроса и эмбеддингом тела.

    Аргументы:
        bm25_results: {query_id: {article_ids, scores, ...}}.
        articles_df: датасет статей.
        queries_df: датасет запросов.
        query_emb: (1000, 384).
        title_emb: (793, 384).
        body_emb: (793, 384).
        article_map: {article_id: {"idx": int}}.

    Возвращает:
        DataFrame с колонками query_id, article_id, признаками.
    """
    article_map_full = articles_df.set_index("article_id")[["article_len"]].to_dict("index")

    query_len_map = {}
    for _, row in queries_df.iterrows():
        query_len_map[int(row["query_id"])] = len(str(row.get("query_lemma", "")))

    query_norm = query_emb / (np.linalg.norm(query_emb, axis=1, keepdims=True) + 1e-9)
    title_norm = title_emb / (np.linalg.norm(title_emb, axis=1, keepdims=True) + 1e-9)
    body_norm = body_emb / (np.linalg.norm(body_emb, axis=1, keepdims=True) + 1e-9)

    sim_title = query_norm @ title_norm.T
    sim_body = query_norm @ body_norm.T
    n_articles = sim_title.shape[1]

    rows = []
    for qid, pred in bm25_results.items():
        qidx = qid - 1
        for i, aid in enumerate(pred["article_ids"]):
            alen = article_map_full.get(aid, {}).get("article_len", 0)
            qlen = query_len_map.get(qid, 1)
            aidx = article_map.get(aid, {}).get("idx", 0)

            if aidx < n_articles:
                d_title = float(sim_title[qidx, aidx])
                d_body = float(sim_body[qidx, aidx])
            else:
                d_title = 0.0
                d_body = 0.0

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
                "dense_title_score": d_title,
                "dense_body_score": d_body,
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


def main() -> None:
    """Основная функция: обучение CatBoost с title/body dense-признаками.

    1. Загружает BM25-результаты, статьи, запросы.
    2. Загружает эмбеддинги запросов, заголовков и тел статей.
    3. Вычисляет dense_title_score и dense_body_score.
    4. Обучает CatBoost на всех calibration-данных.
    5. Сохраняет answer.csv.
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

    print("Загрузка эмбеддингов...")
    _, query_emb = build_embeddings(force=False)

    print("Загрузка title/body эмбеддингов...")
    title_emb, body_emb = build_title_body_embeddings(force=False)

    print("Построение карты статей...")
    article_map = {}
    for idx, aid in enumerate(articles["article_id"]):
        article_map[int(aid)] = {"idx": idx}

    print("Построение признаков для calibration...")
    calib_queries = queries[queries["query_id"] <= 500].copy()
    calib_features = _build_features(
        bm25_calib, articles, calib_queries,
        query_emb, title_emb, body_emb, article_map,
    )
    calib_features = _add_targets(calib_features, relevance)
    feature_cols = _get_feature_cols(calib_features)
    print(f"  {len(calib_features)} строк, "
          f"{calib_features['query_id'].nunique()} запросов, "
          f"{len(feature_cols)} признаков: {feature_cols}")

    print("Обучение финальной модели на всех calibration-данных...")
    model = CatBoostRanker(**LTR_PARAMS)
    model.fit(Pool(
        calib_features[feature_cols].values,
        label=calib_features["target"].values,
        group_id=calib_features["query_id"].values,
    ))

    print("Предсказание для test-запросов...")
    test_queries = queries[queries["query_id"] > 500].copy()
    test_features = _build_features(
        bm25_test, articles, test_queries,
        query_emb, title_emb, body_emb, article_map,
    )

    print("Сохранение answer.csv...")
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
