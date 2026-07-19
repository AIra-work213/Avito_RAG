"""Кодирование текстов в эмбеддинги с помощью BERT-модели.

Используется intfloat/multilingual-e5-small с INT8-квантизацией
для экономии памяти на CPU. Кэширует эмбеддинги на диск.
"""

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


MODEL_NAME = "intfloat/multilingual-e5-small"
MAX_LENGTH = 512
BATCH_SIZE = 16
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


class E5Embedder:
    """Кодировщик текстов на основе e5-small с INT8-квантизацией.

    При инициализации загружает модель и токенизатор,
    применяет динамическую INT8-квантизацию для Linear-слоёв.

    Аргументы:
        device: устройство для вычислений ('cpu' или 'cuda').
        verbose: печатать ли прогресс.
    """

    def __init__(self, device: str = "cpu", verbose: bool = True):
        self.device = device
        self.verbose = verbose

        if verbose:
            print(f"Загрузка токенизатора {MODEL_NAME}...")
        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

        if verbose:
            print(f"Загрузка модели {MODEL_NAME}...")
        model = AutoModel.from_pretrained(MODEL_NAME)
        model.eval()

        if device == "cpu":
            if verbose:
                print("Применение INT8-квантизации...")
            model = torch.quantization.quantize_dynamic(
                model, {torch.nn.Linear}, dtype=torch.qint8,
            )

        self.model = model.to(device)

    def _mean_pooling(self, token_embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Усредняет эмбеддинги токенов с учётом маски.

        Аргументы:
            token_embeddings: [batch, seq_len, hidden].
            attention_mask: [batch, seq_len].

        Возвращает:
            [batch, hidden] — усреднённые эмбеддинги.
        """
        mask = attention_mask.unsqueeze(-1).float()
        return (token_embeddings * mask).sum(1) / mask.sum(1).clamp(min=1e-9)

    def encode(
        self,
        texts: list[str],
        prefix: str = "",
        batch_size: int = BATCH_SIZE,
    ) -> np.ndarray:
        """Кодирует список текстов в эмбеддинги.

        Аргументы:
            texts: список строк для кодирования.
            prefix: префикс для каждого текста (query:/passage:).
            batch_size: размер батча.

        Возвращает:
            numpy array формы (len(texts), hidden_dim).
        """
        all_embeddings = []
        prefixed = [prefix + t for t in texts]

        for i in tqdm(range(0, len(prefixed), batch_size), desc="Кодирование", disable=not self.verbose):
            batch = prefixed[i:i + batch_size]
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            embeddings = self._mean_pooling(outputs.last_hidden_state, inputs["attention_mask"])
            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

        return np.concatenate(all_embeddings, axis=0).astype(np.float32)


def get_article_texts() -> list[str]:
    """Загружает и склеивает заголовок + тело каждой статьи.

    Возвращает:
        список строк вида "заголовок тело" для каждой статьи.
    """
    import pandas as pd
    from src.config import ARTICLES_PROCESSED_PATH
    df = pd.read_feather(ARTICLES_PROCESSED_PATH)
    return (df["title_lemma"].fillna("") + " " + df["body_lemma"].fillna("")).tolist()


def get_query_texts() -> list[str]:
    """Загружает лемматизированные запросы.

    Возвращает:
        список строк запросов (query_lemma).
    """
    import pandas as pd
    from src.config import QUERIES_RAW_PATH
    df = pd.read_feather(QUERIES_RAW_PATH)
    return df["query_lemma"].fillna("").tolist()


def get_article_title_body() -> tuple[list[str], list[str]]:
    """Загружает заголовки и тела статей отдельными списками.

    Возвращает:
        кортеж (title_texts, body_texts), каждый — список строк.
    """
    import pandas as pd
    from src.config import ARTICLES_PROCESSED_PATH
    df = pd.read_feather(ARTICLES_PROCESSED_PATH)
    return (
        df["title_lemma"].fillna("").tolist(),
        df["body_lemma"].fillna("").tolist(),
    )


def build_title_body_embeddings(
    title_path=None,
    body_path=None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит отдельные эмбеддинги для заголовков и тел статей.

    Результат кэшируется на диск.

    Аргументы:
        title_path: путь для сохранения title-эмбеддингов.
        body_path: путь для сохранения body-эмбеддингов.
        force: перестроить даже если кэш есть.

    Возвращает:
        кортеж (title_embeddings, body_embeddings).
    """
    import os
    from src.config import READY_DATA_DIR

    if title_path is None:
        title_path = READY_DATA_DIR / "title_embeddings_e5.npy"
    if body_path is None:
        body_path = READY_DATA_DIR / "body_embeddings_e5.npy"

    if not force and os.path.exists(title_path) and os.path.exists(body_path):
        print("Загрузка title/body эмбеддингов из кэша...")
        return np.load(title_path), np.load(body_path)

    embedder = E5Embedder()
    titles, bodies = get_article_title_body()

    print("Кодирование заголовков...")
    title_emb = embedder.encode(titles, prefix=PASSAGE_PREFIX)
    np.save(title_path, title_emb)

    print("Кодирование тел статей...")
    body_emb = embedder.encode(bodies, prefix=PASSAGE_PREFIX)
    np.save(body_path, body_emb)

    return title_emb, body_emb


def build_embeddings(
    article_path=None,
    query_path=None,
    force: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Строит эмбеддинги статей и запросов.

    Результат кэшируется на диск.

    Аргументы:
        article_path: путь для сохранения эмбеддингов статей.
        query_path: путь для сохранения эмбеддингов запросов.
        force: перестроить даже если кэш есть.

    Возвращает:
        кортеж (article_embeddings, query_embeddings), каждый np.ndarray.
    """
    import os
    from src.config import READY_DATA_DIR

    if article_path is None:
        article_path = READY_DATA_DIR / "article_embeddings_e5.npy"
    if query_path is None:
        query_path = READY_DATA_DIR / "query_embeddings_e5.npy"

    if not force and os.path.exists(article_path) and os.path.exists(query_path):
        print("Загрузка эмбеддингов из кэша...")
        return np.load(article_path), np.load(query_path)

    embedder = E5Embedder()

    print("Кодирование статей...")
    article_texts = get_article_texts()
    article_emb = embedder.encode(article_texts, prefix=PASSAGE_PREFIX)
    np.save(article_path, article_emb)

    print("Кодирование запросов...")
    query_texts = get_query_texts()
    query_emb = embedder.encode(query_texts, prefix=QUERY_PREFIX)
    np.save(query_path, query_emb)

    return article_emb, query_emb
