"""Препроцессинг статей справки Авито.

Конвертирует HTML-тело статьи в Markdown с линеаризацией таблиц,
затем лемматизирует заголовок и тело с помощью spaCy.
"""

import html2text
import pandas as pd
import spacy
from tqdm import tqdm

from src.config import (
    ARTICLES_PATH,
    ARTICLES_PROCESSED_PATH,
    SPACY_MODEL_NAME,
)


def _init_converter() -> html2text.HTML2Text:
    """Создаёт и настраивает конвертер HTML в Markdown.

    Возвращает:
        html2text.HTML2Text: настроенный конвертер с линеаризацией таблиц.
    """
    conv = html2text.HTML2Text()
    conv.body_width = 0
    conv.ignore_links = False
    conv.ignore_images = False
    conv.ignore_emphasis = False
    conv.protect_links = False
    conv.unicode_snob = True
    conv.escape_snob = False
    conv.single_line_break = False
    conv.mark_code = False
    conv.wrap_links = False
    conv.wrap_list_items = False
    conv.pad_tables = True
    conv.convert_internal_links = False
    return conv


def _init_spacy():
    """Загружает модель spaCy для русского языка с отключёнными NER и parser.

    Отключение ненужных компонентов ускоряет лемматизацию.
    """
    nlp = spacy.load(SPACY_MODEL_NAME, disable=["ner", "parser", "textcat"])
    nlp.max_length = 5_000_000
    return nlp


def html_to_markdown(html_text: str, converter: html2text.HTML2Text) -> str:
    """Конвертирует HTML в Markdown с обработкой ошибок.

    При возникновении ошибки (например, в кривой таблице) выполняется
    падение на простое удаление HTML-тегов.

    Аргументы:
        html_text: исходный HTML-текст статьи.
        converter: настроенный конвертер html2text.

    Возвращает:
        текст в формате Markdown.
    """
    try:
        md = converter.handle(html_text)
    except Exception:
        import re
        md = re.sub(r"<[^>]+>", " ", html_text)
        md = re.sub(r"\s+", " ", md).strip()
    lines = [line.strip() for line in md.splitlines()]
    return "\n".join(line for line in lines if line)


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


def process_articles():
    """Загружает статьи, конвертирует HTML в Markdown и лемматизирует.

    Результат сохраняется в ARTICLES_PROCESSED_PATH в формате Feather.
    """
    print("Загрузка статей...")
    articles = pd.read_feather(ARTICLES_PATH)

    print("Инициализация конвертера HTML в Markdown...")
    converter = _init_converter()

    print("Конвертация HTML в Markdown...")
    tqdm.pandas(desc="HTML→MD")
    articles["body_md"] = articles["body"].progress_apply(
        lambda x: html_to_markdown(str(x), converter)
    )

    print("Загрузка spaCy...")
    nlp = _init_spacy()

    print("Лемматизация заголовков...")
    tqdm.pandas(desc="Лемматизация заголовков")
    articles["title_lemma"] = articles["title"].progress_apply(
        lambda x: lemmatize_text(str(x), nlp)
    )

    print("Лемматизация тел статей...")
    tqdm.pandas(desc="Лемматизация тел")
    articles["body_lemma"] = articles["body_md"].progress_apply(
        lambda x: lemmatize_text(str(x), nlp)
    )

    keep = ["article_id", "title", "body_md", "title_lemma", "body_lemma"]
    result = articles[keep].copy()
    result.rename(columns={"body_md": "body"}, inplace=True)

    print(f"Сохранение обработанных статей в {ARTICLES_PROCESSED_PATH}...")
    result.to_feather(ARTICLES_PROCESSED_PATH)
    print(f"Готово. Обработано {len(result)} статей.")
    return result


if __name__ == "__main__":
    process_articles()
