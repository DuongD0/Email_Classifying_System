from __future__ import annotations

import re
from functools import lru_cache
from html import unescape
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_resource(resource: str, download_name: str) -> None:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name, quiet=True)


_ensure_resource('corpora/stopwords', 'stopwords')
_ensure_resource('corpora/wordnet', 'wordnet')

STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({'\n', '\r', '\t'})

try:
    _lemmatizer = WordNetLemmatizer()
except LookupError:  # fallback if loading fails
    _lemmatizer = None


@lru_cache(maxsize=128)
def _compiled_regex(pattern: str) -> re.Pattern[str]:
    return re.compile(pattern, flags=re.UNICODE)


def _tokenize(text: str) -> Iterable[str]:
    # Lightweight normalisation keeps the downstream vocabulary tidy and comparable.
    cleaned = text.lower()
    cleaned = _compiled_regex(r"[^a-z\s]").sub(" ", cleaned)
    tokens = cleaned.split()
    for token in tokens:
        if len(token) <= 2:
            continue
        if token in STOP_WORDS:
            continue
        if _lemmatizer is not None:
            yield _lemmatizer.lemmatize(token)
        else:
            yield token


def clean_email_text(raw_text: str | None) -> str:
    if not raw_text:
        return ''

    text = unescape(raw_text)
    text = _compiled_regex(r'<[^>]+>').sub(' ', text)
    text = _compiled_regex(r'\s+').sub(' ', text)
    tokens = list(_tokenize(text))
    return ' '.join(tokens)
