"""
MODULE: TEXT PREPROCESSING
=============================
Module này chịu trách nhiệm làm sạch và chuẩn hóa văn bản email trước khi đưa vào mô hình ML.

WORKFLOW CHÍNH:
1. Loại bỏ HTML tags và entities
2. Chuyển về chữ thường (lowercase)
3. Loại bỏ ký tự đặc biệt, chỉ giữ lại chữ cái
4. Tokenization (tách thành các từ riêng lẻ)
5. Loại bỏ stopwords (từ phổ biến không mang nhiều ý nghĩa)
6. Lemmatization (đưa từ về dạng gốc: running -> run, better -> good)

NGUYÊN LÝ HOẠT ĐỘNG:
- Text cleaning đồng nhất giúp mô hình học patterns chính xác hơn
- Lemmatization giảm kích thước vocabulary và gom nhóm các biến thể của từ
- Stopwords removal giảm noise và tập trung vào từ có giá trị phân biệt
- Sử dụng caching (@lru_cache) để tối ưu performance khi xử lý nhiều email
"""

from __future__ import annotations

import re
from functools import lru_cache
from html import unescape
from typing import Iterable

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def _ensure_resource(resource: str, download_name: str) -> None:
    """
    Đảm bảo NLTK resource đã được download trước khi sử dụng.
    
    Tự động download nếu chưa có, tránh lỗi khi chạy lần đầu.
    Args:
        resource: Đường dẫn tới resource cần kiểm tra (vd: 'corpora/stopwords')
        download_name: Tên package cần download (vd: 'stopwords')
    """
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(download_name, quiet=True)


# Auto-download các NLTK resources cần thiết khi import module
_ensure_resource('corpora/stopwords', 'stopwords')
_ensure_resource('corpora/wordnet', 'wordnet')

# STOP_WORDS: Tập hợp các từ phổ biến không mang nhiều ý nghĩa phân loại
# (the, is, at, which, on, etc.) sẽ bị loại bỏ để giảm noise
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update({'\n', '\r', '\t'})  # Thêm các ký tự whitespace

try:
    # WordNetLemmatizer: Chuyển từ về dạng gốc (running->run, better->good)
    # Giúp gom nhóm các biến thể của cùng một từ
    _lemmatizer = WordNetLemmatizer()
except LookupError:  # fallback if loading fails
    _lemmatizer = None


@lru_cache(maxsize=128)
def _compiled_regex(pattern: str) -> re.Pattern[str]:
    """
    Cache compiled regex patterns để tăng performance.
    
    Thay vì compile regex mỗi lần gọi, ta cache lại để tái sử dụng.
    LRU cache size=128 đủ lớn cho các patterns thường dùng.
    """
    return re.compile(pattern, flags=re.UNICODE)


def _tokenize(text: str) -> Iterable[str]:
    """
    Tokenize và normalize văn bản thành các token sạch.
    
    WORKFLOW:
    1. Lowercase: Chuẩn hóa về chữ thường (Email -> email)
    2. Remove non-alpha: Loại ký tự đặc biệt, chỉ giữ chữ cái và khoảng trắng
    3. Split: Tách thành các từ riêng lẻ
    4. Filter: Loại bỏ từ ngắn (<=2 ký tự) và stopwords
    5. Lemmatize: Đưa từ về dạng gốc
    
    Args:
        text: Văn bản đầu vào đã loại HTML
        
    Yields:
        Các token đã được làm sạch và lemmatize
    """
    # Lightweight normalisation keeps the downstream vocabulary tidy and comparable.
    cleaned = text.lower()
    # Thay thế mọi ký tự không phải chữ cái bằng space
    cleaned = _compiled_regex(r"[^a-z\s]").sub(" ", cleaned)
    tokens = cleaned.split()
    
    for token in tokens:
        # Bỏ qua token quá ngắn (ít giá trị phân biệt)
        if len(token) <= 2:
            continue
        # Bỏ qua stopwords (the, is, at, etc.)
        if token in STOP_WORDS:
            continue
        # Lemmatize để gom nhóm các biến thể (running->run, better->good)
        if _lemmatizer is not None:
            yield _lemmatizer.lemmatize(token)
        else:
            yield token


def clean_email_text(raw_text: str | None) -> str:
    """
    Hàm chính để làm sạch văn bản email.
    
    WORKFLOW ĐẦY ĐỦ:
    1. Kiểm tra input rỗng
    2. Unescape HTML entities (&amp; -> &, &lt; -> <, etc.)
    3. Strip HTML tags (<div>, <p>, <a>, etc.)
    4. Normalize whitespace (nhiều space -> 1 space)
    5. Tokenize và filter (qua _tokenize)
    6. Join lại thành chuỗi sạch
    
    NGUYÊN LÝ:
    - HTML entities và tags thường xuất hiện trong spam/phishing emails
    - Việc loại bỏ chúng giúp mô hình tập trung vào nội dung thực
    - Tokenization và lemmatization chuẩn hóa vocabulary
    
    Args:
        raw_text: Văn bản email gốc (có thể chứa HTML)
        
    Returns:
        Văn bản đã làm sạch, chỉ chứa các token có giá trị
        Trả về chuỗi rỗng nếu input là None hoặc sau khi clean không còn gì
    """
    if not raw_text:
        return ''

    # Unescape HTML entities (&amp; -> &, &nbsp; -> space, etc.)
    text = unescape(raw_text)
    # Strip tất cả HTML tags (<div>, <p>, <script>, etc.)
    text = _compiled_regex(r'<[^>]+>').sub(' ', text)
    # Normalize nhiều space/tab/newline thành 1 space
    text = _compiled_regex(r'\s+').sub(' ', text)
    
    # Tokenize, filter, và lemmatize
    tokens = list(_tokenize(text))
    
    # Join lại thành chuỗi với space separator
    return ' '.join(tokens)
