"""
MODULE: EXTRACT EMAIL (DATASET ASSEMBLY)
=========================================
Module này chịu trách nhiệm xây dựng dataset tổng hợp từ nhiều nguồn email khác nhau.

WORKFLOW CHÍNH:
1. Load các dataset từ nhiều nguồn CSV (Enron, CEAS, Ling-Spam, phishing, fraud, etc.)
2. Chuẩn hóa schema (subject, body, label) về format thống nhất
3. Map labels từ các nguồn khác nhau về 3 class: Normal(0), Spam(1), Fraud(2)
4. Apply text cleaning (qua text_preprocessing.clean_email_text)
5. Loại bỏ duplicates (dựa trên clean_text)
6. Optional: Balance classes (downsample về class nhỏ nhất)
7. Export ra combined_emails.csv và các file theo từng label

NGUYÊN LÝ HOẠT ĐỘNG:
- Multi-source dataset giúp mô hình generalize tốt hơn
- Label mapping thống nhất giúp train 1 mô hình multi-class duy nhất
- Deduplication tránh overfitting trên các email lặp lại
- Class balancing giúp tránh bias về class chiếm đa số
- Mỗi dataset có schema khác nhau -> cần DatasetSpec để config riêng

LABEL MAPPING:
- 0: Normal (email thông thường, không spam/fraud)
- 1: Spam (email quảng cáo, rác)
- 2: Fraud (email lừa đảo, phishing, Nigerian scam)
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from text_preprocessing import clean_email_text

# ============================================================================
# CONFIGURATION: Paths và Constants
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'Datasets'
COMBINED_DATASET_PATH = DATASETS_DIR / 'combined_emails.csv'

# Output paths cho từng label class
LABELLED_OUTPUTS = {
    0: DATASETS_DIR / 'normal_emails.csv',
    1: DATASETS_DIR / 'spam_emails.csv',
    2: DATASETS_DIR / 'fraud_emails.csv',
}

# Mapping từ số sang tên label (cho readability)
LABEL_NAMES = {0: 'Normal', 1: 'Spam', 2: 'Fraud'}


@dataclass(frozen=True)
class DatasetSpec:
    """
    Specification cho mỗi dataset source.
    
    Mỗi dataset có thể có:
    - Schema khác nhau (column names khác nhau)
    - Label encoding khác nhau (0/1 hoặc 1/2, etc.)
    - Text ở columns khác nhau (subject+body hoặc text_combined)
    
    DatasetSpec cho phép config linh hoạt cho từng source.
    """
    filename: str  # Tên file CSV trong Datasets/
    label_map: Optional[dict] = None  # Mapping từ original label -> target label (0/1/2)
    default_label: Optional[int] = None  # Label mặc định nếu không có column 'label'
    text_column: Optional[str] = None  # Column chứa text (nếu không dùng subject+body)


# ============================================================================
# DATASET SOURCES CONFIGURATION
# ============================================================================
# Danh sách tất cả các nguồn dataset và cách map labels của chúng
DATASET_SOURCES: List[DatasetSpec] = [
    # Enron, Ling, SpamAssassin, CEAS: có label 0=normal, 1=spam
    DatasetSpec('Enron.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('Ling.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('SpamAssasin.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('CEAS_08.csv', label_map={0: 0, 1: 1}),
    
    # Phishing dataset: có label 0=normal, 1=fraud và text ở column 'text_combined'
    DatasetSpec('phishing_email.csv', label_map={0: 0, 1: 2}, text_column='text_combined'),
    
    # Nazario, Nigerian Fraud: toàn bộ là fraud (label=2)
    DatasetSpec('Nazario.csv', label_map={1: 2}, default_label=2),
    DatasetSpec('Nigerian_Fraud.csv', label_map={1: 2}, default_label=2),
]


def _read_safe(df: pd.DataFrame, column: str, length: int) -> pd.Series:
    """
    Đọc column một cách an toàn, trả về empty string nếu column không tồn tại.
    
    Tránh KeyError khi dataset thiếu column subject hoặc body.
    Args:
        df: DataFrame cần đọc
        column: Tên column cần đọc
        length: Số rows để tạo Series rỗng nếu column không tồn tại
    Returns:
        Series với giá trị từ column, hoặc Series rỗng nếu column không có
    """
    if column in df.columns:
        return df[column].fillna('').astype(str)
    return pd.Series([''] * length, dtype='object')


def load_source_dataset(spec: DatasetSpec) -> pd.DataFrame:
    """
    Load và chuẩn hóa một dataset source theo specification.
    
    WORKFLOW:
    1. Đọc CSV file (encoding latin-1 để support các ký tự đặc biệt)
    2. Extract subject và body (safe read, trả về '' nếu không có)
    3. Combine text từ subject+body hoặc từ text_column riêng
    4. Map original labels sang target labels (0/1/2)
    5. Validate không có unmapped labels
    6. Apply clean_email_text để chuẩn hóa văn bản
    7. Filter bỏ rows có text/clean_text rỗng
    
    Args:
        spec: DatasetSpec chứa config cho dataset này
        
    Returns:
        DataFrame với schema chuẩn:
        - subject: Tiêu đề email
        - body: Nội dung email
        - text: Full text (subject + body)
        - clean_text: Text đã làm sạch (qua clean_email_text)
        - original_label: Label gốc từ source
        - label: Label đã map (0/1/2)
        - label_name: Tên label (Normal/Spam/Fraud)
        - source: Tên file nguồn
    """
    path = DATASETS_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(f'Missing dataset: {path}')

    # Đọc CSV với encoding latin-1 (support ký tự đặc biệt trong email)
    df = pd.read_csv(path, encoding='latin-1')
    row_count = len(df.index)

    # Safe read: trả về empty string nếu column không tồn tại
    subject = _read_safe(df, 'subject', row_count)
    body = _read_safe(df, 'body', row_count)

    # Lấy text từ column riêng (nếu có) hoặc combine subject+body
    if spec.text_column and spec.text_column in df.columns:
        text = df[spec.text_column].fillna('').astype(str)
    else:
        text = (subject + ' ' + body).str.strip()

    # Extract original label từ dataset
    if 'label' in df.columns:
        original_label = df['label']
    else:
        # Nếu không có column label, dùng default_label
        original_label = pd.Series([spec.default_label] * row_count)

    # Map original label sang target label (0/1/2)
    if spec.label_map:
        mapped_label = original_label.map(spec.label_map)
    else:
        mapped_label = pd.Series([spec.default_label] * row_count)

    # Fill NaN bằng default_label (nếu có)
    if spec.default_label is not None:
        mapped_label = mapped_label.fillna(spec.default_label)

    # Validate: không được có unmapped labels (NaN)
    if mapped_label.isnull().any():
        missing = sorted(original_label[mapped_label.isnull()].unique())
        raise ValueError(
            f'Encountered unmapped label(s) {missing} in dataset {spec.filename}. '
            'Update the label_map/default_label configuration.'
        )

    mapped_label = mapped_label.astype(int)

    # Tạo DataFrame với schema chuẩn
    prepared = pd.DataFrame(
        {
            'subject': subject,
            'body': body,
            'text': text.str.strip(),
            'original_label': original_label,
            'label': mapped_label,
            'label_name': mapped_label.map(LABEL_NAMES),
            'source': spec.filename,
        }
    )

    # Filter bỏ rows có text rỗng
    prepared = prepared[prepared['text'].str.len() > 0]
    
    # Apply the shared cleaning routine so every dataset feeds identical tokens downstream.
    # Clean text để chuẩn hóa: lowercase, loại HTML, lemmatize, remove stopwords
    prepared['clean_text'] = prepared['text'].map(clean_email_text)
    
    # Filter bỏ rows có clean_text rỗng (sau khi clean không còn gì)
    prepared = prepared[prepared['clean_text'].str.len() > 0]
    
    return prepared


def load_wordcount_dataset(path: Path = DATASETS_DIR / 'emails.csv') -> pd.DataFrame:
    """
    Load dataset kiểu word-count style (mỗi column là term frequency của 1 từ).
    
    ĐỊNH DẠNG:
    - Column "Email No.": ID của email
    - 3000 columns: mỗi column là 1 từ, giá trị là term frequency
    - Column "Prediction": label (1=spam/fraud, 0=normal)
    
    WORKFLOW:
    1. Đọc CSV
    2. Extract word counts từ mỗi row
    3. Reconstruct text bằng cách repeat mỗi token theo frequency
       Ví dụ: {"money": 3, "free": 2} -> "money money money free free"
    4. Convert sang format chuẩn như các dataset khác
    
    NGUYÊN LÝ:
    - Dataset này có format khác biệt (term frequency thay vì raw text)
    - Ta reconstruct lại text để có thể dùng chung pipeline với các dataset khác
    - Reconstructed text vẫn giữ được thông tin về term frequency
    
    Args:
        path: Đường dẫn tới file emails.csv
        
    Returns:
        DataFrame với schema chuẩn (giống load_source_dataset)
        Trả về empty DataFrame nếu file không tồn tại hoặc format sai
    """
    if not path.exists():
        return pd.DataFrame(
            columns=['subject', 'body', 'text', 'clean_text', 'original_label', 'label', 'label_name', 'source']
        )

    df = pd.read_csv(path)
    if 'Prediction' not in df.columns:
        raise ValueError(f'Column "Prediction" not found in {path}')

    # Lấy danh sách các word columns (tất cả columns trừ Email No. và Prediction)
    word_columns = [col for col in df.columns if col not in ('Email No.', 'Prediction')]
    if not word_columns:
        return pd.DataFrame(
            columns=['subject', 'body', 'text', 'clean_text', 'original_label', 'label', 'label_name', 'source']
        )

    word_array = np.array(word_columns)
    records: List[dict] = []

    for _, row in df.iterrows():
        # Extract term frequencies cho row này
        counts = row[word_columns].fillna(0).to_numpy(dtype=float)
        counts = np.maximum(counts, 0)  # Đảm bảo không âm
        counts = counts.astype(int)
        
        # Filter chỉ giữ các từ có count > 0
        mask = counts > 0
        if not mask.any():
            continue
            
        # Recreate a pseudo-email by repeating each token by its observed frequency.
        # Ví dụ: money có count=3 -> ["money", "money", "money"]
        tokens = np.repeat(word_array[mask], counts[mask])
        if tokens.size == 0:
            continue
            
        text = ' '.join(tokens.tolist())
        if not text:
            continue
            
        label = int(row['Prediction'])
        
        # Tạo record với schema chuẩn
        record = {
            'subject': str(row.get('Email No.', '')),
            'body': '',
            'text': text,
            'clean_text': text,  # Text đã là word tokens, không cần clean thêm
            'original_label': label,
            'label': label,
            'label_name': LABEL_NAMES.get(label, 'Unknown'),
            'source': path.name,
        }
        records.append(record)

    if not records:
        return pd.DataFrame(
            columns=['subject', 'body', 'text', 'clean_text', 'original_label', 'label', 'label_name', 'source']
        )

    return pd.DataFrame.from_records(records)


def build_combined_dataset(balance: bool = False, random_state: int = 42) -> pd.DataFrame:
    """
    Xây dựng dataset tổng hợp từ tất cả sources.
    
    WORKFLOW:
    1. Load tất cả các source datasets (qua DATASET_SOURCES)
    2. Load word-count dataset (nếu có)
    3. Concat tất cả thành 1 DataFrame lớn
    4. Loại bỏ duplicates dựa trên clean_text (tránh overfitting)
    5. Optional: Balance classes bằng downsampling
    
    BALANCE CLASSES:
    - Nếu balance=True: Downsample mỗi class về size của class nhỏ nhất
    - Ví dụ: Normal=64k, Spam=35k, Fraud=34k -> downsample tất cả về 34k
    - Giúp tránh bias về class đa số, model học balanced hơn
    
    Args:
        balance: Có downsample để balance classes hay không
        random_state: Seed cho random sampling (reproducibility)
        
    Returns:
        DataFrame tổng hợp từ tất cả sources, đã deduplicate và optionally balanced
    """
    # Load each labelled CSV and the word-count reconstruction so we can merge them.
    frames: List[pd.DataFrame] = [load_source_dataset(spec) for spec in DATASET_SOURCES]
    
    # Thêm word-count dataset nếu có
    wordcount_df = load_wordcount_dataset()
    if not wordcount_df.empty:
        frames.append(wordcount_df)
        
    # Concat tất cả thành 1 DataFrame lớn
    combined = pd.concat(frames, ignore_index=True)
    
    # Remove duplicate emails that might appear across corpora.
    # Dùng clean_text để detect duplicates (sau khi normalize)
    combined.drop_duplicates(subset=['clean_text'], inplace=True)

    if balance:
        # Đếm số lượng mỗi class
        class_counts = combined['label'].value_counts()
        min_count = class_counts.min()  # Class nhỏ nhất
        
        balanced_frames = []
        # Downsample mỗi class về min_count
        for label_value, group in combined.groupby('label'):
            sample_size = min(len(group), min_count)
            balanced_frames.append(group.sample(n=sample_size, random_state=random_state))
        
        # Concat lại các class đã balanced
        combined = pd.concat(balanced_frames, ignore_index=True)

    return combined


def write_labelled_datasets(dataset: pd.DataFrame) -> None:
    """
    Export dataset thành 3 files riêng theo label.
    
    Output:
    - normal_emails.csv: Chỉ chứa emails có label=0
    - spam_emails.csv: Chỉ chứa emails có label=1
    - fraud_emails.csv: Chỉ chứa emails có label=2
    
    Giúp dễ dàng analyze và inspect từng loại email riêng biệt.
    
    Args:
        dataset: DataFrame tổng hợp cần split
    """
    for label, output_path in LABELLED_OUTPUTS.items():
        subset = dataset[dataset['label'] == label]
        subset.to_csv(output_path, index=False, encoding='utf-8')


def summarize(dataset: pd.DataFrame) -> None:
    """
    In ra thống kê tổng quan về dataset.
    
    Show:
    - Số lượng emails của từng class (Normal/Spam/Fraud)
    - Tổng số rows
    """
    print('Dataset summary:')
    counts = dataset['label'].value_counts().sort_index()
    for label, count in counts.items():
        print(f'  {label} ({LABEL_NAMES.get(label, "Unknown")}): {count}')
    print(f'Total rows: {len(dataset)}')


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    Arguments:
    --balance: Downsample classes để cân bằng
    --output: Đường dẫn output file (default: combined_emails.csv)
    """
    parser = argparse.ArgumentParser(
        description='Combine labelled email CSVs into a unified dataset and export class-specific files.'
    )
    parser.add_argument(
        '--balance',
        action='store_true',
        help='Down-sample each class to the smallest class size.',
    )
    parser.add_argument(
        '--output',
        default=str(COMBINED_DATASET_PATH),
        help='Path for the combined dataset CSV (default: %(default)s).',
    )
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> Path:
    """
    Main function: Build và export combined dataset.
    
    WORKFLOW:
    1. Parse arguments
    2. Build combined dataset (optionally balanced)
    3. Save to combined_emails.csv
    4. Export các file theo label (normal/spam/fraud_emails.csv)
    5. Print summary statistics
    
    Returns:
        Path tới combined dataset file
    """
    args = parse_args(argv)
    
    # Build dataset tổng hợp
    combined = build_combined_dataset(balance=args.balance)

    # Resolve output path
    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = DATASETS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save combined dataset
    combined.to_csv(output_path, index=False, encoding='utf-8')
    
    # Export files theo label
    write_labelled_datasets(combined)
    
    # Print summary
    summarize(combined)

    print(f'Combined dataset saved to {output_path.resolve()}')
    for label, path in LABELLED_OUTPUTS.items():
        print(f'  Label {label} -> {path.resolve()}')

    return output_path


if __name__ == '__main__':
    main()
