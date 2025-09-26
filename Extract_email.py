from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from text_preprocessing import clean_email_text
BASE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BASE_DIR / 'Datasets'
COMBINED_DATASET_PATH = DATASETS_DIR / 'combined_emails.csv'
LABELLED_OUTPUTS = {
    0: DATASETS_DIR / 'normal_emails.csv',
    1: DATASETS_DIR / 'spam_emails.csv',
    2: DATASETS_DIR / 'fraud_emails.csv',
}

LABEL_NAMES = {0: 'Normal', 1: 'Spam', 2: 'Fraud'}


@dataclass(frozen=True)
class DatasetSpec:
    filename: str
    label_map: Optional[dict] = None
    default_label: Optional[int] = None
    text_column: Optional[str] = None


DATASET_SOURCES: List[DatasetSpec] = [
    DatasetSpec('Enron.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('Ling.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('SpamAssasin.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('CEAS_08.csv', label_map={0: 0, 1: 1}),
    DatasetSpec('phishing_email.csv', label_map={0: 0, 1: 2}, text_column='text_combined'),
    DatasetSpec('Nazario.csv', label_map={1: 2}, default_label=2),
    DatasetSpec('Nigerian_Fraud.csv', label_map={1: 2}, default_label=2),
]


def _read_safe(df: pd.DataFrame, column: str, length: int) -> pd.Series:
    if column in df.columns:
        return df[column].fillna('').astype(str)
    return pd.Series([''] * length, dtype='object')


def load_source_dataset(spec: DatasetSpec) -> pd.DataFrame:
    path = DATASETS_DIR / spec.filename
    if not path.exists():
        raise FileNotFoundError(f'Missing dataset: {path}')

    df = pd.read_csv(path, encoding='latin-1')
    row_count = len(df.index)

    subject = _read_safe(df, 'subject', row_count)
    body = _read_safe(df, 'body', row_count)

    if spec.text_column and spec.text_column in df.columns:
        text = df[spec.text_column].fillna('').astype(str)
    else:
        text = (subject + ' ' + body).str.strip()

    if 'label' in df.columns:
        original_label = df['label']
    else:
        original_label = pd.Series([spec.default_label] * row_count)

    if spec.label_map:
        mapped_label = original_label.map(spec.label_map)
    else:
        mapped_label = pd.Series([spec.default_label] * row_count)

    if spec.default_label is not None:
        mapped_label = mapped_label.fillna(spec.default_label)

    if mapped_label.isnull().any():
        missing = sorted(original_label[mapped_label.isnull()].unique())
        raise ValueError(
            f'Encountered unmapped label(s) {missing} in dataset {spec.filename}. '
            'Update the label_map/default_label configuration.'
        )

    mapped_label = mapped_label.astype(int)

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

    prepared = prepared[prepared['text'].str.len() > 0]
    # Apply the shared cleaning routine so every dataset feeds identical tokens downstream.
    prepared['clean_text'] = prepared['text'].map(clean_email_text)
    prepared = prepared[prepared['clean_text'].str.len() > 0]
    return prepared


def load_wordcount_dataset(path: Path = DATASETS_DIR / 'emails.csv') -> pd.DataFrame:
    """Load the word-count style dataset where each column is a token count.

    The file is expected to contain an identifier column ("Email No."),
    3,000 word columns representing term frequencies, and a final "Prediction"
    column that encodes spam labels (1 for spam/fraudulent, 0 for normal).
    Rows are converted into synthetic text by repeating each token according
    to its frequency so that downstream text pipelines can consume them.
    """
    if not path.exists():
        return pd.DataFrame(
            columns=['subject', 'body', 'text', 'clean_text', 'original_label', 'label', 'label_name', 'source']
        )

    df = pd.read_csv(path)
    if 'Prediction' not in df.columns:
        raise ValueError(f'Column "Prediction" not found in {path}')

    word_columns = [col for col in df.columns if col not in ('Email No.', 'Prediction')]
    if not word_columns:
        return pd.DataFrame(
            columns=['subject', 'body', 'text', 'clean_text', 'original_label', 'label', 'label_name', 'source']
        )

    word_array = np.array(word_columns)
    records: List[dict] = []

    for _, row in df.iterrows():
        counts = row[word_columns].fillna(0).to_numpy(dtype=float)
        counts = np.maximum(counts, 0)
        counts = counts.astype(int)
        mask = counts > 0
        if not mask.any():
            continue
        # Recreate a pseudo-email by repeating each token by its observed frequency.
        tokens = np.repeat(word_array[mask], counts[mask])
        if tokens.size == 0:
            continue
        text = ' '.join(tokens.tolist())
        if not text:
            continue
        label = int(row['Prediction'])
        record = {
            'subject': str(row.get('Email No.', '')),
            'body': '',
            'text': text,
            'clean_text': text,
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
    # Load each labelled CSV and the word-count reconstruction so we can merge them.
    frames: List[pd.DataFrame] = [load_source_dataset(spec) for spec in DATASET_SOURCES]
    wordcount_df = load_wordcount_dataset()
    if not wordcount_df.empty:
        frames.append(wordcount_df)
    combined = pd.concat(frames, ignore_index=True)
    # Remove duplicate emails that might appear across corpora.
    combined.drop_duplicates(subset=['clean_text'], inplace=True)

    if balance:
        class_counts = combined['label'].value_counts()
        min_count = class_counts.min()
        balanced_frames = []
        for label_value, group in combined.groupby('label'):
            sample_size = min(len(group), min_count)
            balanced_frames.append(group.sample(n=sample_size, random_state=random_state))
        combined = pd.concat(balanced_frames, ignore_index=True)

    return combined


def write_labelled_datasets(dataset: pd.DataFrame) -> None:
    for label, output_path in LABELLED_OUTPUTS.items():
        subset = dataset[dataset['label'] == label]
        subset.to_csv(output_path, index=False, encoding='utf-8')


def summarize(dataset: pd.DataFrame) -> None:
    print('Dataset summary:')
    counts = dataset['label'].value_counts().sort_index()
    for label, count in counts.items():
        print(f'  {label} ({LABEL_NAMES.get(label, "Unknown")}): {count}')
    print(f'Total rows: {len(dataset)}')


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
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
    args = parse_args(argv)
    combined = build_combined_dataset(balance=args.balance)

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = DATASETS_DIR / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)

    combined.to_csv(output_path, index=False, encoding='utf-8')
    write_labelled_datasets(combined)
    summarize(combined)

    print(f'Combined dataset saved to {output_path.resolve()}')
    for label, path in LABELLED_OUTPUTS.items():
        print(f'  Label {label} -> {path.resolve()}')

    return output_path


if __name__ == '__main__':
    main()
