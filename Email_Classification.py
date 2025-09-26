from __future__ import annotations

import argparse
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from Extract_email import LABEL_NAMES, build_combined_dataset

BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'email_classifier.joblib'
PKL_MODEL_PATH = MODELS_DIR / 'email_classifier.pkl'


@dataclass
# Training knobs so CLI arguments and in-code defaults stay aligned.
class TrainingConfig:
    test_size: float = 0.2
    balance: bool = True
    random_state: int = 42
    max_features: Optional[int] = None
    model: str = 'logreg'
    k_best: Optional[int] = None


def get_dataset(config: TrainingConfig) -> pd.DataFrame:
    # Assemble the cleaned dataset from every CSV source (optionally balanced).
    return build_combined_dataset(balance=config.balance, random_state=config.random_state)


def build_pipeline(
    model_type: str = 'logreg',
    max_features: Optional[int] = None,
    k_best: Optional[int] = None,
    random_state: int = 42,
) -> Pipeline:
    # Vectoriser → optional feature selection → estimator flow.
    steps: list[tuple[str, object]] = [
        (
            'tfidf',
            TfidfVectorizer(
                stop_words='english',
                max_features=max_features,
                ngram_range=(1, 2),
                min_df=2,
            ),
        )
    ]

    if k_best is not None and k_best > 0:
        steps.append(('chi2', SelectKBest(score_func=chi2, k=k_best)))

    if model_type == 'rf':
        classifier = RandomForestClassifier(
            n_estimators=300,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1,
        )
    else:
        classifier = LogisticRegression(
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
        )

    steps.append(('clf', classifier))
    return Pipeline(steps=steps)
    


@dataclass
class TrainingResult:
    pipeline: Pipeline
    report: str
    confusion: np.ndarray
    labels: np.ndarray
    accuracy: float
    model_type: str
    precision_macro: float
    recall_macro: float
    f1_macro: float
    per_class_metrics: dict[str, dict[str, float | int]]


def train_model(config: TrainingConfig) -> TrainingResult:
    data = get_dataset(config)
    text_column = 'clean_text' if 'clean_text' in data.columns else 'text'
    data = data[data[text_column].astype(str).str.len() > 0]
    X = data[text_column].astype(str)
    y = data['label'].astype(int)

    # Hold out a validation slice so metrics reflect generalisation, not training fit.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,
        random_state=config.random_state,
        stratify=y,
    )

    pipeline = build_pipeline(
        model_type=config.model,
        max_features=config.max_features,
        k_best=config.k_best,
        random_state=config.random_state,
    )
    # Train the entire pipeline end-to-end so preprocessing stays attached to the model.
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    accuracy = (y_pred == y_test).mean()
    # Compute granular metrics to surface strengths/weaknesses per class.
    sorted_labels = np.array(sorted(LABEL_NAMES))
    target_names = [LABEL_NAMES[i] for i in sorted_labels]
    report = classification_report(
        y_test,
        y_pred,
        labels=sorted_labels,
        target_names=target_names,
        zero_division=0,
    )
    confusion = confusion_matrix(y_test, y_pred, labels=sorted_labels)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, labels=sorted_labels, zero_division=0
    )

    per_class_metrics: dict[str, dict[str, float]] = {}
    for idx, label_value in enumerate(sorted_labels):
        label_name = LABEL_NAMES.get(int(label_value), str(label_value))
        per_class_metrics[label_name] = {
            'precision': float(precision_per_class[idx]),
            'recall': float(recall_per_class[idx]),
            'f1': float(f1_per_class[idx]),
            'support': int(support_per_class[idx]),
        }

    return TrainingResult(
        pipeline=pipeline,
        report=report,
        confusion=confusion,
        labels=sorted_labels,
        accuracy=float(accuracy),
        model_type=config.model,
        precision_macro=float(precision_macro),
        recall_macro=float(recall_macro),
        f1_macro=float(f1_macro),
        per_class_metrics=per_class_metrics,
    )


def save_model(
    pipeline: Pipeline,
    path: Path = MODEL_PATH,
    pickle_path: Path = PKL_MODEL_PATH,
) -> tuple[Path, Path]:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    with pickle_path.open('wb') as pkl_file:
        pickle.dump(pipeline, pkl_file)
    return path, pickle_path


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    if not path.exists():
        raise FileNotFoundError(f'Model not found at {path}. Run training first.')
    return joblib.load(path)


class EmailClassifier:
    def __init__(self, model_path: Path = MODEL_PATH):
        # Load the persisted pipeline so CLI runs and app usage behave identically.
        self.pipeline = load_model(model_path)
        self.feature_names_ = self._compute_feature_names()

    def predict(self, text: str) -> tuple[int, str]:
        label = int(self.pipeline.predict([text])[0])
        return label, LABEL_NAMES.get(label, str(label))

    def predict_with_details(self, text: str, top_n: int = 10) -> dict:
        label = int(self.pipeline.predict([text])[0])
        probabilities = self._predict_proba(text)
        # Translate model coefficients into human-readable token contributions for each class.
        suspicious = self._suspicious_words(text, top_n=top_n)
        label_name = LABEL_NAMES.get(label, str(label))
        predicted_probability = probabilities.get(label_name)
        class_percentages = {
            name: details['percentage']
            for name, details in (suspicious or {}).items()
            if 'percentage' in details
        }
        return {
            'label': label,
            'label_name': label_name,
            'probabilities': probabilities,
            'suspicious_words': suspicious,
            'predicted_probability': predicted_probability,
            'top_contributing_words': suspicious.get(label_name, {}).get('words', []) if suspicious else [],
            'class_percentages': class_percentages,
        }

    def _compute_feature_names(self) -> np.ndarray:
        vectorizer = self.pipeline.named_steps.get('tfidf')
        if vectorizer is None:
            return np.array([])
        feature_names = vectorizer.get_feature_names_out()
        if 'chi2' in self.pipeline.named_steps:
            support = self.pipeline.named_steps['chi2'].get_support()
            feature_names = feature_names[support]
        return np.array(feature_names)

    def _predict_proba(self, text: str) -> dict:
        # Convert raw probabilities to percentages for friendlier CLI output.
        clf = self.pipeline.named_steps.get('clf')
        if clf is None or not hasattr(clf, 'predict_proba'):
            return {}
        proba = self.pipeline.predict_proba([text])[0]
        pairs = [
            (LABEL_NAMES.get(int(cls), str(cls)), float(prob) * 100.0)
            for cls, prob in zip(clf.classes_, proba)
        ]
        pairs.sort(key=lambda item: item[1], reverse=True)
        return {name: value for name, value in pairs}

    def _suspicious_words(self, text: str, top_n: int = 10) -> dict:
        clf = self.pipeline.named_steps.get('clf')
        if clf is None or not hasattr(clf, 'coef_') or self.feature_names_.size == 0:
            return {}
        preprocessor = Pipeline(self.pipeline.steps[:-1])
        vector = preprocessor.transform([text])
        # Align coefficient rows with their readable label names for downstream reporting.
        label_lookup = {
            idx: LABEL_NAMES.get(int(cls), str(cls)) for idx, cls in enumerate(clf.classes_)
        }
        contributions: dict[str, list[tuple[str, float]]] = {
            name: [] for name in label_lookup.values()
        }
        vector = vector.tocoo()
        for col, value in zip(vector.col, vector.data):
            if col >= len(self.feature_names_):
                continue
            word = self.feature_names_[col]
            for idx, _ in enumerate(clf.classes_):
                label_name = label_lookup[idx]
                score = float(value * clf.coef_[idx, col])
                if score > 0:
                    contributions[label_name].append((word, score))

        totals = {
            name: sum(score for _, score in scores) for name, scores in contributions.items()
        }
        total_positive = sum(totals.values())
        results: dict[str, dict[str, object]] = {}
        for label_name, scores in contributions.items():
            if not scores:
                continue
            scores.sort(key=lambda item: item[1], reverse=True)
            top_scores = scores[:top_n]
            total_score = totals[label_name]
            percentage = (total_score / total_positive * 100.0) if total_positive > 0 else 0.0
            results[label_name] = {
                'words': top_scores,
                'total_score': total_score,
                'percentage': percentage,
            }
        return results


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Train or use the email classifier pipeline.')
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument('--balance', dest='balance', action='store_true', help='Down-sample classes to balance the dataset before training (default).')
    balance_group.add_argument('--no-balance', dest='balance', action='store_false', help='Use the raw class distribution without balancing.')
    parser.set_defaults(balance=True)
    parser.add_argument('--test-size', type=float, default=0.2, help='Test size for train/test split (default: 0.2).')
    parser.add_argument('--max-features', type=int, help='Cap TF-IDF vocabulary size.')
    parser.add_argument('--k-best', type=int, help='Select top-K features via chi-squared after TF-IDF.')
    parser.add_argument('--model', choices=['logreg', 'rf'], default='logreg', help='Choose classifier: logistic regression (default) or random forest.')
    parser.add_argument('--predict', help='Skip training and classify the provided text using the saved model.')
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH, help='Path to save/load the joblib pipeline.')
    parser.add_argument('--pickle-path', type=Path, default=PKL_MODEL_PATH, help='Path to save the pickled pipeline during training.')
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> Optional[EmailClassifier]:
    args = parse_args(argv)

    if args.predict:
        classifier = EmailClassifier(model_path=args.model_path)
        details = classifier.predict_with_details(args.predict)
        print(f"Predicted label: {details['label']} ({details['label_name']})")
        if details['probabilities']:
            print('Class probabilities:')
            for name, prob in details['probabilities'].items():
                print(f'  {name}: {prob:.2f}%')
        if details['suspicious_words']:
            print('Top contributing terms:')
            for class_name, info in details['suspicious_words'].items():
                words = info.get('words', [])
                if not words:
                    continue
                formatted = ', '.join(f"{word} ({score:.3f})" for word, score in words)
                percentage = info.get('percentage')
                if percentage is not None:
                    print(f'  {class_name} ({percentage:.2f}% contribution): {formatted}')
                else:
                    print(f'  {class_name}: {formatted}')
        return classifier

    config = TrainingConfig(
        test_size=args.test_size,
        balance=args.balance,
        max_features=args.max_features,
        model=args.model,
        k_best=args.k_best,
    )

    result = train_model(config)
    joblib_path, pickle_path = save_model(result.pipeline, path=args.model_path, pickle_path=args.pickle_path)

    print(f'Model saved to {joblib_path.resolve()}')
    print(f'Pickled pipeline saved to {pickle_path.resolve()}')
    print(f'Accuracy: {result.accuracy:.4f}')
    print(f'Macro Precision: {result.precision_macro:.4f}')
    print(f'Macro Recall: {result.recall_macro:.4f}')
    print(f'Macro F1: {result.f1_macro:.4f}')
    print('Per-class metrics:')
    for label in result.labels:
        name = LABEL_NAMES.get(int(label), str(label))
        metrics = result.per_class_metrics.get(name)
        if not metrics:
            continue
        print(
            f'  {name}: '
            f"Precision={metrics['precision']:.4f}, "
            f"Recall={metrics['recall']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"Support={int(metrics['support'])}"
        )
    print(f'Classifier: {result.model_type}')
    print('Classification report:')
    print(result.report)
    print('Confusion matrix (rows=true, cols=pred):')
    print(pd.DataFrame(
        result.confusion,
        index=[LABEL_NAMES.get(int(label), str(label)) for label in result.labels],
        columns=[LABEL_NAMES.get(int(label), str(label)) for label in result.labels],
    ))

    return EmailClassifier(model_path=args.model_path)


if __name__ == '__main__':
    main()
