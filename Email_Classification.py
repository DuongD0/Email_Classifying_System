"""
MODULE: EMAIL CLASSIFICATION (MACHINE LEARNING)
================================================
Module này chịu trách nhiệm training và prediction cho email classification.

WORKFLOW TỔNG QUAN:
1. TRAINING MODE:
   - Load combined dataset (từ Extract_email)
   - Split train/test
   - Build pipeline: TF-IDF → Chi² feature selection → Classifier
   - Train model
   - Evaluate metrics
   - Save model (.joblib và .pkl)

2. PREDICTION MODE:
   - Load trained model
   - Predict label cho email mới
   - Show probabilities và top contributing words

MACHINE LEARNING PIPELINE:
┌─────────────┐    ┌──────────────┐    ┌─────────────┐    ┌────────────┐
│  Raw Text   │ -> │  TF-IDF      │ -> │  Chi² (opt) │ -> │ Classifier │
│             │    │  Vectorizer  │    │  SelectKBest│    │ LogReg/RF  │
└─────────────┘    └──────────────┘    └─────────────┘    └────────────┘

NGUYÊN LÝ HOẠT ĐỘNG:

1. TF-IDF (Term Frequency - Inverse Document Frequency):
   - Chuyển text thành vectors số
   - TF: Đếm tần suất từ trong document (email)
   - IDF: Giảm trọng số của từ phổ biến (xuất hiện nhiều docs)
   - Kết quả: Từ quan trọng và đặc trưng có trọng số cao
   - Sử dụng bigrams (1,2) để capture phrases ("free money", "click here")

2. Chi² Feature Selection (Optional):
   - Chọn k features quan trọng nhất dựa trên chi-squared statistic
   - Giảm dimensionality (từ 150k features xuống 2k)
   - Tăng tốc training và reduce overfitting
   - Chi² đo độ phụ thuộc giữa feature và label

3. Classifiers:
   a) Logistic Regression (default):
      - Linear classifier, fast và interpretable
      - Class-balanced: tự động weight classes để tránh bias
      - Coefficients cho thấy words nào contribute nhiều nhất
   
   b) Random Forest:
      - Ensemble của nhiều decision trees
      - Robust với noise và outliers
      - Class-balanced với 300 estimators

LABELS:
- 0: Normal (email thông thường)
- 1: Spam (email quảng cáo, rác)
- 2: Fraud (email lừa đảo, phishing)

METRICS:
- Accuracy: Tỷ lệ dự đoán đúng tổng thể
- Precision: Trong các emails dự đoán là X, bao nhiêu % thực sự là X
- Recall: Trong các emails thực sự là X, bao nhiêu % được phát hiện
- F1: Harmonic mean của Precision và Recall
- Macro: Average của từng class (treat classes equally)
"""

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

# ============================================================================
# CONFIGURATION: Paths
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / 'models'
MODEL_PATH = MODELS_DIR / 'email_classifier.joblib'  # Joblib format (fast)
PKL_MODEL_PATH = MODELS_DIR / 'email_classifier.pkl'  # Pickle format (compatible)


@dataclass
# Training knobs so CLI arguments and in-code defaults stay aligned.
class TrainingConfig:
    """
    Configuration cho training process.
    
    Tập hợp tất cả hyperparameters và settings để dễ quản lý.
    """
    test_size: float = 0.2  # 20% data cho testing, 80% cho training
    balance: bool = True  # Có balance classes hay không
    random_state: int = 42  # Seed cho reproducibility
    max_features: Optional[int] = None  # Giới hạn vocabulary size của TF-IDF
    model: str = 'logreg'  # Model type: 'logreg' hoặc 'rf'
    k_best: Optional[int] = None  # Số features sau chi² selection


def get_dataset(config: TrainingConfig) -> pd.DataFrame:
    """
    Load dataset cho training.
    
    Gọi build_combined_dataset từ Extract_email module để lấy
    dataset tổng hợp từ tất cả sources.
    
    Args:
        config: Training configuration
        
    Returns:
        DataFrame với clean_text và labels
    """
    # Assemble the cleaned dataset from every CSV source (optionally balanced).
    return build_combined_dataset(balance=config.balance, random_state=config.random_state)


def build_pipeline(
    model_type: str = 'logreg',
    max_features: Optional[int] = None,
    k_best: Optional[int] = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Xây dựng sklearn Pipeline cho training/prediction.
    
    PIPELINE STRUCTURE:
    1. TfidfVectorizer: Text -> TF-IDF vectors
       - stop_words='english': Loại bỏ stopwords (the, is, at, etc.)
       - max_features: Giới hạn vocabulary (nếu có)
       - ngram_range=(1,2): Unigrams và bigrams (capture phrases)
       - min_df=2: Bỏ qua từ xuất hiện < 2 documents (giảm noise)
    
    2. SelectKBest (Optional): Feature selection bằng chi²
       - Chọn k features có chi² score cao nhất
       - Giảm dimensionality để tăng tốc training
    
    3. Classifier: LogisticRegression hoặc RandomForest
       - class_weight='balanced': Tự động weight classes
       - Compensate cho imbalanced data
    
    NGUYÊN LÝ:
    - Pipeline đóng gói toàn bộ preprocessing + model
    - Khi save pipeline, mọi steps được save cùng
    - Prediction tự động apply đúng preprocessing
    - Tránh train/test mismatch
    
    Args:
        model_type: 'logreg' hoặc 'rf'
        max_features: Cap TF-IDF vocabulary
        k_best: Số features cho chi² selection
        random_state: Random seed
        
    Returns:
        Sklearn Pipeline sẵn sàng cho fit/predict
    """
    # Vectoriser → optional feature selection → estimator flow.
    steps: list[tuple[str, object]] = [
        (
            'tfidf',
            TfidfVectorizer(
                stop_words='english',  # Loại stopwords
                max_features=max_features,  # Cap vocabulary (None = unlimited)
                ngram_range=(1, 2),  # Unigrams + bigrams
                min_df=2,  # Bỏ từ xuất hiện < 2 docs
            ),
        )
    ]

    # Optional: Chi² feature selection
    if k_best is not None and k_best > 0:
        steps.append(('chi2', SelectKBest(score_func=chi2, k=k_best)))

    # Classifier selection
    if model_type == 'rf':
        # Random Forest: Ensemble của decision trees
        classifier = RandomForestClassifier(
            n_estimators=300,  # 300 trees
            class_weight='balanced',  # Auto-weight classes
            random_state=random_state,
            n_jobs=-1,  # Use all CPU cores
        )
    else:
        # Logistic Regression (default): Linear classifier
        classifier = LogisticRegression(
            max_iter=1000,  # Đủ iterations cho convergence
            class_weight='balanced',  # Auto-weight classes
            solver='lbfgs',  # Optimizer (tốt cho multi-class)
        )

    steps.append(('clf', classifier))
    return Pipeline(steps=steps)
    


@dataclass
class TrainingResult:
    """
    Container cho kết quả training.
    
    Chứa mọi thông tin cần thiết về model đã train:
    - Pipeline đã fit
    - Metrics (accuracy, precision, recall, F1)
    - Classification report
    - Confusion matrix
    - Per-class metrics
    """
    pipeline: Pipeline  # Trained pipeline
    report: str  # Classification report text
    confusion: np.ndarray  # Confusion matrix
    labels: np.ndarray  # Label values ([0, 1, 2])
    accuracy: float  # Overall accuracy
    model_type: str  # 'logreg' or 'rf'
    precision_macro: float  # Macro-averaged precision
    recall_macro: float  # Macro-averaged recall
    f1_macro: float  # Macro-averaged F1
    per_class_metrics: dict[str, dict[str, float | int]]  # Metrics per label


def train_model(config: TrainingConfig) -> TrainingResult:
    """
    Train model với configuration cho trước.
    
    WORKFLOW ĐẦY ĐỦ:
    1. Load dataset (get_dataset)
    2. Extract X (clean_text) và y (labels)
    3. Train/test split (stratified để giữ class distribution)
    4. Build pipeline
    5. Fit pipeline trên training data
    6. Predict trên test data
    7. Calculate metrics (accuracy, precision, recall, F1)
    8. Generate classification report và confusion matrix
    
    STRATIFIED SPLIT:
    - Đảm bảo train và test có cùng class distribution
    - Ví dụ: Nếu dataset có 40% Normal, 30% Spam, 30% Fraud
    - Train và test đều có tỷ lệ 40-30-30
    - Tránh bias do split không đều
    
    METRICS EXPLAINED:
    - Accuracy: (TP + TN) / Total
    - Precision: TP / (TP + FP) - "Trong dự đoán positive, bao nhiêu đúng?"
    - Recall: TP / (TP + FN) - "Trong thực tế positive, bao nhiêu detect được?"
    - F1: 2 * (Precision * Recall) / (Precision + Recall) - Harmonic mean
    - Macro: Average các class (treat equally), Micro: Average tất cả samples
    
    Args:
        config: TrainingConfig với hyperparameters
        
    Returns:
        TrainingResult chứa trained pipeline và metrics
    """
    # Load combined dataset
    data = get_dataset(config)
    
    # Determine text column (prioritize clean_text)
    text_column = 'clean_text' if 'clean_text' in data.columns else 'text'
    
    # Filter bỏ rows có text rỗng
    data = data[data[text_column].astype(str).str.len() > 0]
    
    # Extract features (X) và labels (y)
    X = data[text_column].astype(str)
    y = data['label'].astype(int)

    # Hold out a validation slice so metrics reflect generalisation, not training fit.
    # Stratified split: giữ class distribution trong train và test
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=config.test_size,  # 20% test, 80% train
        random_state=config.random_state,
        stratify=y,  # Keep same class distribution
    )

    # Build pipeline với config
    pipeline = build_pipeline(
        model_type=config.model,
        max_features=config.max_features,
        k_best=config.k_best,
        random_state=config.random_state,
    )
    
    # Train the entire pipeline end-to-end so preprocessing stays attached to the model.
    # Fit trên training data (TF-IDF learn vocabulary, model learn weights)
    pipeline.fit(X_train, y_train)

    # Predict trên test data (unseen data)
    y_pred = pipeline.predict(X_test)
    
    # Calculate accuracy
    accuracy = (y_pred == y_test).mean()
    
    # Compute granular metrics to surface strengths/weaknesses per class.
    sorted_labels = np.array(sorted(LABEL_NAMES))  # [0, 1, 2]
    target_names = [LABEL_NAMES[i] for i in sorted_labels]  # ['Normal', 'Spam', 'Fraud']
    
    # Classification report: tổng hợp metrics cho từng class
    report = classification_report(
        y_test,
        y_pred,
        labels=sorted_labels,
        target_names=target_names,
        zero_division=0,
    )
    
    # Confusion matrix: true labels (rows) vs predicted labels (cols)
    # confusion[i][j] = số emails có true label i được predict là j
    confusion = confusion_matrix(y_test, y_pred, labels=sorted_labels)
    
    # Macro-averaged metrics (average across classes)
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average='macro', zero_division=0
    )
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_test, y_pred, labels=sorted_labels, zero_division=0
    )

    # Organize per-class metrics vào dict
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
    """
    Save trained pipeline ra files.
    
    Save 2 formats:
    1. .joblib: Joblib format - nhanh, hiệu quả cho sklearn objects
    2. .pkl: Pickle format - universal Python serialization
    
    Joblib được recommend cho sklearn vì nhanh hơn và xử lý
    large numpy arrays tốt hơn pickle.
    
    Args:
        pipeline: Trained sklearn Pipeline
        path: Path cho .joblib file
        pickle_path: Path cho .pkl file
        
    Returns:
        Tuple (joblib_path, pickle_path)
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    # Save as joblib (recommended for sklearn)
    joblib.dump(pipeline, path)
    
    # Save as pickle (for compatibility)
    with pickle_path.open('wb') as pkl_file:
        pickle.dump(pipeline, pkl_file)
    
    return path, pickle_path


def load_model(path: Path = MODEL_PATH) -> Pipeline:
    """
    Load trained pipeline từ file.
    
    Args:
        path: Path tới .joblib file
        
    Returns:
        Sklearn Pipeline đã trained
        
    Raises:
        FileNotFoundError: Nếu model file không tồn tại
    """
    if not path.exists():
        raise FileNotFoundError(f'Model not found at {path}. Run training first.')
    return joblib.load(path)


class EmailClassifier:
    """
    Wrapper class cho trained model, cung cấp interface dễ dùng.
    
    USAGE:
        clf = EmailClassifier()
        label, label_name = clf.predict("Click here to win money!")
        # Returns: (1, "Spam")
        
        details = clf.predict_with_details("Your account has been suspended")
        # Returns: {'label': 2, 'label_name': 'Fraud', 'probabilities': {...}, ...}
    
    FEATURES:
    - predict(): Dự đoán label và label name
    - predict_with_details(): Dự đoán + probabilities + top contributing words
    - _suspicious_words(): Phân tích words nào contribute nhiều nhất cho mỗi class
    
    INTERPRETABILITY:
    - LogReg coefficients cho biết từ nào quan trọng cho class nào
    - Positive coef: từ này làm tăng xác suất class đó
    - Negative coef: từ này làm giảm xác suất class đó
    """
    
    def __init__(self, model_path: Path = MODEL_PATH):
        """
        Initialize classifier bằng cách load trained pipeline.
        
        Args:
            model_path: Path tới trained model (.joblib)
        """
        # Load the persisted pipeline so CLI runs and app usage behave identically.
        self.pipeline = load_model(model_path)
        # Extract feature names cho interpretability
        self.feature_names_ = self._compute_feature_names()

    def predict(self, text: str) -> tuple[int, str]:
        """
        Dự đoán label cho email text.
        
        WORKFLOW:
        1. Text đi qua TF-IDF vectorizer
        2. Optionally qua chi² feature selection
        3. Qua classifier để predict label
        
        Args:
            text: Văn bản email cần classify
            
        Returns:
            Tuple (label, label_name)
            Ví dụ: (1, "Spam") hoặc (2, "Fraud")
        """
        label = int(self.pipeline.predict([text])[0])
        return label, LABEL_NAMES.get(label, str(label))

    def predict_with_details(self, text: str, top_n: int = 10) -> dict:
        """
        Dự đoán label kèm thông tin chi tiết.
        
        Trả về:
        - label: Predicted label (0/1/2)
        - label_name: Label name (Normal/Spam/Fraud)
        - probabilities: Xác suất của từng class (%)
        - suspicious_words: Top words contribute nhiều nhất cho mỗi class
        - predicted_probability: Xác suất của predicted class
        - top_contributing_words: Top words cho predicted class
        - class_percentages: % contribution của mỗi class
        
        INTERPRETABILITY:
        - Suspicious words giúp hiểu TẠI SAO model predict class đó
        - Ví dụ: "money" và "free" contribute cao cho Spam
        - "account" và "verify" contribute cao cho Fraud
        
        Args:
            text: Văn bản email cần classify
            top_n: Số top words cần show cho mỗi class
            
        Returns:
            Dict chứa label, probabilities, và suspicious words
        """
        # Predict label
        label = int(self.pipeline.predict([text])[0])
        
        # Get probabilities cho tất cả classes
        probabilities = self._predict_proba(text)
        
        # Translate model coefficients into human-readable token contributions for each class.
        # Phân tích words nào contribute nhiều nhất
        suspicious = self._suspicious_words(text, top_n=top_n)
        
        label_name = LABEL_NAMES.get(label, str(label))
        predicted_probability = probabilities.get(label_name)
        
        # Extract class percentages từ suspicious words
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
        """
        Extract feature names từ pipeline.
        
        WORKFLOW:
        1. Get feature names từ TF-IDF vectorizer
        2. Nếu có chi² selection, filter chỉ giữ selected features
        
        Feature names là các words/bigrams trong vocabulary.
        Ví dụ: ["money", "free", "account", "click here", ...]
        
        Returns:
            Array of feature names (words/bigrams)
        """
        vectorizer = self.pipeline.named_steps.get('tfidf')
        if vectorizer is None:
            return np.array([])
        
        # Get all feature names từ TF-IDF
        feature_names = vectorizer.get_feature_names_out()
        
        # Nếu có chi² selection, filter features
        if 'chi2' in self.pipeline.named_steps:
            support = self.pipeline.named_steps['chi2'].get_support()
            feature_names = feature_names[support]
        
        return np.array(feature_names)

    def _predict_proba(self, text: str) -> dict:
        """
        Get class probabilities dưới dạng percentages.
        
        Convert raw probabilities (0.0-1.0) sang percentages (0-100).
        Sort theo probability giảm dần.
        
        Args:
            text: Văn bản email
            
        Returns:
            Dict {label_name: probability%}
            Ví dụ: {'Spam': 67.78, 'Fraud': 30.07, 'Normal': 2.16}
        """
        # Convert raw probabilities to percentages for friendlier CLI output.
        clf = self.pipeline.named_steps.get('clf')
        if clf is None or not hasattr(clf, 'predict_proba'):
            return {}
        
        # Get probabilities từ classifier
        proba = self.pipeline.predict_proba([text])[0]
        
        # Convert sang percentages và map labels
        pairs = [
            (LABEL_NAMES.get(int(cls), str(cls)), float(prob) * 100.0)
            for cls, prob in zip(clf.classes_, proba)
        ]
        
        # Sort giảm dần theo probability
        pairs.sort(key=lambda item: item[1], reverse=True)
        
        return {name: value for name, value in pairs}

    def _suspicious_words(self, text: str, top_n: int = 10) -> dict:
        """
        Phân tích words nào contribute nhiều nhất cho mỗi class.
        
        NGUYÊN LÝ (chỉ áp dụng cho LogisticRegression):
        1. Transform text thành TF-IDF vector
        2. Lấy coefficients từ model (mỗi class có 1 set coefficients)
        3. Score của mỗi word = TF-IDF value * coefficient
        4. Positive score = từ này push về class đó
        5. Top words = words có score cao nhất
        
        VÍ DỤ:
        Text: "free money now"
        - "free" có coef cao cho Spam -> high score
        - "money" có coef cao cho Spam và Fraud -> high scores cho cả 2
        - Total scores: Spam=5.2, Fraud=3.1, Normal=0.5
        - Percentages: Spam=59%, Fraud=35%, Normal=6%
        
        Args:
            text: Văn bản email
            top_n: Số top words cho mỗi class
            
        Returns:
            Dict {label_name: {'words': [(word, score), ...], 
                               'total_score': float,
                               'percentage': float}}
            Trả về {} nếu model không có coefficients (ví dụ: Random Forest)
        """
        clf = self.pipeline.named_steps.get('clf')
        
        # Chỉ work với models có coefficients (LogReg)
        if clf is None or not hasattr(clf, 'coef_') or self.feature_names_.size == 0:
            return {}
        
        # Build preprocessor (tất cả steps trừ classifier)
        preprocessor = Pipeline(self.pipeline.steps[:-1])
        
        # Transform text thành vector (TF-IDF)
        vector = preprocessor.transform([text])
        
        # Align coefficient rows with their readable label names for downstream reporting.
        label_lookup = {
            idx: LABEL_NAMES.get(int(cls), str(cls)) for idx, cls in enumerate(clf.classes_)
        }
        
        # Initialize contributions dict
        contributions: dict[str, list[tuple[str, float]]] = {
            name: [] for name in label_lookup.values()
        }
        
        # Convert sparse matrix sang COO format (dễ iterate)
        vector = vector.tocoo()
        
        # Iterate qua non-zero elements trong vector
        for col, value in zip(vector.col, vector.data):
            if col >= len(self.feature_names_):
                continue
            
            word = self.feature_names_[col]
            
            # Calculate score cho mỗi class
            for idx, _ in enumerate(clf.classes_):
                label_name = label_lookup[idx]
                # Score = TF-IDF value * coefficient
                score = float(value * clf.coef_[idx, col])
                
                # Chỉ keep positive scores (contribute về class đó)
                if score > 0:
                    contributions[label_name].append((word, score))

        # Calculate total scores cho mỗi class
        totals = {
            name: sum(score for _, score in scores) for name, scores in contributions.items()
        }
        total_positive = sum(totals.values())
        
        # Build results dict
        results: dict[str, dict[str, object]] = {}
        for label_name, scores in contributions.items():
            if not scores:
                continue
            
            # Sort words giảm dần theo score
            scores.sort(key=lambda item: item[1], reverse=True)
            
            # Take top N words
            top_scores = scores[:top_n]
            
            total_score = totals[label_name]
            
            # Calculate percentage contribution
            percentage = (total_score / total_positive * 100.0) if total_positive > 0 else 0.0
            
            results[label_name] = {
                'words': top_scores,
                'total_score': total_score,
                'percentage': percentage,
            }
        
        return results


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """
    Parse command-line arguments.
    
    ARGUMENTS:
    --balance / --no-balance: Balance classes hay không
    --test-size: Tỷ lệ test set (0.0-1.0)
    --max-features: Cap TF-IDF vocabulary size
    --k-best: Số features sau chi² selection
    --model: 'logreg' hoặc 'rf'
    --predict: Text để predict (skip training mode)
    --model-path: Path để save/load model
    --pickle-path: Path cho pickle file
    """
    parser = argparse.ArgumentParser(description='Train or use the email classifier pipeline.')
    
    # Balance group: mutually exclusive
    balance_group = parser.add_mutually_exclusive_group()
    balance_group.add_argument('--balance', dest='balance', action='store_true', 
                              help='Down-sample classes to balance the dataset before training (default).')
    balance_group.add_argument('--no-balance', dest='balance', action='store_false', 
                              help='Use the raw class distribution without balancing.')
    parser.set_defaults(balance=True)
    
    parser.add_argument('--test-size', type=float, default=0.2, 
                       help='Test size for train/test split (default: 0.2).')
    parser.add_argument('--max-features', type=int, 
                       help='Cap TF-IDF vocabulary size.')
    parser.add_argument('--k-best', type=int, 
                       help='Select top-K features via chi-squared after TF-IDF.')
    parser.add_argument('--model', choices=['logreg', 'rf'], default='logreg', 
                       help='Choose classifier: logistic regression (default) or random forest.')
    parser.add_argument('--predict', 
                       help='Skip training and classify the provided text using the saved model.')
    parser.add_argument('--model-path', type=Path, default=MODEL_PATH, 
                       help='Path to save/load the joblib pipeline.')
    parser.add_argument('--pickle-path', type=Path, default=PKL_MODEL_PATH, 
                       help='Path to save the pickled pipeline during training.')
    
    return parser.parse_args(list(argv) if argv is not None else None)


def main(argv: Optional[Iterable[str]] = None) -> Optional[EmailClassifier]:
    """
    Main function: Training hoặc Prediction mode.
    
    PREDICTION MODE (--predict):
    1. Load trained model
    2. Predict label cho text
    3. Show probabilities và top contributing words
    
    TRAINING MODE (default):
    1. Load dataset
    2. Train model với config
    3. Evaluate metrics
    4. Save model
    5. Print results
    
    Args:
        argv: Command-line arguments (None = sys.argv)
        
    Returns:
        EmailClassifier instance (trained hoặc loaded)
    """
    args = parse_args(argv)

    # ========================================================================
    # PREDICTION MODE
    # ========================================================================
    if args.predict:
        # Load trained model
        classifier = EmailClassifier(model_path=args.model_path)
        
        # Predict với details
        details = classifier.predict_with_details(args.predict)
        
        # Print results
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
                # Format: word (score), word (score), ...
                formatted = ', '.join(f"{word} ({score:.3f})" for word, score in words)
                percentage = info.get('percentage')
                if percentage is not None:
                    print(f'  {class_name} ({percentage:.2f}% contribution): {formatted}')
                else:
                    print(f'  {class_name}: {formatted}')
        
        return classifier

    # ========================================================================
    # TRAINING MODE
    # ========================================================================
    config = TrainingConfig(
        test_size=args.test_size,
        balance=args.balance,
        max_features=args.max_features,
        model=args.model,
        k_best=args.k_best,
    )

    # Train model
    result = train_model(config)
    
    # Save model
    joblib_path, pickle_path = save_model(result.pipeline, path=args.model_path, pickle_path=args.pickle_path)

    # Print results
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
