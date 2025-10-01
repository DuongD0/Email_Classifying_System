# TÀI LIỆU WORKFLOW VÀ NGUYÊN LÝ HOẠT ĐỘNG
# Email Classification System

## TỔNG QUAN HỆ THỐNG

Hệ thống này là một **Email Classification Pipeline** sử dụng Machine Learning để phân loại email thành 3 loại:
- **0 - Normal**: Email thông thường, hợp pháp
- **1 - Spam**: Email quảng cáo, rác
- **2 - Fraud**: Email lừa đảo, phishing, Nigerian scam

---

## KIẾN TRÚC TỔNG THỂ

```
┌──────────────────────────────────────────────────────────────────┐
│                    EMAIL CLASSIFICATION SYSTEM                    │
└──────────────────────────────────────────────────────────────────┘

┌─────────────────────┐
│  DATASETS (Raw CSVs)│
│  - Enron.csv        │
│  - CEAS_08.csv      │
│  - Ling.csv         │
│  - SpamAssassin.csv │
│  - phishing_email   │
│  - Nazario.csv      │
│  - Nigerian_Fraud   │
└──────────┬──────────┘
           │
           ▼
┌──────────────────────┐
│ Extract_email.py     │
│ (Dataset Assembly)   │
│                      │
│ 1. Load từng source  │
│ 2. Normalize schema  │
│ 3. Map labels 0/1/2  │
│ 4. Clean text        │
│ 5. Deduplicate       │
│ 6. Balance classes   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────┐
│ text_preprocessing.py    │
│ (Text Cleaning)          │
│                          │
│ 1. HTML stripping        │
│ 2. Tokenization          │
│ 3. Lowercase             │
│ 4. Remove stopwords      │
│ 5. Lemmatization         │
└──────────┬───────────────┘
           │
           ▼
┌────────────────────────────┐
│  combined_emails.csv       │
│  (Unified Dataset)         │
│  ~134k rows                │
└──────────┬─────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ Email_Classification.py      │
│ (Machine Learning)           │
│                              │
│ Pipeline:                    │
│ Text → TF-IDF → Chi² → Model│
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────┐
│  Trained Models          │
│  - email_classifier.pkl  │
│  - email_classifier.     │
│    joblib                │
└──────────┬───────────────┘
           │
           ▼
┌──────────────────────┐
│  PREDICTION          │
│  Input: Email text   │
│  Output: Label +     │
│          Probability │
└──────────────────────┘
```

---

## MODULE 1: text_preprocessing.py

### Mục đích
Làm sạch và chuẩn hóa văn bản email để chuẩn bị cho ML model.

### Workflow Chi Tiết

```
Raw Email Text
    │
    ▼
┌─────────────────────────────┐
│ 1. HTML Unescape            │
│    &amp; → &                 │
│    &lt; → <                  │
│    &nbsp; → space            │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 2. Strip HTML Tags          │
│    <div>, <p>, <script>     │
│    → removed                │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 3. Normalize Whitespace     │
│    Multiple spaces → 1 space│
│    Tabs, newlines → space   │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 4. Lowercase                │
│    "Email" → "email"        │
│    "FREE" → "free"          │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 5. Remove Non-Alpha Chars   │
│    Only keep a-z and spaces │
│    "money!!!" → "money"     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 6. Tokenization             │
│    Split by spaces          │
│    "free money" → ["free",  │
│                    "money"] │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 7. Filter Short Tokens      │
│    Remove tokens <= 2 chars │
│    "to", "at" → removed     │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 8. Remove Stopwords         │
│    "the", "is", "at", etc.  │
│    → removed                │
└─────────────┬───────────────┘
              │
              ▼
┌─────────────────────────────┐
│ 9. Lemmatization            │
│    "running" → "run"        │
│    "better" → "good"        │
│    "cars" → "car"           │
└─────────────┬───────────────┘
              │
              ▼
    Clean Token List
    ["free", "money", "win"]
              │
              ▼
┌─────────────────────────────┐
│ 10. Join Tokens             │
│     "free money win"        │
└─────────────────────────────┘
```

### Nguyên Lý Kỹ Thuật

1. **HTML Processing**
   - Email thường chứa HTML (đặc biệt spam/phishing)
   - HTML entities và tags không mang ý nghĩa phân loại
   - Loại bỏ giúp tập trung vào nội dung thực

2. **Lowercase Normalization**
   - "Free", "FREE", "free" → tất cả thành "free"
   - Giảm vocabulary size
   - Cải thiện generalization

3. **Stopwords Removal**
   - Các từ phổ biến (the, is, at) không giúp phân biệt classes
   - Loại bỏ giảm noise và feature space
   - Tập trung vào từ có giá trị

4. **Lemmatization**
   - Đưa từ về dạng gốc (root form)
   - "running", "runs", "ran" → "run"
   - Gom nhóm các biến thể của cùng một concept
   - Giảm vocabulary size, tăng signal

5. **Caching với @lru_cache**
   - Regex compilation tốn thời gian
   - Cache để tái sử dụng
   - Tăng performance khi xử lý nhiều emails

---

## MODULE 2: Extract_email.py

### Mục đích
Xây dựng unified dataset từ nhiều nguồn email khác nhau.

### Workflow Chi Tiết

```
┌──────────────────────────────────────────────────────────┐
│              MULTI-SOURCE DATASET ASSEMBLY               │
└──────────────────────────────────────────────────────────┘

┌────────────┐  ┌────────────┐  ┌────────────┐  ┌────────────┐
│ Enron.csv  │  │ CEAS_08    │  │ phishing   │  │ Nigerian   │
│ label:     │  │ label:     │  │ label:     │  │ label:     │
│ 0=normal   │  │ 0=normal   │  │ 0=normal   │  │ 1=fraud    │
│ 1=spam     │  │ 1=spam     │  │ 1=fraud    │  │            │
└─────┬──────┘  └─────┬──────┘  └─────┬──────┘  └─────┬──────┘
      │               │               │               │
      └───────────────┴───────────────┴───────────────┘
                            │
                            ▼
              ┌──────────────────────────┐
              │  load_source_dataset()   │
              │                          │
              │  For each source:        │
              │  1. Read CSV             │
              │  2. Extract subject+body │
              │  3. Map labels → 0/1/2   │
              │  4. Validate labels      │
              │  5. Clean text           │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Standardized DataFrame  │
              │  Columns:                │
              │  - subject               │
              │  - body                  │
              │  - text                  │
              │  - clean_text            │
              │  - label (0/1/2)         │
              │  - label_name            │
              │  - source                │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  pd.concat() all sources │
              │  ~134,000 rows           │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Remove Duplicates       │
              │  Based on clean_text     │
              │  (Same email from        │
              │   different sources)     │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Optional: Balance       │
              │  Classes                 │
              │                          │
              │  Before:                 │
              │  Normal: 64.6k           │
              │  Spam:   35.4k           │
              │  Fraud:  34.4k           │
              │                          │
              │  After (balanced):       │
              │  Normal: 34.4k           │
              │  Spam:   34.4k           │
              │  Fraud:  34.4k           │
              │  Total: ~103k            │
              └──────────┬───────────────┘
                         │
                         ▼
              ┌──────────────────────────┐
              │  Output Files            │
              │                          │
              │  - combined_emails.csv   │
              │  - normal_emails.csv     │
              │  - spam_emails.csv       │
              │  - fraud_emails.csv      │
              └──────────────────────────┘
```

### Label Mapping Strategy

Mỗi dataset có label encoding khác nhau. Ta cần map về unified schema:

| Dataset          | Original Labels | Mapped Labels        |
|------------------|----------------|----------------------|
| Enron            | 0=ham, 1=spam  | 0=Normal, 1=Spam     |
| CEAS_08          | 0=ham, 1=spam  | 0=Normal, 1=Spam     |
| Ling-Spam        | 0=ham, 1=spam  | 0=Normal, 1=Spam     |
| SpamAssassin     | 0=ham, 1=spam  | 0=Normal, 1=Spam     |
| phishing_email   | 0=legit, 1=bad | 0=Normal, 2=Fraud    |
| Nazario          | all fraud      | all 2=Fraud          |
| Nigerian_Fraud   | all fraud      | all 2=Fraud          |

### Nguyên Lý Kỹ Thuật

1. **Multi-Source Dataset**
   - Mỗi corpus có đặc điểm riêng
   - Kết hợp nhiều nguồn → model generalize tốt hơn
   - Tránh overfitting vào style của 1 corpus

2. **Label Standardization**
   - Unified labels (0/1/2) giúp train 1 model duy nhất
   - Label mapping flexible qua DatasetSpec
   - Validate để catch unmapped labels sớm

3. **Deduplication**
   - Email có thể xuất hiện ở nhiều corpus
   - Duplicate → model học bias
   - Dùng clean_text để detect (sau normalize)

4. **Class Balancing**
   - Imbalanced data → model bias về majority class
   - Downsampling về smallest class
   - Trade-off: mất data nhưng được fairness
   - Optional: cho phép chọn balanced vs full dataset

5. **Word-Count Dataset**
   - Format khác: term frequency thay vì raw text
   - Reconstruct text bằng repeat tokens
   - Ví dụ: {"money": 3} → "money money money"
   - Giữ được thông tin frequency

---

## MODULE 3: Email_Classification.py

### Mục đích
Training và prediction với Machine Learning.

### Workflow Training

```
┌─────────────────────────────────────────────────────────┐
│                   TRAINING WORKFLOW                      │
└─────────────────────────────────────────────────────────┘

┌──────────────────────┐
│ combined_emails.csv  │
│ ~134k rows           │
│ (or ~103k balanced)  │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Load Dataset         │
│ X = clean_text       │
│ y = labels (0/1/2)   │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│ Train/Test Split (Stratified)│
│                              │
│ Train: 80% (stratified)      │
│ Test:  20% (stratified)      │
│                              │
│ Stratified = giữ nguyên      │
│ class distribution           │
└──────────┬───────────────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                  BUILD ML PIPELINE                       │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 1: TF-IDF Vectorizer                                │
│                                                           │
│ Transform text → numerical vectors                        │
│                                                           │
│ Input:  ["free money", "meeting tomorrow"]               │
│                                                           │
│ Process:                                                  │
│ 1. Build vocabulary từ training data                     │
│    Vocab = {free, money, meeting, tomorrow, ...}         │
│                                                           │
│ 2. Calculate TF (Term Frequency)                         │
│    TF(term, doc) = count(term in doc) / total_terms      │
│                                                           │
│ 3. Calculate IDF (Inverse Document Frequency)            │
│    IDF(term) = log(total_docs / docs_containing_term)    │
│                                                           │
│ 4. TF-IDF = TF * IDF                                     │
│    - Từ xuất hiện nhiều trong doc → TF cao               │
│    - Từ xuất hiện nhiều docs → IDF thấp                  │
│    - Từ quan trọng & đặc trưng → TF-IDF cao              │
│                                                           │
│ Settings:                                                 │
│ - ngram_range=(1,2): unigrams + bigrams                  │
│   "free money" → ["free", "money", "free money"]         │
│ - min_df=2: bỏ từ xuất hiện < 2 docs (noise)             │
│ - stop_words='english': bỏ stopwords                     │
│ - max_features: cap vocabulary (nếu set)                 │
│                                                           │
│ Output: Sparse matrix (N_samples x N_features)           │
│         Ví dụ: (103000, 150000) - 150k features          │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 2: Chi² Feature Selection (Optional)                │
│                                                           │
│ Reduce features từ 150k → 2k (ví dụ)                     │
│                                                           │
│ Chi-Squared Test:                                         │
│ - Đo độ phụ thuộc giữa feature và label                  │
│ - χ² = Σ (Observed - Expected)² / Expected               │
│ - High χ² = feature strongly related to label            │
│                                                           │
│ Process:                                                  │
│ 1. Calculate χ² score cho mỗi feature                    │
│ 2. Rank features theo χ² giảm dần                        │
│ 3. Select top K features                                 │
│                                                           │
│ Benefits:                                                 │
│ - Giảm dimensionality → faster training                  │
│ - Reduce overfitting                                     │
│ - Keep most informative features                         │
│                                                           │
│ Output: Matrix (N_samples x K_features)                  │
│         Ví dụ: (103000, 2000)                            │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 3: Classifier                                        │
│                                                           │
│ OPTION A: Logistic Regression (default)                  │
│ ────────────────────────────────────────                 │
│ Linear model: P(class) = sigmoid(w·x + b)                │
│                                                           │
│ Multi-class strategy: One-vs-Rest                        │
│ - Train 3 binary classifiers:                            │
│   * Normal vs (Spam+Fraud)                               │
│   * Spam vs (Normal+Fraud)                               │
│   * Fraud vs (Normal+Spam)                               │
│                                                           │
│ Weights (coefficients):                                  │
│ - Positive coef → word pushes toward class               │
│ - Negative coef → word pushes away from class            │
│ - Large |coef| → strong influence                        │
│                                                           │
│ Ví dụ coefficients:                                      │
│ Spam:  {money: +2.3, free: +1.8, meeting: -1.2}          │
│ Fraud: {account: +2.7, verify: +2.1, password: +1.9}     │
│                                                           │
│ Prediction:                                              │
│ score_class = Σ(feature_value * coef) + bias             │
│ probability = softmax(scores)                            │
│                                                           │
│ Settings:                                                │
│ - max_iter=1000: đủ iterations cho convergence           │
│ - class_weight='balanced': auto-weight classes           │
│ - solver='lbfgs': optimizer tốt cho multi-class          │
│                                                           │
│ OPTION B: Random Forest                                  │
│ ────────────────────────────────────                     │
│ Ensemble of decision trees                               │
│                                                           │
│ Process:                                                 │
│ 1. Train 300 decision trees                              │
│ 2. Each tree trained on random subset                    │
│ 3. Each split considers random features                  │
│ 4. Prediction = majority vote of trees                   │
│                                                           │
│ Benefits:                                                │
│ - Non-linear decision boundaries                         │
│ - Robust to outliers                                     │
│ - Less interpretable than LogReg                         │
│                                                           │
│ Settings:                                                │
│ - n_estimators=300: 300 trees                            │
│ - class_weight='balanced'                                │
│ - n_jobs=-1: use all CPU cores                           │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│ Fit Pipeline         │
│ pipeline.fit(X_train,│
│              y_train)│
│                      │
│ - TF-IDF learns vocab│
│ - Chi² selects feats │
│ - Classifier learns  │
│   weights            │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Predict on Test Set  │
│ y_pred =             │
│   pipeline.predict(  │
│     X_test)          │
└──────────┬───────────┘
           │
           ▼
┌─────────────────────────────────────────────────────────┐
│                  EVALUATION METRICS                      │
└─────────────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ Confusion Matrix                                          │
│                                                           │
│              Predicted                                    │
│            Normal  Spam  Fraud                            │
│ Actual                                                    │
│ Normal     6467    283   127     (6877 total)            │
│ Spam        134   6561   182     (6877 total)            │
│ Fraud        89    493  6295     (6877 total)            │
│                                                           │
│ Diagonal = correct predictions                           │
│ Off-diagonal = misclassifications                        │
│                                                           │
│ Analysis:                                                │
│ - Normal: 6467/6877 = 94.0% correct                      │
│ - Spam: 6561/6877 = 95.4% correct                        │
│ - Fraud: 6295/6877 = 91.5% correct                       │
│                                                           │
│ Common mistakes:                                         │
│ - Spam ↔ Fraud confusion (493+182)                       │
│   (Both can contain urgency, money keywords)             │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ Per-Class Metrics                                         │
│                                                           │
│ PRECISION = TP / (TP + FP)                                │
│ "Trong dự đoán là X, bao nhiêu % thực sự là X?"          │
│                                                           │
│ Example: Spam Precision = 6561/(6561+283+493) = 89.4%    │
│ → 89.4% emails dự đoán là Spam thực sự là Spam           │
│                                                           │
│ RECALL = TP / (TP + FN)                                   │
│ "Trong thực tế là X, bao nhiêu % được phát hiện?"        │
│                                                           │
│ Example: Spam Recall = 6561/(134+6561+182) = 95.4%       │
│ → 95.4% Spam emails được phát hiện đúng                  │
│                                                           │
│ F1 = 2 * (Precision * Recall) / (Precision + Recall)     │
│ Harmonic mean, balance giữa precision và recall          │
│                                                           │
│ Results:                                                 │
│ Normal:  P=96.7%, R=94.0%, F1=95.3%                       │
│ Spam:    P=89.4%, R=95.4%, F1=92.3%                       │
│ Fraud:   P=95.3%, R=91.5%, F1=93.4%                       │
│                                                           │
│ MACRO AVERAGE = average across classes                   │
│ Treat all classes equally (good for imbalanced data)     │
│                                                           │
│ Accuracy: 93.7%                                          │
│ Macro P:  93.8%                                          │
│ Macro R:  93.7%                                          │
│ Macro F1: 93.7%                                          │
└──────────┬────────────────────────────────────────────────┘
           │
           ▼
┌──────────────────────┐
│ Save Model           │
│ - .joblib (fast)     │
│ - .pkl (compatible)  │
└──────────────────────┘
```

### Workflow Prediction

```
┌─────────────────────────────────────────────────────────┐
│                  PREDICTION WORKFLOW                     │
└─────────────────────────────────────────────────────────┘

┌──────────────────────┐
│ Load Trained Model   │
│ pipeline = joblib.   │
│   load('model.joblib')│
└──────────┬───────────┘
           │
           ▼
┌──────────────────────┐
│ Input Email Text     │
│ "Verify your PayPal  │
│  account now!"       │
└──────────┬───────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 1: TF-IDF Transform     │
│                              │
│ Use vocabulary learned       │
│ during training              │
│                              │
│ Text → TF-IDF vector         │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 2: Feature Selection    │
│                              │
│ Keep only selected features  │
│ (same as training)           │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 3: Classifier Predict   │
│                              │
│ Calculate scores for each    │
│ class using learned weights  │
│                              │
│ Fraud:  +5.2                 │
│ Spam:   +3.1                 │
│ Normal: -2.3                 │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 4: Softmax              │
│                              │
│ Convert scores → probs       │
│                              │
│ Fraud:  64.8%                │
│ Spam:   31.4%                │
│ Normal:  3.8%                │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────┐
│ STEP 5: Select Max           │
│                              │
│ Predicted Label: 2 (Fraud)   │
└──────────┬───────────────────┘
           │
           ▼
┌──────────────────────────────────────────────────────────┐
│ STEP 6: Interpretability (LogReg only)                   │
│                                                           │
│ Top Contributing Words:                                  │
│                                                           │
│ Fraud (64.8%):                                           │
│   - account (score: 1.053)                               │
│   - verify  (score: 0.892)                               │
│   - paypal  (score: 0.731)                               │
│                                                           │
│ Spam (31.4%):                                            │
│   - click   (score: 0.521)                               │
│   - now     (score: 0.412)                               │
│                                                           │
│ Normal (3.8%):                                           │
│   - (no significant positive contributions)              │
│                                                           │
│ Calculation:                                             │
│ For each word in email:                                  │
│   score = TF-IDF_value × coefficient                     │
│                                                           │
│ Example:                                                 │
│ "account" has TF-IDF = 0.8                               │
│ Fraud coef for "account" = +1.316                        │
│ score = 0.8 × 1.316 = 1.053                              │
└──────────────────────────────────────────────────────────┘
```

---

## NGUYÊN LÝ MACHINE LEARNING

### 1. TF-IDF (Term Frequency - Inverse Document Frequency)

**Vấn đề**: Làm sao convert text thành numbers?

**Giải pháp**: TF-IDF

**Chi tiết**:

```
Term Frequency (TF):
───────────────────
Đếm tần suất từ trong document

TF(word, doc) = count(word in doc) / total_words_in_doc

Example: "free money free"
TF("free") = 2/3 = 0.667
TF("money") = 1/3 = 0.333

Inverse Document Frequency (IDF):
─────────────────────────────────
Giảm trọng số của từ phổ biến

IDF(word) = log(total_docs / docs_containing_word)

Example: 1000 documents
- "the" appears in 990 docs → IDF = log(1000/990) = 0.01 (low)
- "viagra" appears in 10 docs → IDF = log(1000/10) = 2.0 (high)

TF-IDF:
──────
TF-IDF(word, doc) = TF(word, doc) × IDF(word)

Ý nghĩa:
- Từ xuất hiện nhiều trong doc này → TF cao
- Từ xuất hiện ít trong các docs khác → IDF cao
- → TF-IDF cao = từ quan trọng và đặc trưng cho doc này

Example:
Doc: "free money free"
Corpus: 1000 docs, "free" in 50 docs, "money" in 30 docs

TF-IDF("free") = (2/3) × log(1000/50) = 0.667 × 1.3 = 0.867
TF-IDF("money") = (1/3) × log(1000/30) = 0.333 × 1.52 = 0.506
```

### 2. N-grams

**Vấn đề**: Single words mất context

**Giải pháp**: N-grams (phrases)

```
Unigrams (1-gram): individual words
"Click here now" → ["click", "here", "now"]

Bigrams (2-gram): consecutive word pairs
"Click here now" → ["click here", "here now"]

Full with ngram_range=(1,2):
"Click here now" → ["click", "here", "now", "click here", "here now"]

Lợi ích:
- "free money" khác với "money free"
- "click here" là spam phrase
- "thank you" là normal phrase
- Bigrams capture context tốt hơn unigrams
```

### 3. Chi-Squared Feature Selection

**Vấn đề**: Too many features (150k) → slow, overfit

**Giải pháp**: Select top K most informative features

```
Chi-Squared Test:
────────────────
Đo độ phụ thuộc giữa feature và label

χ²(feature, label) = Σ (Observed - Expected)² / Expected

High χ² = feature strongly related to label

Example:
Word "viagra":
- Normal emails: observed=5, expected=500 → large difference
- Spam emails: observed=800, expected=300 → large difference
- χ² high → select this feature

Word "the":
- Appears equally in all classes
- Observed ≈ Expected
- χ² low → discard this feature

Process:
1. Calculate χ² for all 150k features
2. Rank features by χ² descending
3. Keep top 2k features
4. Discard rest

Benefits:
- Faster training (2k vs 150k)
- Less overfitting
- Keep most informative words
```

### 4. Class Balancing

**Vấn đề**: Imbalanced data

```
Original distribution:
Normal: 64.6k (48%)
Spam:   35.4k (26%)
Fraud:  34.4k (26%)

Without balancing:
Model learns: "Predict Normal most of the time"
→ High accuracy on Normal, poor on Spam/Fraud

Solutions:
─────────

A. Downsampling (what we do):
Random sample 34.4k from each class
→ Balanced: 34.4k / 34.4k / 34.4k
Pro: Perfect balance
Con: Lose data (từ 134k xuống 103k)

B. Class Weights (also applied):
class_weight='balanced' in sklearn
Automatically weight loss function:
weight(class) = n_samples / (n_classes × n_samples_class)

Example:
Normal: weight = 134k / (3 × 64.6k) = 0.69
Spam:   weight = 134k / (3 × 35.4k) = 1.26
Fraud:  weight = 134k / (3 × 34.4k) = 1.30

→ Errors on minority classes penalized more
→ Model pays more attention to Spam/Fraud
```

### 5. Stratified Split

**Vấn đề**: Random split có thể tạo imbalance giữa train/test

```
Dataset: 40% Normal, 30% Spam, 30% Fraud

Random split có thể tạo:
Train: 45% Normal, 28% Spam, 27% Fraud
Test:  30% Normal, 35% Spam, 35% Fraud
→ Different distributions!

Stratified split guarantees:
Train: 40% Normal, 30% Spam, 30% Fraud
Test:  40% Normal, 30% Spam, 30% Fraud
→ Same distributions!

Implementation:
train_test_split(X, y, stratify=y)
```

---

## COMMAND-LINE INTERFACE

### Training

```bash
# Basic training (default: LogReg, balanced, 20% test)
python Email_Classification.py

# Advanced training
python Email_Classification.py \
    --model rf \                    # Random Forest
    --max-features 150000 \         # Cap vocabulary at 150k
    --k-best 2000 \                 # Select top 2k features
    --test-size 0.2 \               # 20% test set
    --balance \                     # Balance classes
    --model-path models/my_model.joblib

# Training without balancing (use full dataset)
python Email_Classification.py --no-balance
```

### Prediction

```bash
# Basic prediction
python Email_Classification.py --predict "Click here to win money!"

# Output:
# Predicted label: 1 (Spam)
# Class probabilities:
#   Spam: 67.78%
#   Fraud: 30.07%
#   Normal: 2.16%
# Top contributing terms:
#   Spam (67.78% contribution): money (0.372), click (0.289), win (0.203)
#   Fraud (30.07% contribution): money (0.521)
```

### Dataset Preparation

```bash
# Build combined dataset (unbalanced)
python Extract_email.py

# Build balanced dataset
python Extract_email.py --balance

# Output:
# - Datasets/combined_emails.csv
# - Datasets/normal_emails.csv
# - Datasets/spam_emails.csv
# - Datasets/fraud_emails.csv
```

---

## PROGRAMMATIC API

### Basic Usage

```python
from Email_Classification import EmailClassifier

# Load trained model
clf = EmailClassifier()

# Predict
label, label_name = clf.predict("Verify your account now")
print(f"Label: {label}, Name: {label_name}")
# Output: Label: 2, Name: Fraud
```

### Detailed Prediction

```python
# Get full details
details = clf.predict_with_details("Click here for free money")

print(f"Label: {details['label_name']}")
# Output: Label: Spam

print("Probabilities:")
for name, prob in details['probabilities'].items():
    print(f"  {name}: {prob:.2f}%")
# Output:
#   Spam: 67.5%
#   Fraud: 28.3%
#   Normal: 4.2%

print("Top contributing words:")
for word, score in details['top_contributing_words']:
    print(f"  {word}: {score:.3f}")
# Output:
#   money: 0.521
#   free: 0.412
#   click: 0.289
```

### Custom Training

```python
from Email_Classification import TrainingConfig, train_model, save_model

# Custom configuration
config = TrainingConfig(
    test_size=0.3,           # 30% test
    balance=True,
    max_features=200000,     # Larger vocabulary
    k_best=5000,             # More features
    model='rf',              # Random Forest
    random_state=42
)

# Train
result = train_model(config)

# Save
save_model(result.pipeline)

# Check metrics
print(f"Accuracy: {result.accuracy:.4f}")
print(f"F1: {result.f1_macro:.4f}")
```

---

## PERFORMANCE METRICS EXPLAINED

### Confusion Matrix Analysis

```
              Predicted
           N      S      F
Actual N  6467   283    127    (6877)
       S   134  6561    182    (6877)
       F    89   493   6295    (6877)

Insights:
─────────

1. Normal emails (row 1):
   - 6467 correctly classified as Normal
   - 283 misclassified as Spam (4.1%)
   - 127 misclassified as Fraud (1.8%)
   → Normal classification is strong

2. Spam emails (row 2):
   - 6561 correctly classified as Spam
   - 134 misclassified as Normal (1.9%)
   - 182 misclassified as Fraud (2.6%)
   → Spam detection is excellent

3. Fraud emails (row 3):
   - 6295 correctly classified as Fraud
   - 89 misclassified as Normal (1.3%)
   - 493 misclassified as Spam (7.2%)
   → Fraud vs Spam confusion is main issue

4. Common confusion: Spam ↔ Fraud
   - Both contain urgency ("act now!")
   - Both mention money
   - Both have suspicious links
   → Need more discriminative features
```

### Precision vs Recall Trade-off

```
Precision: "When I say X, am I correct?"
Recall:    "Of all X, how many did I find?"

Example: Fraud Detection

High Precision, Low Recall:
─────────────────────────
P=98%, R=70%
→ When flag as Fraud, 98% are actually Fraud (few false alarms)
→ But only catch 70% of Frauds (miss 30%)
→ Good for: Minimizing false positives (don't annoy users)

High Recall, Low Precision:
──────────────────────────
P=75%, R=95%
→ Catch 95% of Frauds (very few escape)
→ But 25% of Fraud flags are false alarms
→ Good for: Catching threats (security-critical)

Balanced (our result):
─────────────────────
P=95.3%, R=91.5%, F1=93.4%
→ Good balance for general use
```

---

## TROUBLESHOOTING

### Issue 1: Low Accuracy on Specific Class

```
Symptom: Normal=95%, Spam=92%, Fraud=75%

Causes:
1. Class imbalance → use --balance
2. Insufficient Fraud training data
3. Fraud features overlap with other classes

Solutions:
1. Collect more Fraud examples
2. Feature engineering (domain-specific keywords)
3. Adjust class weights manually
4. Try different k_best values
```

### Issue 2: Overfitting (High Train, Low Test)

```
Symptom: Train accuracy=99%, Test accuracy=88%

Causes:
1. Too many features
2. Model too complex
3. Data leakage

Solutions:
1. Reduce max_features
2. Increase k_best (more feature selection)
3. Use simpler model (LogReg instead of RF)
4. Check for duplicates between train/test
```

### Issue 3: Model Bias

```
Symptom: Predicts mostly one class

Causes:
1. Imbalanced training data
2. class_weight not set

Solutions:
1. Use --balance flag
2. Verify class_weight='balanced' in model
3. Check class distribution in training data
```

---

## EXTENSIBILITY

### Adding New Dataset Source

```python
# In Extract_email.py

# 1. Add DatasetSpec
DATASET_SOURCES.append(
    DatasetSpec(
        'new_dataset.csv',
        label_map={0: 0, 1: 2},  # Adjust mapping
        text_column='email_text'  # If different
    )
)

# 2. Place CSV in Datasets/

# 3. Rebuild
python Extract_email.py --balance
```

### Adding New Feature Type

```python
# In Email_Classification.py

from sklearn.feature_extraction.text import CountVectorizer

def build_pipeline(...):
    steps = [
        ('tfidf', TfidfVectorizer(...)),
        
        # Add custom feature extractor
        ('custom_features', CustomFeatureExtractor()),
        
        ('chi2', SelectKBest(...)),
        ('clf', LogisticRegression(...))
    ]
    return Pipeline(steps)

class CustomFeatureExtractor:
    """Extract domain-specific features"""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        # Extract features like:
        # - Number of URLs
        # - Number of exclamation marks
        # - Presence of "urgent" keywords
        # - Suspicious email patterns
        ...
```

---

## BEST PRACTICES

1. **Always use stratified split** cho train/test
2. **Balance classes** khi có imbalance
3. **Save both .joblib và .pkl** cho compatibility
4. **Version your models** (date/metrics in filename)
5. **Monitor per-class metrics**, không chỉ overall accuracy
6. **Retrain periodically** với new data
7. **Use chi² selection** khi có nhiều features
8. **Document hyperparameters** khi deploy model
9. **Test on real-world emails** trước khi deploy
10. **Set up monitoring** cho model performance trong production

---

## KIẾN TRÚC FILES

```
Email_Classifying_System/
├── text_preprocessing.py          # Text cleaning utilities
├── Extract_email.py                # Dataset assembly
├── Email_Classification.py         # ML training/prediction
├── requirements.txt                # Python dependencies
├── README.md                       # User documentation
├── WORKFLOW_DOCUMENTATION.md       # This file (technical docs)
│
├── Datasets/                       # Email datasets
│   ├── Enron.csv
│   ├── CEAS_08.csv
│   ├── Ling.csv
│   ├── SpamAssasin.csv
│   ├── phishing_email.csv
│   ├── Nazario.csv
│   ├── Nigerian_Fraud.csv
│   ├── emails.csv                  # Word-count dataset
│   ├── combined_emails.csv         # Generated: unified dataset
│   ├── normal_emails.csv           # Generated: per-class
│   ├── spam_emails.csv             # Generated: per-class
│   └── fraud_emails.csv            # Generated: per-class
│
└── models/                         # Trained models
    ├── email_classifier.joblib     # Scikit-learn pipeline (fast)
    └── email_classifier.pkl        # Pickle format (compatible)
```

---

## TÓM TẮT WORKFLOW TOÀN HỆ THỐNG

```
1. DATA COLLECTION
   ↓
   Multiple CSV sources (Enron, CEAS, phishing, etc.)

2. DATA ASSEMBLY (Extract_email.py)
   ↓
   - Load each source
   - Normalize schema
   - Map labels to 0/1/2
   - Deduplicate
   - Balance classes
   ↓
   combined_emails.csv (~103k rows)

3. TEXT PREPROCESSING (text_preprocessing.py)
   ↓
   - HTML stripping
   - Tokenization
   - Lowercase
   - Remove stopwords
   - Lemmatization
   ↓
   Clean text tokens

4. MACHINE LEARNING (Email_Classification.py)
   ↓
   TRAINING:
   - Load dataset
   - Train/test split (stratified)
   - TF-IDF vectorization
   - Chi² feature selection
   - Train classifier (LogReg/RF)
   - Evaluate metrics
   - Save model
   ↓
   PREDICTION:
   - Load trained model
   - Transform input text (TF-IDF + Chi²)
   - Predict class
   - Show probabilities + top words
   ↓
   Output: Label (0/1/2) + confidence + explanation
```

---

Hệ thống này sử dụng các best practices trong ML:
- Data cleaning và normalization
- Multi-source dataset cho generalization
- Feature engineering (TF-IDF, n-grams)
- Feature selection (Chi²)
- Class balancing
- Stratified evaluation
- Model interpretability (coefficients analysis)
- Proper train/test separation

Kết quả: **93.7% accuracy** với good balance giữa các classes.

