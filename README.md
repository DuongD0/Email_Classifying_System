# Email Classification

This project bundles a trainable email-classification pipeline that flags messages as `Normal`, `Spam`, or `Fraud`. The workflow cleans and combines multiple public corpora (e.g. CEAS, Enron, Nazario, Nigerian fraud/phishing) into a single dataset, applies text normalisation/lemmatisation, and trains either a class-balanced logistic regression baseline or a chi-squared-filtered random forest. Both the preprocessing steps and the model are persisted together so predictions stay aligned with training-time transformations.

## Technologies
- **Python 3.9+**: glue language for the entire pipeline.
- **pandas & numpy**: CSV ingestion, dataset merges, deduplication, and numeric feature handling in `Extract_email.py`.
- **nltk**: stopwords and WordNet lemmatisation (downloaded automatically in `text_preprocessing.py`).
- **scikit-learn**:
  - `TfidfVectorizer`, `SelectKBest(chi2)`, `LogisticRegression`, `RandomForestClassifier`, and evaluation utilities (`classification_report`, `confusion_matrix`, `precision_recall_fscore_support`) in `Email_Classification.py`.
  - `train_test_split` and `Pipeline` stitch the preprocessing + model together for consistent training/prediction flows.
- **joblib & pickle**: persist the fitted pipeline to `models/email_classifier.joblib` and `models/email_classifier.pkl`.
- **argparse**: exposes CLI controls for both training (`Email_Classification.py`) and dataset generation (`Extract_email.py`).
- **Git LFS**: tracks multi-hundred-MB CSV corpora and model artefacts without bloating Git history.

## Project Structure
- `Extract_email.py`: assembles the master dataset by reading every CSV in `Datasets`, normalising the text with `text_preprocessing.clean_email_text`, deduplicating messages, and (optionally) balancing classes before exporting:
  - `Datasets/combined_emails.csv` (full corpus)
  - `Datasets/normal_emails.csv`, `spam_emails.csv`, `fraud_emails.csv` (per-label splits)
- `text_preprocessing.py`: reusable cleaning helpers (HTML stripping, token filtering, lemmatisation, stopword removal).
- `Email_Classification.py`: training/prediction CLI that builds a TF-IDF → optional chi² feature selector → classifier pipeline, evaluates it, and persists artefacts (`models/email_classifier.joblib` + `.pkl`).
- `models/`: saved pipelines ready for reuse.
- `Datasets/`: source corpora plus generated CSV outputs.

## Dataset Snapshot
- **Sources**: CEAS08, Enron, Ling-Spam, SpamAssassin, phishing corpora (Nazario, Nigerian fraud, dedicated phishing CSV), and a word-count dataset reconstructed into plain text.
- **Current size (unbalanced)**: ~134k emails (`Normal`≈64.6k, `Spam`≈35.4k, `Fraud`≈34.4k). Duplicate `clean_text` rows are removed when merging.
- **Balanced export**: `--balance` downsamples each label to the smallest class (currently 34,385 per class, ~103k rows total).
- All CSV outputs are tracked via Git LFS to keep repository clones lightweight.

## Training & Evaluation
```bash
python Email_Classification.py \  # default: logistic regression, class-balanced
  --test-size 0.2 \
  --max-features 150000 \
  --k-best 2000 \
  --model logreg|rf
```

Key options:
- `--balance` / `--no-balance`: choose balanced vs raw label distribution before train/test split.
- `--max-features`: cap TF-IDF vocabulary size (default: unlimited bigrams with `min_df=2`).
- `--k-best`: select top-k features using chi-squared statistics.
- `--model`: `logreg` (balanced logistic regression, default) or `rf` (balanced random forest, 300 estimators).
- `--model-path` / `--pickle-path`: customise persistence destinations.

### Evaluation Metrics
Running `python Email_Classification.py --max-features 150000 --k-best 2000 --model logreg` on the balanced dataset (20% hold-out) produced:
- **Accuracy**: share of total predictions that match the ground-truth label.
- **Macro Precision**: average of class-level precision (TP / (TP + FP)), giving each class equal weight.
- **Macro Recall**: average of class-level recall (TP / (TP + FN)), indicating how completely each class is recovered.
- **Macro F1**: harmonic mean of macro precision and macro recall, balancing false positives and false negatives.
- **Per-class metrics**: precision/recall/F1/support reported separately for `Normal`, `Spam`, and `Fraud` to highlight class-specific behaviour.
- **Confusion matrix**: table of true labels (rows) vs predicted labels (columns) showing misclassification paths.

Latest results:
- **Accuracy**: 0.9366
- **Macro Precision**: 0.9380
- **Macro Recall**: 0.9366
- **Macro F1**: 0.9368
- **Per-class metrics**
  - Normal: Precision=0.9667, Recall=0.9404, F1=0.9533, Support=6,877
  - Spam: Precision=0.8942, Recall=0.9540, F1=0.9232, Support=6,877
  - Fraud: Precision=0.9532, Recall=0.9154, F1=0.9339, Support=6,877
- **Confusion matrix (rows=true, cols=pred)**

```
        Normal  Spam  Fraud
Normal    6467   283    127
Spam       134  6561    182
Fraud       89   493   6295
```

### Prediction Helpers
```bash
python Email_Classification.py --predict "Verify your bank account now"
```
- Loads the latest persisted pipeline and prints the predicted label, class probabilities, and top contributing tokens per class (when the classifier exposes coefficients).

Example CLI runs with the trained logistic-regression model:
```
$ python Email_Classification.py --predict "Save money and give us your credit card now!"
Predicted label: 1 (Spam)
Class probabilities:
  Spam: 67.78%
  Fraud: 30.07%
  Normal: 2.16%
Top contributing terms:
  Spam (29.62% contribution): money (0.372), save (0.130)
  Fraud (70.38% contribution): money (1.192)

$ python Email_Classification.py --predict "Attention! Your Paypal account will close soon!"
Predicted label: 2 (Fraud)
Class probabilities:
  Fraud: 64.83%
  Spam: 31.39%
  Normal: 3.78%
Top contributing terms:
  Fraud (100.00% contribution): account (1.053), soon (0.608)
```

Programmatic use:
```python
from Email_Classification import EmailClassifier

clf = EmailClassifier()
label, label_name = clf.predict("Schedule our standup meeting")
print(label_name)
```

## Data Refresh Workflow
1. `python Extract_email.py [--balance]` to rebuild the combined dataset and per-label CSVs.
2. `python Email_Classification.py [training flags]` to fit a new pipeline and overwrite saved artefacts.
3. Consume `models/email_classifier.joblib` or `models/email_classifier.pkl` in your application for instant predictions.

## Environment & Dependencies
Install requirements (Python 3.9+):
```bash
pip install -r requirements.txt
```
The scripts auto-download NLTK stopwords and WordNet on first run. If that fails, manually execute:
```bash
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet')"
```

## Tips
- Delete the files inside `models/` to force a clean retrain.
- Use `--k-best` with the random-forest mode to focus on the most informative tokens.
- When running in production, prefer the `.joblib` artefact for fastest reloads; keep the `.pkl` for interoperability with non-Joblib tooling.
- Combined dataset size (~134k rows) can stress memory on low-resource machines—consider running with `--no-balance` and a smaller `--max-features` if needed.
