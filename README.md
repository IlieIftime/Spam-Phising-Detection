# Spam & Phishing Detection

A binary classification project for **ham vs spam/phishing** messages across two text domains:
- **SMS** (`spam.csv`)
- **Corporate email from the Enron dataset** (`enron_spam.csv`)

The main implementation is contained in the **[`Proj_Final_VFINAL1.ipynb`](./Proj_Final_VFINAL1.ipynb)** notebook, with shared preprocessing in **[`preprocess.py`](./preprocess.py)** and a narrative summary in **[`report_consolidated.md`](./report_consolidated.md)**.

## Objective

Build and compare spam/phishing detection pipelines using:
- text representation with TF-IDF;
- complementary heuristic features;
- comparisons between classical models and an LSTM;
- explainability analysis with SHAP and adversarial robustness testing.

## Data and Task Definition

- **Task:** supervised binary classification (`label`: `ham`/`spam`).
- **Evaluated views:** `sms`, `enron`, and `combined`.
- The notebook loads `data/spam.csv` and `data/enron_spam.csv`.
- In the current repository state, the raw CSV files have not been extracted into `data/`; they are stored in **`data.7z`**, together with derived artifacts in `data/`.

## Methodology (Pipeline Summary)

1. Domain-specific text cleaning (`clean_sms`, `clean_email`).
2. TF-IDF extraction from cleaned text.
3. Extraction of heuristic features from raw text.
4. Concatenation of TF-IDF and heuristic features (`hstack`) for classical models.
5. Training and evaluation by view (`sms`, `enron`, `combined`) using an 80/20 stratified split.
6. Threshold selection using **F-beta (β=2)** based on the precision-recall curve.
7. Model comparison, XAI analysis, and adversarial testing.

## Preprocessing and 12 Heuristic Features

`preprocess.py` defines the final set of **12** features:

1. `char_count`
2. `word_count`
3. `avg_word_len`
4. `num_digits`
5. `has_currency`
6. `has_unsubscribe`
7. `ratio_stopwords`
8. `punctuation_ratio`
9. `entropy_text`
10. `has_reply_marker`
11. `has_signature_block`
12. `has_shortcode`

The module itself documents the removal of HTML/URL-based features because of their redundancy with TF-IDF and limited adversarial robustness.

## Models and Evaluation Protocol

The models trained in the notebook are:
- **ComplementNB**
- **LinearSVC** (calibrated with `CalibratedClassifierCV`)
- **LogisticRegression**
- **Bidirectional LSTM**
- **IsolationForest** (OOD analysis/filtering)

Recorded metrics include PR-AUC, ROC-AUC, F1, F-beta, precision, recall, confusion matrix, and threshold.

## Verified Results (Measured)

Source: [`tables/summary_classical.csv`](./tables/summary_classical.csv).

- **SVC (calibrated LinearSVC)ต่อ**
  - SMS: PR-AUC `0.9754`, F1 `0.9317`
  - Enron: PR-AUC `0.9981`, F1 `0.9943`
  - Combined: PR-AUC `0.9987`, F1 `0.9913`
- **LR** achieves the highest PR-AUC/F1 values in the table for Enron and Combined (`1.0000` and `0.9974`, respectively).
- **LSTM** delivers competitive performance, with higher computational cost, as discussed in the notebook/report.

### Qualitative Conclusions (Not Isolated Metrics)

- The pipeline generalizes well across the two domains when trained and evaluated on the `combined` view as well.
- Heuristic features and TF-IDF are used complementarily in the classical pipeline.

## Explainability (XAI)

SHAP-based explainability is implemented in the notebook, with artifacts in [`xai/`](./xai):
- `shap_feature_importance_LinearSVC_*.csv`
- `svc_shap_comparison.csv`
- `top15_tokens_*.csv` and `top15_tokens_*.png`

## Adversarial Robustness

The attacks implemented in the notebook are `char_substitution`, `whitespace_injection`, `synonym_replacement`, and `textfooler_lite`, with an `NFKD+collapse` defense variant.

Aggregated results are available in:

- [`adversarial/adversarial_summary_by_source.csv`](./adversarial/adversarial_summary_by_source.csv)
- [`adversarial/vulnerability_score_by_source.csv`](./adversarial/vulnerability_score_by_source.csv)

Measured examples:
- `sms + whitespace_injection (without defense)`: `success_rate_pct = 0.9434`
- `enron + char_substitution (without defense)`: `success_rate_pct = 19.4320`

## Repository Structure

```text
.
├── Proj_Final_VFINAL1.ipynb
├── preprocess.py
├── report_consolidated.md
├── README.md
├── data.7z
├── data/
│   ├── baseline.json
│   └── features_heuristic_*.csv
├── tables/
│   └── summary_classical.csv
├── models/
│   ├── *_model.joblib / *_vectorizer.joblib / *_config.joblib
│   └── lstm_*.keras
├── xai/
├── adversarial/
├── eda/
└── dashboard/
    └── dashboard_cybersec_spam.html
```

## How to Reproduce (Current State)

> There is no `requirements.txt`, `pyproject.toml`, or `environment.yml` in the repository. Therefore, the steps below are conservative and based on the notebook.

1. **Create a virtual environment** (Python 3.10+ recommended).
2. **Install the dependencies** used in the notebook:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly nltk joblib shap tensorflow
```

3. **Make sure the raw data is available**:
   - extract `data.7z` to obtain `data/spam.csv` and `data/enron_spam.csv`;
   - if they already exist, confirm that they are located at the paths expected by the notebook.

4. **Run the notebook**:

```bash
jupyter notebook Proj_Final_VFINAL1.ipynb
```

or:

```bash
jupyter nbconvert --to notebook --execute Proj_Final_VFINAL1.ipynb --output Proj_Final_VFINAL1.executed.ipynb --ExecutePreprocessor.timeout=2400
```

## Application API/Dashboard

`report_consolidated.md` refers to execution of `api/` and `dash_app/`, but those directories are **not present** in the current repository state.

The available artifact that can be opened directly is:
- [`dashboard/dashboard_cybersec_spam.html`](./dashboard/dashboard_cybersec_spam.html)

## Limitations and Privacy

Observable limitations in the current state include:
- the absence of a versioned dependency file, which limits environmental reproducibility;
- datasets focused on specific languages/domains (SMS and Enron);
- robustness that varies by adversarial attack type and domain.

Privacy/GDPR:
- the repository discusses GDPR as a design objective and methodological framework;
- this documentation **does not** claim formal certification or automatic legal compliance in production.

## Future Work (Aligned with the Report)

- adversarial training and/or subword tokenization;
- multilingual extension;
- integration of additional signals, such as email authentication;
- complete environment packaging with a version-pinned dependency file.
