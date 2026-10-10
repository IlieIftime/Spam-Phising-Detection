# Spam & Phishing Detection

Binary classification of **ham vs spam/phishing** messages across two text domains, with classical models, an LSTM, explainability (SHAP) and adversarial robustness testing.

*Course project — Machine Learning for Cybersecurity, ISCTE-Sintra, 2025/2026.*
*Authors: Ilie Iftime and team members [add names].*

- **Domains:** SMS (`spam.csv`) and corporate email from the Enron dataset (`enron_spam.csv`)
- **Evaluated views:** `sms`, `enron`, `combined`
- **Main implementation:** [`Proj_Final_VFINAL1.ipynb`](./Proj_Final_VFINAL1.ipynb), with shared preprocessing in [`preprocess.py`](./preprocess.py)
- **Primary classifier:** calibrated LinearSVC on TF-IDF + 12 heuristic features

> **Scope.** This repository contains the reproducible study notebook, its outputs and a static HTML dashboard. It does **not** include a deployed API or web application.

## Contents

1. [Motivation](#motivation)
2. [Data](#data)
3. [Methodology](#methodology)
4. [Heuristic features](#heuristic-features)
5. [Models and evaluation protocol](#models-and-evaluation-protocol)
6. [Results](#results)
7. [Explainability (XAI)](#explainability-xai)
8. [Adversarial robustness](#adversarial-robustness)
9. [Discussion](#discussion)
10. [Limitations](#limitations)
11. [Future work](#future-work)
12. [Repository structure](#repository-structure)
13. [How to reproduce](#how-to-reproduce)
14. [Privacy and GDPR](#privacy-and-gdpr)
15. [References](#references)

## Motivation

Cybercrime has reached economic maturity (the FBI IC3 reported roughly 12 billion USD in losses in 2023). The two datasets differ in message length, structural noise and class imbalance. The working thesis is that a well-designed model should generalise from short, telegraphic SMS to long corporate email.

## Data

| Source | Description |
|---|---|
| SMS Spam Collection (UCI) | 5,572 messages, 2011 [8] |
| Enron Spam | ~33,700 emails, 1999–2002 [9] |

- **Task:** supervised binary classification (`label`: `ham` / `spam`).
- The notebook loads `data/spam.csv` and `data/enron_spam.csv`.
- In the current repository state the raw CSV files are stored in **`data.7z`** (together with derived artifacts); extract it before running the notebook.

## Methodology

1. Domain-specific cleaning (`clean_sms`, `clean_email`).
2. TF-IDF extraction from the cleaned text.
3. Heuristic features extracted from the raw text.
4. TF-IDF and heuristic features concatenated (`hstack`) for the classical models.
5. Training and evaluation per view (`sms`, `enron`, `combined`) with a stratified 80/20 hold-out split.
6. Decision threshold selected with **F-beta (β = 2)** on the precision-recall curve.
7. Model comparison, XAI analysis and adversarial testing (4 attacks × 3 views × with/without defence).

## Heuristic features

`preprocess.py` defines the final set of **12** features:

`char_count`, `word_count`, `avg_word_len`, `num_digits`, `has_currency`, `has_unsubscribe`, `ratio_stopwords`, `punctuation_ratio`, `entropy_text`, `has_reply_marker`, `has_signature_block`, `has_shortcode`.

Four HTML/URL-based features were removed because they overlap with TF-IDF and are vulnerable to adversarial manipulation (documented in `preprocess.py`).

## Models and evaluation protocol

- **ComplementNB**
- **LinearSVC**, calibrated with `CalibratedClassifierCV` (Platt scaling)
- **Logistic Regression**
- **Bidirectional LSTM**
- **Isolation Forest** (OOD analysis / filtering)

Recorded metrics: PR-AUC, ROC-AUC, F1, F-beta, precision, recall, confusion matrix and decision threshold.

## Results

Source: [`tables/summary_classical.csv`](./tables/summary_classical.csv).

**Calibrated LinearSVC**

| View | PR-AUC | F1 |
|---|---|---|
| SMS | 0.9754 | 0.9317 |
| Enron | 0.9981 | 0.9943 |
| Combined | 0.9987 | 0.9913 |

- Logistic Regression reaches the highest PR-AUC/F1 values in the table for Enron and Combined (1.0000 and 0.9974, respectively).
- The LSTM is competitive in PR-AUC at a higher computational cost, as discussed in the notebook.
- Heuristic features and TF-IDF are used complementarily in the classical pipeline.
- Training on the `combined` view also performs well on both sub-domains, which supports cross-domain generalisation.

## Explainability (XAI)

SHAP is implemented in the notebook; for the linear model the SHAP values are exact (equivalent to TF-IDF value × coefficient). Artifacts are in [`xai/`](./xai):

- `shap_feature_importance_LinearSVC_*.csv`
- `svc_shap_comparison.csv`
- `top15_tokens_*.csv` and `top15_tokens_*.png`

## Adversarial robustness

Attacks implemented in the notebook: `char_substitution`, `whitespace_injection`, `synonym_replacement` and `textfooler_lite`, evaluated with and without an `NFKD+collapse` normalisation defence.

Aggregated results:

- [`adversarial/adversarial_summary_by_source.csv`](./adversarial/adversarial_summary_by_source.csv)
- [`adversarial/vulnerability_score_by_source.csv`](./adversarial/vulnerability_score_by_source.csv)

Measured examples (without defence):

- `sms + whitespace_injection`: `success_rate_pct = 0.9434`
- `enron + char_substitution`: `success_rate_pct = 19.4320`

## Discussion

LinearSVC is a rational choice for a security-operations setting: it is cheap, offers exact SHAP explanations and matches the LSTM in PR-AUC, although Logistic Regression scores marginally higher on some views. Vulnerability to whitespace/character perturbations is structural to bag-of-words representations. Unicode normalisation (NFKD + collapse) is a cheap mitigation but saturates; intrinsic robustness would require adversarial training or sub-word models.

## Limitations

1. Single language (English).
2. The Enron corpus dates from 1999–2002.
3. No coverage of image-based or QR-code phishing.
4. Platt calibration can distort probability estimates.
5. No versioned dependency file in the repository (limits environmental reproducibility).
6. Datasets are specific to SMS and corporate email.
7. Robustness varies by attack type and domain.

## Future work

- Adversarial training [6] and/or sub-word tokenisation (BPE).
- Behavioural and authentication signals (SPF/DKIM/DMARC).
- Active learning.
- Multilingual extension (e.g. DistilBERT).
- Model Cards [4] and a packaged, version-pinned environment.

## Repository structure

```text
.
├── Proj_Final_VFINAL1.ipynb
├── preprocess.py
├── report_consolidated.md      # superseded by this README
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

The only directly viewable application artifact is the static dashboard [`dashboard/dashboard_cybersec_spam.html`](./dashboard/dashboard_cybersec_spam.html).

## How to reproduce

> There is no `requirements.txt`, `pyproject.toml` or `environment.yml` yet, so the steps below are based on the notebook's imports.

1. Create a virtual environment (Python 3.10+ recommended).
2. Install the dependencies:

   ```bash
   pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly nltk joblib shap tensorflow
   ```

3. Extract `data.7z` to obtain `data/spam.csv` and `data/enron_spam.csv` (or confirm they exist at the paths the notebook expects).
4. Run the notebook:

   ```bash
   jupyter notebook Proj_Final_VFINAL1.ipynb
   ```

   or execute it headlessly:

   ```bash
   jupyter nbconvert --to notebook --execute Proj_Final_VFINAL1.ipynb \
       --output Proj_Final_VFINAL1.executed.ipynb --ExecutePreprocessor.timeout=2400
   ```

## Privacy and GDPR

The project treats GDPR (notably Art. 22 on automated decision-making [3]) as a design and methodological framework. This documentation does **not** claim formal certification or automatic legal compliance in production.

## References

1. S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
2. M. T. Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier," *KDD*, 2016.
3. European Parliament, "Regulation (EU) 2016/679 (GDPR)," *OJEU*, 2016, Art. 22.
4. M. Mitchell et al., "Model Cards for Model Reporting," *FAT\**, 2019.
5. I. J. Goodfellow et al., "Explaining and Harnessing Adversarial Examples," *ICLR*, 2015.
6. A. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," *ICLR*, 2018.
7. J. Li et al., "TextBugger," *NDSS*, 2019.
8. T. Almeida et al., "Contributions to the Study of SMS Spam Filtering," *DocEng*, 2011.
9. B. Klimt, Y. Yang, "The Enron Corpus," *ECML*, 2004.
10. F. T. Liu et al., "Isolation Forest," *IEEE ICDM*, 2008.
