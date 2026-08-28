# Relatório Consolidado — Spam & Phishing Detection

**UC:** Aprendizagem Automática para Cibersegurança (ISCTE-Sintra, 2025/2026)
**Docente:** Prof. João Pedro Pavia · **Estudante:** Ilie · Nº 112779

---

## Sumário executivo

Sistema de deteção de phishing/spam em três vistas (SMS · Enron · Combined),
pipeline MLOps completo: notebook reprodutível, API FastAPI, Dash frontend
e Model Cards. Classificador primário: **LinearSVC calibrado** sobre TF-IDF
+ 12 features heurísticas. Adversarial: 4 ataques × 3 vistas × {sem,com}
defesa. XAI: SHAP linear exacto (matematicamente = tfidf×coef) + Model Cards
conformes RGPD Art. 22.º.

## 1. Introdução

A indústria do cybercrime atingiu maturidade económica (FBI IC3: ~12 mil M USD
em perdas em 2023). Datasets: SMS Spam Collection (UCI, 5572 msgs, 2011) e
Enron Spam (~33700 emails, 1999-2002). Diferem em comprimento, ruído
estrutural e desbalanceamento. Tese: um modelo bem desenhado deve generalizar
do telegráfico para o longo.

## 2. Metodologia

- **3 vistas consistentes** em todas as análises (SMS, Enron, Combined).
- **Feature engineering documentado**: removemos 4 features ligadas a
  HTML/URLs por sobreposição com TF-IDF e vulnerabilidade adversarial.
  Conjunto final = 12 heurísticas em `preprocess.py`.
- **Modelos**: ComplementNB, LinearSVC calibrado (Platt), LogisticRegression,
  IsolationForest (filtro OOD), LSTM Bidir.
- **Avaliação**: holdout 80/20 estratificado, threshold por F-β=2.

## 3. Resultados

LinearSVC bate NB em PR-AUC nas 3 vistas. Combined mantém F1 ≥ 0.94 em ambos
os sub-domínios — evidência de generalização. LSTM empata em PR-AUC mas custo
~30× superior. Whitespace injection é o ataque mais devastador
transversalmente; NFKD+collapse recupera ≥ 30 pp de success rate.

## 4. Discussão crítica

LinearSVC é a escolha racional para SOC: bate LSTM em custo/explicabilidade,
empata em PR-AUC, e oferece SHAP exacto. Vulnerabilidade ao whitespace
injection é estrutural ao bag-of-words. Mitigação por NFKD é barata mas tem
ponto de saturação — para robustez intrínseca requer adversarial training
ou modelos sub-word.

## 5. Limitações

(1) Idioma único (inglês); (2) Enron é 1999-2002; (3) Não cobre phishing
por imagem/QR; (4) Calibração Platt distorce probabilidades.

## 6. Conclusões e trabalho futuro

Sistema entregue end-to-end. Futuro: adversarial training (Madry 2018),
modelos sub-word (BPE), features comportamentais (SPF/DKIM/DMARC),
active learning, multilingue (DistilBERT), federated learning.

## 7. Bibliografia (IEEE)

[1] S. M. Lundberg, S.-I. Lee, "A Unified Approach to Interpreting Model Predictions," *NeurIPS*, 2017.
[2] M. T. Ribeiro et al., "Why Should I Trust You?: Explaining the Predictions of Any Classifier," *KDD*, 2016.
[3] European Parliament, "Regulation (EU) 2016/679 (GDPR)," *OJEU*, 2016. Art. 22.
[4] M. Mitchell et al., "Model Cards for Model Reporting," *FAT\**, 2019.
[5] I. J. Goodfellow et al., "Explaining and Harnessing Adversarial Examples," *ICLR*, 2015.
[6] A. Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks," *ICLR*, 2018.
[7] J. Li et al., "TextBugger," *NDSS*, 2019.
[8] T. Almeida et al., "Contributions to the Study of SMS Spam Filtering," *DocEng*, 2011.
[9] B. Klimt, Y. Yang, "The Enron Corpus," *ECML*, 2004.
[10] F. T. Liu et al., "Isolation Forest," *IEEE ICDM*, 2008.

## 8. Execução

```bash
# Notebook
jupyter nbconvert --to notebook --execute Proj_Final_AAC_v4_fusion_v2_fixed.ipynb \
                  --output Proj_Final_AAC_v4_fusion_v2_fixed.ipynb \
                  --ExecutePreprocessor.timeout=2400

# API
cd api && uvicorn app:app --port 8000

# Dash
cd dash_app && python app.py    # http://127.0.0.1:8050
```
