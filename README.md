# Spam & Phishing Detection

Projeto de classificação binária de mensagens **ham vs spam/phishing** em dois domínios de texto:
- **SMS** (`spam.csv`)
- **email corporativo Enron** (`enron_spam.csv`)

A implementação principal está no notebook **[`Proj_Final_VFINAL1.ipynb`](./Proj_Final_VFINAL1.ipynb)**, com pré-processamento partilhado em **[`preprocess.py`](./preprocess.py)** e síntese narrativa em **[`report_consolidated.md`](./report_consolidated.md)**.

## Objetivo

Construir e comparar pipelines de deteção de spam/phishing com:
- representação textual (TF-IDF);
- features heurísticas complementares;
- comparação entre modelos clássicos e LSTM;
- análise de explicabilidade (SHAP) e robustez adversarial.

## Dados e definição da tarefa

- **Tarefa:** classificação supervisionada binária (`label`: `ham`/`spam`).
- **Vistas avaliadas:** `sms`, `enron` e `combined`.
- O notebook carrega `data/spam.csv` e `data/enron_spam.csv`.
- No estado atual do repositório, os CSV brutos não estão extraídos em `data/`; estão no arquivo **`data.7z`** (juntamente com artefactos derivados em `data/`).

## Metodologia (resumo do pipeline)

1. Limpeza textual específica por domínio (`clean_sms`, `clean_email`).
2. Extração TF-IDF do texto limpo.
3. Extração de features heurísticas no texto bruto.
4. Concatenação TF-IDF + heurísticas (`hstack`) para modelos clássicos.
5. Treino/avaliação por vista (`sms`, `enron`, `combined`) com divisão estratificada 80/20.
6. Seleção de threshold por **F-β (β=2)** com base na curva precision-recall.
7. Comparação de modelos, XAI (SHAP) e testes adversariais.

## Pré-processamento e 12 features heurísticas

`preprocess.py` define o conjunto final de **12** features:

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

O próprio módulo documenta a remoção de features baseadas em HTML/URL por redundância com TF-IDF e baixa robustez adversarial.

## Modelos e protocolo de avaliação

Modelos treinados no notebook:
- **ComplementNB**
- **LinearSVC** (calibrado com `CalibratedClassifierCV`)
- **LogisticRegression**
- **Bidirectional LSTM**
- **IsolationForest** (análise/filtro OOD)

Métricas registadas: PR-AUC, ROC-AUC, F1, F-β, precisão, recall, matriz de confusão e threshold.

## Resultados verificados (medidos)

Fonte: [`tables/summary_classical.csv`](./tables/summary_classical.csv).

- **SVC (LinearSVC calibrado)**
  - SMS: PR-AUC `0.9754`, F1 `0.9317`
  - Enron: PR-AUC `0.9981`, F1 `0.9943`
  - Combined: PR-AUC `0.9987`, F1 `0.9913`
- **LR** atinge PR-AUC/F1 máximos na tabela em Enron e Combined (`1.0000` e `0.9974`, respetivamente).
- **LSTM** apresenta desempenho competitivo, com maior custo computacional (conforme discussão no notebook/relatório).

### Conclusões qualitativas (não métricas isoladas)

- O pipeline generaliza bem entre os dois domínios quando treinado/avaliado também em `combined`.
- Features heurísticas e TF-IDF são usadas de forma complementar no pipeline clássico.

## Explicabilidade (XAI)

Implementada com SHAP no notebook, com artefactos em [`xai/`](./xai):
- `shap_feature_importance_LinearSVC_*.csv`
- `svc_shap_comparison.csv`
- `top15_tokens_*.csv` e `top15_tokens_*.png`

## Robustez adversarial

Ataques implementados no notebook: `char_substitution`, `whitespace_injection`, `synonym_replacement`, `textfooler_lite`, com variante de defesa `NFKD+collapse`.

Resultados agregados em:
- [`adversarial/adversarial_summary_by_source.csv`](./adversarial/adversarial_summary_by_source.csv)
- [`adversarial/vulnerability_score_by_source.csv`](./adversarial/vulnerability_score_by_source.csv)

Exemplos medidos:
- `sms + whitespace_injection (sem defesa)`: `success_rate_pct = 0.9434`
- `enron + char_substitution (sem defesa)`: `success_rate_pct = 19.4320`

## Estrutura do repositório

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

## Como reproduzir (estado atual)

> Não existe `requirements.txt`, `pyproject.toml` nem `environment.yml` no repositório. Por isso, os passos abaixo são conservadores e baseados no notebook.

1. **Criar ambiente virtual** (Python 3.10+ recomendado).
2. **Instalar dependências** usadas no notebook:

```bash
pip install numpy pandas scipy scikit-learn matplotlib seaborn plotly nltk joblib shap tensorflow
```

3. **Garantir dados brutos**:
   - extrair `data.7z` para obter `data/spam.csv` e `data/enron_spam.csv`;
   - se já existirem, confirmar os caminhos esperados pelo notebook.

4. **Executar o notebook**:

```bash
jupyter notebook Proj_Final_VFINAL1.ipynb
```

ou

```bash
jupyter nbconvert --to notebook --execute Proj_Final_VFINAL1.ipynb --output Proj_Final_VFINAL1.executed.ipynb --ExecutePreprocessor.timeout=2400
```

## Sobre API/Dashboard aplicacional

`report_consolidated.md` refere execução de `api/` e `dash_app/`, mas essas pastas **não estão presentes** neste estado do repositório.

O que existe e pode ser aberto diretamente é:
- [`dashboard/dashboard_cybersec_spam.html`](./dashboard/dashboard_cybersec_spam.html)

## Limitações e privacidade

Limitações observáveis no estado atual:
- ausência de ficheiro de dependências versionado (reprodutibilidade ambiental limitada);
- datasets de uma língua/domínio específicos (SMS + Enron);
- robustez variável a perturbações adversariais por tipo de ataque/domínio.

Privacidade/GDPR:
- o repositório discute RGPD como objetivo de desenho e enquadramento metodológico;
- esta documentação **não** assume certificação formal nem conformidade legal automática em produção.

## Trabalho futuro (alinhado com o relatório)

- treino adversarial e/ou tokenização subword;
- extensão multilingue;
- integração de sinais adicionais (ex.: autenticação de email);
- empacotamento completo do ambiente (ficheiro de dependências com versões).
