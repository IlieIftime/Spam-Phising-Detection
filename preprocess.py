"""
preprocess.py — Pipeline de pré-processamento partilhado
=========================================================
Este módulo é importado pelo notebook, pela API e pelo Dash.
Tudo o que aqui está deve ser idêntico nos três contextos —
caso contrário a predição diverge entre notebook e app.

Decisão metodológica documentada
--------------------------------
Removemos as features ligadas a HTML e a URLs (`has_html_tags`,
`num_urls`, `url_count`, `num_emails`) do conjunto final por dois motivos:

1. **Sobreposição com TF-IDF.** O placeholder `url` e `emailaddr`
   substitui os links/emails no texto limpo (`clean_*`), pelo que
   o TF-IDF já capta esta informação como token. Manter como feature
   numérica adicional cria redundância (correlação ρ > 0.9 com o token).

2. **Vulnerabilidade adversarial.** `has_html_tags` é trivialmente
   contornado por re-encoding ou Markdown; URLs por shorteners ou
   redireccionamentos. Heurísticas que custam pouco a evadir oferecem
   ganho marginal e dão falsa segurança.

Mantemos features que **não** sobrepõem o TF-IDF:
char/word counts, ratios, entropy, currency (símbolo único `£$€¥`),
unsubscribe, reply markers, signature blocks, shortcodes, phone numbers.
"""
from __future__ import annotations

import re
import string
import html as html_mod
from typing import Iterable

import numpy as np

# ─── NLTK (opcional — fallback gracioso) ────────────────────────────────────
try:
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    STOP_WORDS = set(stopwords.words("english"))
    STEMMER    = PorterStemmer()
except Exception:
    STOP_WORDS = set()
    STEMMER    = None


# ─── Conjunto final de features (12) — sem HTML/URLs ───────────────────────
HEURISTIC_FEATURE_NAMES: list[str] = [
    "char_count",            # 0
    "word_count",            # 1
    "avg_word_len",          # 2
    "num_digits",            # 3
    "has_currency",          # 4   £ $ € ¥
    "has_unsubscribe",       # 5
    "ratio_stopwords",       # 6
    "punctuation_ratio",     # 7
    "entropy_text",          # 8   Shannon (bits)
    "has_reply_marker",      # 9   linhas iniciadas por '>'
    "has_signature_block",   # 10  "Best regards", "Sincerely", "--"
    "has_shortcode",         # 11  número 4-6 dígitos (SMS premium)
]
N_HEURISTIC: int = len(HEURISTIC_FEATURE_NAMES)


# ─── Regex pré-compiladas ──────────────────────────────────────────────────
_URL_RE        = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_EMAIL_RE      = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_HTML_RE       = re.compile(r"<[a-z][^>]*>", re.IGNORECASE)
_CURRENCY_RE   = re.compile(r"[£$€¥]")
_SHORTCODE_RE  = re.compile(r"\b\d{4,6}\b")
_REPLY_RE      = re.compile(r"^>", re.MULTILINE)
_SIGNATURE_RE  = re.compile(
    r"(?:best regards|sincerely|kind regards|cheers|^--\s*$)",
    re.IGNORECASE | re.MULTILINE,
)
_MIME_RE       = re.compile(
    r"^(?:From|To|Cc|Date|Subject|Message-Id|Content-Type|Mime-Version|"
    r"X-\w+|Return-Path|Received):[^\n]*\n",
    flags=re.MULTILINE | re.IGNORECASE,
)
_DIGITS_RE     = re.compile(r"\d+")
_QUOTE_RE      = re.compile(r"^>.*$", flags=re.MULTILINE)
_PUNCT_TBL     = str.maketrans(string.punctuation, " " * len(string.punctuation))


# ─── Entropia de Shannon ───────────────────────────────────────────────────
def text_entropy(text: str) -> float:
    """Entropia de Shannon em bits sobre as frequências de caracteres."""
    if not text:
        return 0.0
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return 0.0
    _, counts = np.unique(chars, return_counts=True)
    probs = counts.astype(float) / counts.sum()
    return float(-np.sum(probs * np.log2(probs)))


# ─── 12 features finais ────────────────────────────────────────────────────
def extract_features(texts: Iterable[str]) -> np.ndarray:
    """
    Calcula as 12 features heurísticas (sem HTML/URLs) sobre texto bruto.

    Returns
    -------
    np.ndarray  shape=(n, 12), dtype=float32, sem NaN/Inf.
    """
    texts = [str(t) if not isinstance(t, str) else t for t in texts]
    n = len(texts)
    F = np.zeros((n, N_HEURISTIC), dtype=np.float32)
    for i, t in enumerate(texts):
        words  = t.split()
        nchars = max(len(t), 1)
        nwords = max(len(words), 1)
        F[i, 0]  = float(len(t))
        F[i, 1]  = float(len(words))
        F[i, 2]  = float(sum(len(w) for w in words) / nwords)
        F[i, 3]  = float(sum(c.isdigit() for c in t))
        F[i, 4]  = float(bool(_CURRENCY_RE.search(t)))
        F[i, 5]  = float(bool(re.search(r"\bunsubscribe\b", t, re.I)))
        F[i, 6]  = float(sum(w.lower() in STOP_WORDS for w in words) / nwords)
        F[i, 7]  = float(sum(c in string.punctuation for c in t) / nchars)
        F[i, 8]  = text_entropy(t)
        F[i, 9]  = float(bool(_REPLY_RE.search(t)))
        F[i, 10] = float(bool(_SIGNATURE_RE.search(t)))
        F[i, 11] = float(bool(_SHORTCODE_RE.search(t)))
    return np.nan_to_num(F, nan=0.0, posinf=0.0, neginf=0.0)


# ─── Limpeza textual ───────────────────────────────────────────────────────
def clean_sms(text: str) -> str:
    """Limpeza para SMS: lower, URL/email/dígitos como placeholder, stopwords, stem."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text.lower()
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" emailaddr ", text)
    text = _DIGITS_RE.sub(" num ", text)
    text = text.translate(_PUNCT_TBL)
    text = re.sub(r"\s+", " ", text).strip()
    toks = [t for t in text.split() if t not in STOP_WORDS]
    if STEMMER is not None:
        toks = [STEMMER.stem(t) for t in toks]
    return " ".join(toks)


def clean_email(text: str, max_len: int = 50_000) -> str:
    """Limpeza para email: strip HTML/MIME/citações, depois mesmo fluxo do SMS sem stem."""
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    text = text[:max_len]
    text = _HTML_RE.sub(" ", text)
    text = html_mod.unescape(text)
    text = _MIME_RE.sub("", text)
    text = _QUOTE_RE.sub("", text)
    text = text.lower()
    text = _URL_RE.sub(" url ", text)
    text = _EMAIL_RE.sub(" emailaddr ", text)
    text = _DIGITS_RE.sub(" num ", text)
    text = text.translate(_PUNCT_TBL)
    text = re.sub(r"\s+", " ", text).strip()
    toks = [t for t in text.split() if t not in STOP_WORDS]
    return " ".join(toks)


# ─── Atalhos para combinar limpeza + features ──────────────────────────────
def transform_for_model(text: str, mode: str):
    """
    Devolve (texto_limpo, features_heuristicas[shape=(1,12)]).
    Pipeline idêntico ao usado no notebook para garantir reproducibilidade.
    """
    cleaner = clean_sms if mode == "sms" else clean_email
    cleaned = cleaner(text)
    features = extract_features([text])
    return cleaned, features
