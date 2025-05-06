<div align="center">

# IMDb Sentiment Analysis

### Web Scraping · NLP Preprocessing · BERT Fine-tuning · Multi-Model Comparison

**5 models · DistilBERT accuracy 88.6% · 1 273 scraped reviews · seed=42**

[![Pipeline](https://github.com/houcem58/imdb-sentiment-analysis/actions/workflows/pipeline.yml/badge.svg)](https://github.com/houcem58/imdb-sentiment-analysis/actions/workflows/pipeline.yml)
[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green)](LICENSE)

</div>

---

> End-to-end NLP pipeline on real IMDb movie reviews: scraping, preprocessing, zero-shot
> labelling, class balancing, and 5-model comparison from Logistic Regression to DistilBERT
> fine-tuning. Structured as a reproducible ML study with clean notebook outputs and CI validation.

---

## The Problem

Sentiment classification on user-generated reviews presents several compounding challenges
that standard benchmarks don't capture: class imbalance (most scraped reviews skew positive),
informal language, mixed-language text, and the need for a zero-shot labelling strategy
when no ground-truth labels exist at collection time.

This notebook addresses each of these systematically before any model is trained.

---

## Approach

| Stage | Technique | Why |
|---|---|---|
| Labelling | Zero-shot DistilBERT SST-2 | No hand-labelled data at collection time |
| Class balancing | `nlpaug` contextual word-embedding augmentation | Prevents majority-class bias |
| Baseline | TF-IDF + Logistic Regression / SVM | Fast, interpretable, strong lower bound |
| Deep learning | 1D-CNN + LSTM (Keras) | Sequence structure without full fine-tuning |
| Fine-tuning | DistilBERT end-to-end (HuggingFace Trainer) | Full transformer on domain-specific reviews |

---

## Results (Inception — tt1375666, 1 273 reviews)

| Model | Accuracy |
|---|---|
| DistilBERT fine-tuned | **0.8858** |
| LSTM | ~0.84 |
| 1D-CNN | ~0.82 |
| Linear SVM | ~0.81 |
| Logistic Regression | ~0.80 |

Single deterministic run (`seed=42`) on the augmented balanced dataset (2 865 samples, 3 equal classes).

---

## Pipeline

```
IMDb reviews (Selenium scraper)
    │  1 273 reviews — Inception (tt1375666)
    ▼
Text Preprocessing
    │  lowercase · punct removal · stopwords · WordNet lemmatisation
    ▼
Zero-shot Labelling (DistilBERT SST-2)
    │  positif / négatif / neutre
    ▼
Class Balancing (nlpaug contextual augmentation)
    │  → 2 865 samples, 3 equal classes
    ▼
Feature Extraction
    │  TF-IDF (classical) · tokenizer embeddings (neural)
    ▼
Model Training × 5
    │  LogReg · SVM · CNN · LSTM · DistilBERT
    ▼
Evaluation
    │  accuracy · confusion matrix · learning curves · word clouds
    ▼
Inference Function
    │  predict_all(text) → all 5 model outputs
```

---

## Quick Start

### Google Colab (recommended — free GPU)

1. Open [`imdb_sentiment_analysis.ipynb`](imdb_sentiment_analysis.ipynb) in Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4)
3. Run all cells top-to-bottom

### Local

```bash
git clone https://github.com/houcem58/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter lab imdb_sentiment_analysis.ipynb
```

---

## Notebook Structure

| # | Section | Description |
|---|---|---|
| 1 | Configuration | All paths, model names, hyper-parameters in one cell |
| 2 | Install & Import | Quiet install, imports, GPU detection |
| 3 | Data Collection | Selenium IMDb scraper with local cache |
| 4 | Text Preprocessing | Tokenise · lemmatise · stop-words |
| 5 | Sentiment Labelling | Zero-shot DistilBERT pipeline |
| 6 | Class Balancing | Contextual augmentation via `nlpaug` |
| 7 | Classical Models | LogReg + SVM with TF-IDF |
| 8 | Deep Learning | CNN + LSTM with Keras |
| 9 | BERT Fine-tuning | HuggingFace Trainer, 4 epochs |
| 10 | Visualisation | Accuracy bars · confusion matrices · learning curves · word clouds |
| 11 | Results Summary | Ranked model comparison table |
| 12 | Inference Function | `predict_all(text)` runs all five models |

---

## Project Structure

```
imdb-sentiment-analysis/
├── imdb_sentiment_analysis.ipynb   # Main notebook (12 sections, clean outputs)
├── generate_notebook.py            # Programmatic notebook builder
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Excludes data/, model weights, outputs
├── .github/
│   └── workflows/
│       └── pipeline.yml            # Notebook cleanliness + structure validation
├── CHANGELOG.md
├── CONTRIBUTING.md
├── SECURITY.md
└── LICENSE                         # Apache 2.0
```

---

## CI

Three automated checks on every push:

| Check | What it does |
|---|---|
| `Notebook outputs clean` | Fails if any cell has uncommitted outputs or execution counts |
| `Validate notebook structure` | Checks 10 required sections exist via nbformat |
| `Requirements sanity check` | `pip install --dry-run` to verify dependencies resolve |

Designed for notebooks: no ruff (inappropriate for cross-cell scope), no nbstripout --verify
(serialization false positives). Uses `nbformat.validate()` directly.

---

## Author

**Houcem Hammami** — Technical Manager, AI & Data Engineering

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://linkedin.com/in/houcem-hammami)
[![Email](https://img.shields.io/badge/Email-houcem0508%40gmail.com-red)](mailto:houcem0508@gmail.com)

---

## License

Copyright 2025–2026 Houcem Hammami. Licensed under the Apache License, Version 2.0 — see [LICENSE](LICENSE).
