# IMDb Sentiment Analysis

**Web scraping · NLP preprocessing · BERT fine-tuning · Multi-model comparison**

[![CI](https://github.com/houcem58/imdb-sentiment-analysis/actions/workflows/ci.yml/badge.svg)](https://github.com/houcem58/imdb-sentiment-analysis/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

---

## Overview

End-to-end sentiment analysis pipeline on IMDb movie reviews:

1. **Scraping** — Selenium + Chromium headless scrapes 1 000+ reviews from a target movie
2. **Preprocessing** — lowercase · punctuation removal · stop-word filtering · WordNet lemmatisation
3. **Labelling** — zero-shot DistilBERT SST-2 assigns *positif / négatif / neutre* labels
4. **Balancing** — `nlpaug` contextual word-embedding augmentation upsamples minority classes
5. **Baseline models** — TF-IDF + Logistic Regression · Linear SVM
6. **Deep learning** — 1D-CNN · LSTM (Keras / TensorFlow)
7. **BERT fine-tuning** — `distilbert-base-uncased` fine-tuned end-to-end (HuggingFace Trainer)
8. **Evaluation** — confusion matrices · learning curves · word clouds · model comparison

---

## Results (Inception — tt1375666, 1 273 reviews)

| Model | Accuracy |
|---|---|
| DistilBERT fine-tuned | **0.8858** |
| LSTM | ~0.84 |
| CNN | ~0.82 |
| SVM | ~0.81 |
| Logistic Regression | ~0.80 |

*Results from a single deterministic run (`seed=42`) on the augmented balanced dataset (2 865 samples, 3 equal classes).*

---

## Quickstart

### Google Colab (recommended — free GPU)

1. Open [`imdb_sentiment_analysis.ipynb`](imdb_sentiment_analysis.ipynb) in Colab
2. Set runtime to **GPU** (Runtime → Change runtime type → T4)
3. Run all cells top-to-bottom — the notebook handles Drive mounting and all installs

### Local

```bash
git clone https://github.com/houcem58/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Open the notebook in JupyterLab / VS Code
jupyter lab imdb_sentiment_analysis.ipynb
```

> Set `IN_COLAB = False` detection is automatic — local runs save to `data/` and `outputs/`.

---

## Project Structure

```
imdb-sentiment-analysis/
├── imdb_sentiment_analysis.ipynb   # Main notebook (12 sections, clean outputs)
├── requirements.txt                # Pinned dependencies
├── .gitignore                      # Excludes data CSVs, model weights, outputs
├── .github/
│   └── workflows/
│       └── ci.yml                  # Lint + structure validation on every push
└── data/                           # Created at runtime (gitignored)
```

---

## Notebook Sections

| # | Section | Description |
|---|---|---|
| 1 | Configuration | All paths, model names, hyper-parameters in one cell |
| 2 | Install & Import | Quiet install, all imports, GPU detection |
| 3 | Data Collection | Selenium IMDb scraper with caching |
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

## Author

**Houcem Hammami** — Technical Manager, AI & Data Engineering
houcem0508@gmail.com

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE) for details.
