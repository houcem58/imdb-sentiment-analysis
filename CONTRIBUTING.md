# Contributing

## Ways to Contribute

- **Bug reports** — issues with the scraper, broken cells, or dependency conflicts
- **Model additions** — add a new classifier in a new notebook section (e.g. RoBERTa)
- **Dataset extensions** — scrape a different movie; document the IMDb title ID used

## Standards

- All cells must run top-to-bottom without errors on a clean environment
- No cell outputs committed — run `nbstripout imdb_sentiment_analysis.ipynb` before committing
- All random operations must use `seed=42` for reproducibility
- No real user data — only publicly available IMDb reviews

## CI

Three checks run on every push:

```
Notebook outputs clean      → fails if any cell has outputs or execution_count
Validate notebook structure → checks 10 required sections exist
Requirements sanity check   → pip install --dry-run
```

All three must pass before a PR can merge.

## Running Locally

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
jupyter lab imdb_sentiment_analysis.ipynb
```

For Selenium scraping: Chromium must be installed and chromedriver must be on PATH.
