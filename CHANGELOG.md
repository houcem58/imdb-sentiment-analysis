# Changelog

All notable changes to this project are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.1.0] — 2026-01-20

### Changed
- `ci.yml` renamed to `pipeline.yml` for consistency with portfolio repos
- Workflow name updated: "CI" → "Pipeline"
- `concurrency` group added to cancel stale runs on push
- README restructured: Approach table, Pipeline diagram, CI rationale section
- Author section with LinkedIn and email badges added
- README badge updated to `pipeline.yml`

### Added
- `CONTRIBUTING.md`, `SECURITY.md`, `CHANGELOG.md`

---

## [1.0.0] — 2025-11-10

### Added
- `imdb_sentiment_analysis.ipynb` — 12-section end-to-end NLP pipeline
- Sections: scraping · preprocessing · zero-shot labelling · class balancing · 5 models · evaluation · inference
- `generate_notebook.py` — programmatic notebook builder with cell ID injection (nbformat 4.5)
- `pipeline.yml` — 3-job CI: notebook cleanliness check, structure validation, requirements dry-run
- Notebook CI uses `nbformat.validate()` directly — no ruff/nbqa (inappropriate for cross-cell scope)
- `requirements.txt` with pinned dependencies
- Apache 2.0 license
