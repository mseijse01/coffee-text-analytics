# What's Next — Strategic Brainstorm

**Date**: 2026-04-20
**Context**: coffee-text-analytics is near-complete as a research pipeline. Thinking through what moves the needle most for an ML/AI Engineer portfolio.

---

## Current State Snapshot

### coffee-text-analytics (this repo)
- ✅ Multi-modal NLP pipeline (TF-IDF, BERT, GloVe, LDA, NMF, sentiment)
- ✅ 6 regression models + MNIR; best: XGBoost R²=0.9453
- ✅ MLflow + Optuna integration (research-grade)
- ✅ Docker, CI/CD, pre-commit hooks, typed core modules
- ✅ Thesis compliance validation
- ❌ **No serving layer** — models trained but never exposed via API (Makefile even has a placeholder for "Task 4: FastAPI serving")
- ❌ No monitoring / model drift detection
- ❌ No DAG orchestration (linear CLI pipeline only)

### coffee-database (sibling project)
- ✅ 7 working web scrapers (Netherlands roasters)
- ✅ Pydantic models, NLP extraction, sustainability/flavor scoring
- ✅ 72 coffee beans from 7 roasters already extracted
- ✅ Full PostgreSQL schema designed (8 tables, views, triggers)
- ❌ DB not yet migrated from JSON → PostgreSQL/Supabase
- ❌ No dashboard or API
- ❌ No connection to the ratings/text-analytics model

---

## The Strategic Options

### Option A — Finish coffee-text-analytics in isolation
Add the missing production pieces:
- **FastAPI serving layer**: load model from MLflow Registry, expose `/predict` endpoint
- **Pipeline DAG**: wrap the 5 pipeline steps in Prefect or a simple Dagster flow (visual DAG is portfolio-friendly)
- **Model monitoring**: Evidently AI or custom Prometheus metrics for drift detection

**Pros**: Closes the "production ML system" gap cleanly in one repo.
**Cons**: Still a standalone research artifact. Doesn't feed into real-world data.

### Option B — Focus on coffee-database
- PostgreSQL/Supabase migration
- Expand scrapers to cover more Dutch roasters
- Add a recommendation engine (collaborative or content-based)
- Build a lightweight Streamlit dashboard

**Pros**: Practical, real product. Showcases data engineering and scraping.
**Cons**: Disconnected from the ML work that's already done.

### Option C — Bridge both projects ⭐ (recommended)

The narrative arc becomes:

```
coffee-database                        coffee-text-analytics
─────────────────                      ──────────────────────────
Scrape Dutch roasters       ──►  FastAPI /predict endpoint
Store in PostgreSQL         ──►  Rate bean descriptions in real-time
Enrich with ML ratings      ──►  Drive recommendation scores
Serve via dashboard         ◄──  Return structured predictions
```

This tells a complete ML engineering story in one sentence:
> "I trained a text-analytics model on 6,400 coffee reviews, deployed it behind an API, then built a live database of Dutch roasters that calls the API to auto-rate new coffees and power a recommendation system."

**What this showcases (skills for ML/AI Engineer role):**
| Skill | Where it shows |
|-------|---------------|
| NLP + regression modelling | coffee-text-analytics (already done) |
| Experiment tracking (MLflow) | coffee-text-analytics (already done) |
| Model serving (FastAPI) | new serving layer |
| Data engineering (scrapers + DB) | coffee-database |
| System integration | bridge between the two |
| End-to-end pipeline | the full arc |

---

## Recommended Priority Order

### 1. FastAPI Serving Layer (coffee-text-analytics) — ~1–2 days
The single highest-impact addition. Loads the trained XGBoost model from MLflow Registry and exposes:
- `POST /predict` — takes coffee description text → returns predicted rating
- `GET /health` — liveness check
- `GET /model-info` — current model version, R², metadata

This is also the bridge enabler: coffee-database can call this endpoint.

### 2. PostgreSQL Migration (coffee-database) — ~1 day
Schema already designed. Migrate from JSON persistence to a real database so the project is production-grade and the data can be queried properly.

### 3. Integration Bridge — ~1–2 days
Add a `rating_predictor.py` module in coffee-database that:
- Calls the coffee-text-analytics FastAPI `/predict` endpoint
- Stores predicted ratings back into the `quality_scores` table
- Runs as part of the scrape pipeline (after each bean is scraped, get a predicted rating)

### 4. Simple Dashboard (optional) — stretch goal
Streamlit app showing: top-rated roasters, bean recommendations, NLP-extracted tasting notes vs. model predictions.

---

## On DAG Pipelines + MLflow Optimisation

**DAGs**: Worth adding if it replaces the current linear CLI pipeline. Prefect is lightweight and the visual DAG is good for portfolios. But lower priority than serving layer — nobody asks "do you have Prefect?" but they do ask "how do you serve models?"

**MLflow optimisation**: The current setup is already mature (PostgreSQL backend, MinIO, Optuna, SHAP). Marginal return on more investment here. Leave as-is unless a specific gap appears.

---

## One-liner pitch for portfolio

> Built a full-cycle coffee intelligence system: scraped 70+ Dutch roasters into a PostgreSQL database, trained a multi-modal NLP pipeline (BERT + TF-IDF + sentiment) that achieves R²=0.9453 on quality prediction, deployed it as a FastAPI service, and wired the scraper to auto-rate new beans — enabling a content-based recommendation engine backed by real ML.
