# Context

Domain glossary for the coffee-text-analytics project. Keep terms here consistent with code, issues, and docs. See `docs/agents/domain.md` for how this file and `docs/adr/` are consumed.

## Purpose

A research ML pipeline that predicts coffee quality ratings from consumer review text (CoffeeReview.com dataset), plus a FastAPI serving layer for the trained models.

## Terms

- **Rating**: the prediction target, on an 80-100 scale.
- **Description columns**: `desc_1`, `desc_2`, `desc_3`, the three free-text review fields used as text input.
- **Sensory attributes**: `aroma`, `acid`, `body`, `flavor`, `aftertaste`. Kept separate from the text features per the thesis methodology; excluded from LASSO input via `EXCLUDE_COLUMNS` in `pipeline/constants.py`.
- **Feature group**: a family of inputs (flavor, text, categorical) that can be evaluated independently.
- **Feature selection**: LASSO-based reduction of the extracted feature set.
- **MNIR**: Multinomial Inverse Regression, included for interpretability rather than performance.
- **Main pipeline**: `main.py`, the thesis-compliant path.
- **Validation scripts**: `validate_*_and_save.py`, a convenience path for generating serving-layer artifacts; not thesis-comparable.
- **Serving layer**: the FastAPI app in `src/serving/` that loads trained artifacts and exposes `/predict`.

## Architecture decisions

ADRs live in `docs/adr/` (not yet created). Add one when making a decision that is hard to reverse or that future contributors would otherwise question.
