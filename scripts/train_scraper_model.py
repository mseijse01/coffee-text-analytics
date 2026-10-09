"""Train a lightweight coffee quality model using only scraper-compatible features.

This model is designed to run in-process inside the coffee-database scraper pipeline
(no FastAPI server needed). It uses the same CoffeeReview.com dataset but extracts
only the features available from roaster website scraping:
  - Origin tier (high/mid/commercial specialty reputation)
  - Roast level (ordinal)
  - Process (natural / honey / washed)
  - Variety tier (premium / heritage / commercial)
  - Note count (number of tasting notes, capped at 5)
  - Tasting note categories (9 binary features, names matching extraction_service.py)

Category names match extraction_service.py exactly:
  fruit, aroma, sweetness, nutty, earthy, spice, acidity, body, complexity

The full XGBoost model (BERT + GloVe + TF-IDF) is preserved unchanged.
This script produces two artifacts alongside the existing models:
  models/scraper_model.pkl        — trained best regressor
  models/scraper_feature_cols.pkl — ordered list of feature column names

Usage:
    cd /Users/seijas/Code/coffee-text-analytics
    ~/.virtualenvs/coffee-analytics/bin/python scripts/train_scraper_model.py
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Allow importing from src/ (TwoStepHyperparameterTuner lives there)
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from utils.hyperparameter_tuning import TwoStepHyperparameterTuner  # noqa: E402

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "coffee_clean.csv"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Feature definitions — category names match extraction_service.py exactly
# ---------------------------------------------------------------------------

NOTE_KEYWORDS = {
    # extraction_service category: "fruit"
    "fruit": [
        "peach",
        "berry",
        "strawberry",
        "blueberry",
        "raspberry",
        "apple",
        "citrus",
        "lemon",
        "lime",
        "orange",
        "grapefruit",
        "mango",
        "cherry",
        "plum",
        "apricot",
        "pear",
        "grape",
        "banana",
        "blackcurrant",
        "currant",
        "compote",
        "zest",
        "mandarin",
        "cassis",
        "mulberry",
        "fig",
        "date",
        "tamarind",
        "passion fruit",
        "passionfruit",
        "melon",
        "watermelon",
    ],
    # extraction_service category: "aroma"
    "aroma": [
        "floral",
        "jasmine",
        "lavender",
        "rose",
        "bergamot",
        "herbal",
        "mint",
        "sage",
        "white flowers",
        "flowers",
        "blossom",
        "hibiscus",
        "elderflower",
        "violet",
        "lilac",
        "lily",
    ],
    # extraction_service category: "sweetness"
    "sweetness": [
        "honey",
        "caramel",
        "chocolate",
        "cocoa",
        "toffee",
        "sweet",
        "sugar",
        "vanilla",
        "maple",
        "molasses",
        "brown sugar",
        "dark chocolate",
        "milk chocolate",
        "fudge",
        "praline",
        "marzipan",
        "candy",
        "cane sugar",
        "butterscotch",
        "nougat",
        "syrup",
    ],
    # extraction_service category: "nutty"
    "nutty": [
        "almond",
        "hazelnut",
        "walnut",
        "peanut",
        "pecan",
        "cashew",
        "coconut",
        "macadamia",
        "malt",
        "grain",
        "cereal",
        "biscuit",
        "cookie",
        "bread",
        "toast",
        "toasted",
    ],
    # extraction_service category: "earthy"
    "earthy": [
        "earthy",
        "woody",
        "cedar",
        "tobacco",
        "soil",
        "mushroom",
        "forest",
        "leather",
        "rubber",
        "pine",
        "resin",
    ],
    # extraction_service category: "spice"
    "spice": [
        "spice",
        "spicy",
        "cinnamon",
        "cardamom",
        "clove",
        "nutmeg",
        "pepper",
        "ginger",
        "allspice",
        "anise",
        "fennel",
        "licorice",
        "liquorice",
        "star anise",
        "black pepper",
    ],
    # extraction_service category: "acidity"
    "acidity": [
        "bright",
        "tart",
        "sharp",
        "vibrant",
        "crisp",
        "tangy",
        "juicy",
        "lively",
        "lemon zest",
        "citric",
    ],
    # extraction_service category: "body"
    "body": [
        "smooth",
        "creamy",
        "bold",
        "full",
        "heavy",
        "silky",
        "velvety",
        "round",
        "rich",
        "buttery",
        "thick",
        "dense",
        "tea",
        "light body",
        "medium body",
        "full body",
    ],
    # extraction_service category: "complexity"
    "complexity": [
        "balanced",
        "complex",
        "layered",
        "nuanced",
        "sophisticated",
        "elegant",
        "refined",
        "lingering",
        "long finish",
        "aftertaste",
    ],
}

ORIGIN_TIERS = {
    # Tier 3 — consistently high scores in specialty coffee world
    "Ethiopia": 3,
    "Kenya": 3,
    "Panama": 3,
    "Yemen": 3,
    "Rwanda": 3,
    "Burundi": 3,
    "Tanzania": 3,
    "Malawi": 3,
    # Tier 2 — strong specialty reputation
    "Colombia": 2,
    "Guatemala": 2,
    "Costa Rica": 2,
    "El Salvador": 2,
    "Honduras": 2,
    "Peru": 2,
    "Bolivia": 2,
    "Ecuador": 2,
    "Papua New Guinea": 2,
    "Jamaica": 2,
    "Mexico": 2,
    "Nicaragua": 2,
    "Sumatra": 2,
    "Sulawesi": 2,
    "Timor-Leste": 2,
    # Tier 1 — mostly commercial (default for unknown)
    "Brazil": 1,
    "Vietnam": 1,
    "Indonesia": 1,
    "India": 1,
    "Laos": 1,
    "Philippines": 1,
    "Uganda": 1,
    "Zambia": 1,
    "Zimbabwe": 1,
    "Congo": 1,
    "Myanmar": 1,
}

ROAST_MAP = {
    "light": 1,
    "medium-light": 2,
    "medium light": 2,
    "medium": 3,
    "medium-dark": 4,
    "medium dark": 4,
    "dark": 5,
}

VARIETY_TIERS = {
    # Tier 3 — rare / premium
    "geisha": 3,
    "pacamara": 3,
    "laurina": 3,
    "maragogipe": 3,
    "java": 3,
    # Tier 2 — heritage cultivars
    "typica": 2,
    "bourbon": 2,
    "heirloom": 2,
    "caturra": 2,
    "catuai": 2,
    "sl28": 2,
    "sl34": 2,
    "batian": 2,
    "villa sarchi": 2,
    # default = 1 (catimor, ruiru, commercial hybrids, unknown)
}

PROCESS_NATURAL_KEYWORDS = [
    "natural",
    "dry process",
    "dry-process",
    "sun dried",
    "sun-dried",
]
PROCESS_HONEY_KEYWORDS = [
    "honey",
    "semi-washed",
    "pulped natural",
    "semi washed",
]


# ---------------------------------------------------------------------------
# Feature extraction helpers
# ---------------------------------------------------------------------------


def _text_has_any(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return int(
        any(re.search(r"\b" + re.escape(kw) + r"\b", text_lower) for kw in keywords)
    )


def _origin_tier(origin: str) -> int:
    if not origin or not isinstance(origin, str):
        return 1
    for key, tier in ORIGIN_TIERS.items():
        if key.lower() in origin.lower():
            return tier
    return 1


def _roast_ord(roast: str) -> int:
    if not roast or not isinstance(roast, str):
        return 3  # default: medium
    return ROAST_MAP.get(roast.lower().strip(), 3)


def _variety_tier(variety: str) -> int:
    if not variety or not isinstance(variety, str):
        return 1
    v = variety.lower()
    for key, tier in VARIETY_TIERS.items():
        if key in v:
            return tier
    return 1


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 15-feature matrix from CoffeeReview data."""
    text = (
        df.get("desc_1", pd.Series([""] * len(df))).fillna("")
        + " "
        + df.get("desc_2", pd.Series([""] * len(df))).fillna("")
        + " "
        + df.get("desc_3", pd.Series([""] * len(df))).fillna("")
    )

    rows = []
    for i, (_, row) in enumerate(df.iterrows()):
        t = text.iloc[i]

        # Count how many distinct note categories appear in this review
        cat_hits = sum(_text_has_any(t, kws) for kws in NOTE_KEYWORDS.values())

        feat = {
            "origin_tier": _origin_tier(row.get("origin", "")),
            "roast_ord": _roast_ord(row.get("roast", "")),
            "is_natural": _text_has_any(t, PROCESS_NATURAL_KEYWORDS),
            "is_honey": _text_has_any(t, PROCESS_HONEY_KEYWORDS),
            "variety_tier": _variety_tier(str(row.get("variety", "") or "")),
            "note_count": min(cat_hits, 5),
        }
        for cat, keywords in NOTE_KEYWORDS.items():
            feat[f"has_{cat}"] = _text_has_any(t, keywords)
        rows.append(feat)

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main training routine
# ---------------------------------------------------------------------------


def main():
    print("=" * 60)
    print("Sparse Scraper Model — Training (Phase 2)")
    print("=" * 60)

    # Load data
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=["rating", "origin"])
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    print(f"\nDataset: {len(df)} rows after filtering")
    print(
        f"Rating range: {df['rating'].min():.0f}–{df['rating'].max():.0f}  "
        f"mean={df['rating'].mean():.1f}  std={df['rating'].std():.2f}"
    )

    # Build features
    print("\nBuilding features...")
    X = build_features(df)
    y = df["rating"].values
    feature_cols = list(X.columns)

    print(f"Feature matrix: {X.shape[0]} rows × {X.shape[1]} columns")
    print(f"Columns: {feature_cols}")
    print("\nFeature means (how often each fires):")
    for col in feature_cols:
        print(f"  {col:<25} {X[col].mean():.3f}")

    # Stratified split by rating bin
    bins = pd.cut(y, bins=[80, 84, 87, 90, 95, 100], labels=False, include_lowest=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=bins
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # Two-step hyperparameter tuning (project's signature methodology):
    #   Phase 1 — RandomizedSearchCV (wide, cheap: n_iter=30, 3-fold CV)
    #   Phase 2 — GridSearchCV zoomed around Phase 1 best params (5-fold CV)
    # Applied independently to Ridge and GradientBoosting; winner keeps the title.

    tuner_config = {
        "randomized_search_config": {
            "n_iter": 30,  # enough to explore; dataset is small
            "cv": 3,
            "scoring": "r2",
            "n_jobs": -1,
            "random_state": 42,
        },
        "grid_search_config": {
            "cv": 5,
            "scoring": "r2",
            "n_jobs": -1,
        },
    }

    results: list[tuple[str, object, float, float, float]] = []

    # --- Ridge ---
    print("\nPhase 1+2: Ridge ...")
    ridge_tuner = TwoStepHyperparameterTuner(tuner_config)
    ridge_tuner.fit(
        estimator=Ridge(),
        X=X_train,
        y=y_train,
        param_distributions={"alpha": [0.001, 0.01, 0.1, 1.0, 10, 100]},
        grid_refinement_factor=3,
    )
    ridge_preds = ridge_tuner.best_estimator_.predict(X_test)
    ridge_r2 = r2_score(y_test, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_test, ridge_preds))
    ridge_mae = mean_absolute_error(y_test, ridge_preds)
    print(f"  Ridge best params: {ridge_tuner.best_params_}")
    print(f"  R²={ridge_r2:.4f}  RMSE={ridge_rmse:.4f}  MAE={ridge_mae:.4f}")
    results.append(
        ("Ridge", ridge_tuner.best_estimator_, ridge_r2, ridge_rmse, ridge_mae)
    )

    # --- GradientBoosting ---
    print("\nPhase 1+2: GradientBoosting ...")
    gbr_tuner = TwoStepHyperparameterTuner(tuner_config)
    gbr_tuner.fit(
        estimator=GradientBoostingRegressor(random_state=42),
        X=X_train,
        y=y_train,
        param_distributions={
            "n_estimators": [50, 100, 200, 300],
            "max_depth": [2, 3, 4, 5, 6],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
        },
        grid_refinement_factor=3,
    )
    gbr_preds = gbr_tuner.best_estimator_.predict(X_test)
    gbr_r2 = r2_score(y_test, gbr_preds)
    gbr_rmse = np.sqrt(mean_squared_error(y_test, gbr_preds))
    gbr_mae = mean_absolute_error(y_test, gbr_preds)
    print(f"  GBR best params: {gbr_tuner.best_params_}")
    print(f"  R²={gbr_r2:.4f}  RMSE={gbr_rmse:.4f}  MAE={gbr_mae:.4f}")
    results.append(
        ("GradientBoosting", gbr_tuner.best_estimator_, gbr_r2, gbr_rmse, gbr_mae)
    )

    # Pick winner
    best_name, best_model, best_r2, _, _ = max(results, key=lambda t: t[2])
    print(f"\nWinner: {best_name}  (R²={best_r2:.4f})")

    # Sanity checks
    print("\nSanity check predictions:")
    checks = [
        {
            "name": "Ethiopian natural, fruit+aroma (Geisha)",
            "origin_tier": 3,
            "roast_ord": 2,
            "is_natural": 1,
            "is_honey": 0,
            "variety_tier": 3,
            "note_count": 5,
            "has_fruit": 1,
            "has_aroma": 1,
            "has_sweetness": 1,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spice": 0,
            "has_acidity": 1,
            "has_body": 0,
            "has_complexity": 1,
        },
        {
            "name": "Colombian washed, sweetness+nutty",
            "origin_tier": 2,
            "roast_ord": 3,
            "is_natural": 0,
            "is_honey": 0,
            "variety_tier": 1,
            "note_count": 3,
            "has_fruit": 0,
            "has_aroma": 0,
            "has_sweetness": 1,
            "has_nutty": 1,
            "has_earthy": 0,
            "has_spice": 0,
            "has_acidity": 0,
            "has_body": 1,
            "has_complexity": 0,
        },
        {
            "name": "Vietnamese dark, no notes",
            "origin_tier": 1,
            "roast_ord": 5,
            "is_natural": 0,
            "is_honey": 0,
            "variety_tier": 1,
            "note_count": 0,
            "has_fruit": 0,
            "has_aroma": 0,
            "has_sweetness": 0,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spice": 0,
            "has_acidity": 0,
            "has_body": 0,
            "has_complexity": 0,
        },
        {
            "name": "Kenyan natural, berry+acidity",
            "origin_tier": 3,
            "roast_ord": 2,
            "is_natural": 1,
            "is_honey": 0,
            "variety_tier": 2,
            "note_count": 3,
            "has_fruit": 1,
            "has_aroma": 0,
            "has_sweetness": 0,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spice": 0,
            "has_acidity": 1,
            "has_body": 0,
            "has_complexity": 0,
        },
    ]
    for check in checks:
        label = check.pop("name")
        X_check = np.array([[check[c] for c in feature_cols]])
        pred = float(best_model.predict(X_check)[0])
        print(f"  {label:<50} → {pred:.1f}")

    # Save artifacts
    model_path = MODELS_DIR / "scraper_model.pkl"
    cols_path = MODELS_DIR / "scraper_feature_cols.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(best_model, f)
    with open(cols_path, "wb") as f:
        pickle.dump(feature_cols, f)

    print(f"\nSaved: {model_path}")
    print(f"Saved: {cols_path}")
    print("\nCopy to coffee-database:")
    print(
        f"  cp {model_path} /Users/seijas/Code/coffee-database/models/scraper_model.pkl"
    )
    print(
        f"  cp {cols_path} /Users/seijas/Code/coffee-database/models/scraper_feature_cols.pkl"
    )
    print("\nDone.")


if __name__ == "__main__":
    main()
