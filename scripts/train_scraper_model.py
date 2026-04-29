"""Train a lightweight coffee quality model using only scraper-compatible features.

This model is designed to run in-process inside the coffee-database scraper pipeline
(no FastAPI server needed). It uses the same CoffeeReview.com dataset but extracts
only the features available from roaster website scraping:
  - Origin tier (high/mid/commercial specialty reputation)
  - Roast level (ordinal)
  - Process (natural / honey / washed)
  - Tasting note categories (9 binary features extracted from review text)

The full XGBoost model (BERT + GloVe + TF-IDF) is preserved unchanged.
This script produces two additional artifacts alongside the existing models:
  models/scraper_model.pkl        — trained Ridge or RandomForest regressor
  models/scraper_feature_cols.pkl — ordered list of feature column names

Usage:
    cd /Users/seijas/Code/coffee-text-analytics
    python scripts/train_scraper_model.py
"""

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).parent.parent
DATA_PATH = ROOT / "data" / "raw" / "coffee_clean.csv"
MODELS_DIR = ROOT / "models"

# ---------------------------------------------------------------------------
# Feature definitions — mirroring extraction_service.py categories
# ---------------------------------------------------------------------------

NOTE_KEYWORDS = {
    "fruity": [
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
    "floral": [
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
    "sweet": [
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
    "spicy": [
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
    "bright": [
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
    "complex": [
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

PROCESS_NATURAL_KEYWORDS = [
    "natural",
    "dry process",
    "dry-process",
    "sun dried",
    "sun-dried",
]
PROCESS_HONEY_KEYWORDS = ["honey", "semi-washed", "pulped natural", "semi washed"]


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
    # Try exact match first
    for key, tier in ORIGIN_TIERS.items():
        if key.lower() in origin.lower():
            return tier
    return 1


def _roast_ord(roast: str) -> int:
    if not roast or not isinstance(roast, str):
        return 3  # default: medium
    return ROAST_MAP.get(roast.lower().strip(), 3)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build the 13-feature matrix from CoffeeReview data."""
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
        feat = {
            "origin_tier": _origin_tier(row.get("origin", "")),
            "roast_ord": _roast_ord(row.get("roast", "")),
            "is_natural": _text_has_any(t, PROCESS_NATURAL_KEYWORDS),
            "is_honey": _text_has_any(t, PROCESS_HONEY_KEYWORDS),
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
    print("Sparse Scraper Model — Training")
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
        print(f"  {col:<20} {X[col].mean():.3f}")

    # Stratified split by rating bin
    bins = pd.cut(y, bins=[80, 84, 87, 90, 95, 100], labels=False, include_lowest=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y, test_size=0.2, random_state=42, stratify=bins
    )
    print(f"\nTrain: {len(X_train)}  Test: {len(X_test)}")

    # Train and compare models
    candidates = {
        "Ridge(α=1.0)": Ridge(alpha=1.0),
        "Ridge(α=0.1)": Ridge(alpha=0.1),
        "RandomForest": RandomForestRegressor(
            n_estimators=300, max_depth=6, min_samples_leaf=10, random_state=42
        ),
    }

    print("\nModel comparison:")
    print(f"  {'Model':<20} {'R²':>8} {'RMSE':>8} {'MAE':>8}")
    print("  " + "-" * 46)

    best_name, best_model, best_r2 = None, None, -np.inf
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mae = mean_absolute_error(y_test, preds)
        marker = " ←" if r2 > best_r2 else ""
        print(f"  {name:<20} {r2:>8.4f} {rmse:>8.4f} {mae:>8.4f}{marker}")
        if r2 > best_r2:
            best_r2, best_name, best_model = r2, name, model

    print(f"\nWinner: {best_name}  (R²={best_r2:.4f})")

    # Sanity checks
    print("\nSanity check predictions:")
    checks = [
        {
            "name": "Ethiopian natural, fruity+floral",
            "origin_tier": 3,
            "roast_ord": 2,
            "is_natural": 1,
            "is_honey": 0,
            "has_fruity": 1,
            "has_floral": 1,
            "has_sweet": 1,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spicy": 0,
            "has_bright": 1,
            "has_body": 0,
            "has_complex": 1,
        },
        {
            "name": "Colombian washed, chocolate+caramel",
            "origin_tier": 2,
            "roast_ord": 3,
            "is_natural": 0,
            "is_honey": 0,
            "has_fruity": 0,
            "has_floral": 0,
            "has_sweet": 1,
            "has_nutty": 1,
            "has_earthy": 0,
            "has_spicy": 0,
            "has_bright": 0,
            "has_body": 1,
            "has_complex": 0,
        },
        {
            "name": "Vietnamese dark, no notes",
            "origin_tier": 1,
            "roast_ord": 5,
            "is_natural": 0,
            "is_honey": 0,
            "has_fruity": 0,
            "has_floral": 0,
            "has_sweet": 0,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spicy": 0,
            "has_bright": 0,
            "has_body": 0,
            "has_complex": 0,
        },
        {
            "name": "Kenyan natural, berry+bright",
            "origin_tier": 3,
            "roast_ord": 2,
            "is_natural": 1,
            "is_honey": 0,
            "has_fruity": 1,
            "has_floral": 0,
            "has_sweet": 0,
            "has_nutty": 0,
            "has_earthy": 0,
            "has_spicy": 0,
            "has_bright": 1,
            "has_body": 0,
            "has_complex": 0,
        },
    ]
    for check in checks:
        label = check.pop("name")
        X_check = np.array([[check[c] for c in feature_cols]])
        pred = float(np.clip(best_model.predict(X_check)[0], 80, 100))
        print(f"  {label:<45} → {pred:.1f}")

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
