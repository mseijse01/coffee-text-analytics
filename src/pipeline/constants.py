"""
Shared constants for pipeline steps.

EXCLUDE_COLUMNS is the single source of truth for non-feature columns.
Both feature selection and model training import from here — no duplication.
"""

EXCLUDE_COLUMNS: list = [
    # Target variable
    "rating",
    # Raw text columns (processed versions are also excluded below)
    "desc_1",
    "desc_2",
    "desc_3",
    # Processed text
    "processed_desc_1",
    "processed_desc_2",
    "processed_desc_3",
    "merged_text",
    "processed_text",
    "all_text",
    # Identifiers / metadata
    "id",
    "slug",
    "url",
    "name",
    "location",
    "loc",
    # Categorical (raw — encoded versions are kept as features)
    "roaster",
    "roast",
    "origin",
    "country_of_origin",
    # Numeric non-feature columns
    "est_price",
    "review_date",
    "agtron",
    "with_milk",
    "price_value",
    "price_unit",
    "price_standardized",
    # Sensory attributes (kept separate per thesis methodology)
    "aroma",
    "acid",
    "body",
    "flavor",
    "aftertaste",
]
