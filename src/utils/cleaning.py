"""
Utility functions for cleaning and preprocessing coffee review data.
Includes functions for:
- Price standardization (conversion to USD per kilogram)
- Country extraction and standardization
- Text preprocessing and cleaning
- Data saving and loading
- Data quality checks
"""

# cleaning.py imports section
import re
import pycountry
import polars as pl
import pandas as pd
import plotly.express as px
from pathlib import Path
from typing import Union, Optional, List
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


# Price cleaning utilities
def clean_price(price_str: str) -> Optional[float]:
    """
    Extract and standardize price to USD per kilogram.
    Handles different units and currencies.

    Args:
        price_str: String containing price information

    Returns:
        float: Standardized price in USD per kilogram
        None: If price cannot be extracted or standardized

    Conversion rates:
    - 1 kilogram = 2.20462 pounds = 35.274 ounces = 1000 grams
    - NT$ (Taiwan Dollar) to USD conversion rate: approximately 1 NT$ = 0.032 USD
    """
    if pl.Series([price_str]).is_null()[0]:
        return None

    price_str = str(price_str).lower()

    # Extract the numeric price
    price_match = re.search(r"(?:nt)?\s*\$?(\d+,?\d*\.?\d*)", price_str)
    if not price_match:
        return None

    # Remove commas and convert to float
    price = float(price_match.group(1).replace(",", ""))

    # Convert NT$ to USD
    if "nt" in price_str:
        price = price * 0.032

    # Convert to price per kilogram based on units
    if "ounce" in price_str:
        # Extract number of ounces
        oz_match = re.search(r"(\d+)\s*ounce", price_str)
        if oz_match:
            ounces = float(oz_match.group(1))
            price = (price / ounces) * 35.274
    elif "gram" in price_str:
        # Extract number of grams
        g_match = re.search(r"(\d+)\s*gram", price_str)
        if g_match:
            grams = float(g_match.group(1))
            price = (price / grams) * 1000
    else:  # Assume price is per pound if no unit specified
        price = price * 2.20462

    return price


def standardize_prices(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize all prices in the dataset to USD per kilogram.

    Args:
        df: DataFrame containing 'est_price' column

    Returns:
        DataFrame with new 'price_per_kg' column
    """
    if "est_price" not in df.columns:
        raise ValueError("DataFrame must contain 'est_price' column")

    # Apply price cleaning to the est_price column
    price_per_kg = df["est_price"].to_pandas().apply(clean_price)

    return df.with_columns(pl.Series("price_per_kg", price_per_kg))


# Country extraction utilities
def extract_country(origin: str) -> str:
    """
    Extracts country information from the 'origin' column.

    Args:
        origin: String containing origin information

    Returns:
        str: Extracted country name(s) or 'ND' if none found
    """
    countries = []
    for country in pycountry.countries:
        if country.name in origin:
            countries.append(country.name)
        elif hasattr(country, "official_name") and country.official_name in origin:
            countries.append(country.official_name)
    return ", ".join(countries) if countries else "ND"


def correct_country_name(country: str, origin: str) -> str:
    """
    Corrects country names based on specific rules.

    Args:
        country: Extracted country name
        origin: Original origin string

    Returns:
        str: Corrected country name
    """
    if "Taiwan" in origin:
        return "Taiwan"
    if "Tanzania" in origin:
        return "Tanzania"
    if "Bolivia" in origin:
        return "Bolivia"
    if "Vietnam" in origin:
        return "Vietnam"
    if "Hawaii" in origin or "Hawai'i" in origin or "Big Island" in origin:
        return "United States"
    return country


def extract_and_correct_country(df: pl.DataFrame) -> pl.DataFrame:
    """
    Extracts and corrects country information from the 'origin' column.

    Args:
        df: DataFrame containing 'origin' column

    Returns:
        DataFrame with new 'country_of_origin' column
    """
    extracted_countries = df["origin"].to_list()
    extracted_countries = [extract_country(origin) for origin in extracted_countries]

    corrected_countries = [
        correct_country_name(country, origin)
        for country, origin in zip(extracted_countries, df["origin"].to_list())
    ]

    return df.with_columns(pl.Series("country_of_origin", corrected_countries))


def analyze_price_distribution(df: pl.DataFrame) -> None:
    """
    Analyze and visualize price distribution.

    Args:
        df: DataFrame with 'price_per_kg' column
    """
    print("\nPrice Statistics (USD/kg):")
    print("-" * 50)

    # Calculate statistics
    price_stats = df.select(
        pl.col("price_per_kg").mean().alias("Mean"),
        pl.col("price_per_kg").median().alias("Median"),
        pl.col("price_per_kg").std().alias("Std"),
        pl.col("price_per_kg").min().alias("Min"),
        pl.col("price_per_kg").max().alias("Max"),
    )
    display(price_stats)

    # Create distribution plot
    fig = px.box(
        df.select("price_per_kg").to_pandas(),
        y="price_per_kg",
        title="Distribution of Coffee Prices (USD/kg)",
    )
    fig.show()


def analyze_country_distribution(df: pl.DataFrame) -> pl.DataFrame:
    """
    Analyze and visualize country distribution.

    Args:
        df: DataFrame with 'country_of_origin' column
    """
    print("\nCountry Distribution Analysis:")
    print("-" * 50)

    # Calculate country statistics
    country_stats = (
        df.group_by("country_of_origin")
        .agg(
            [
                pl.count().alias("count"),
                pl.col("rating").mean().alias("avg_rating"),
                pl.col("price_per_kg").mean().alias("avg_price_per_kg"),
            ]
        )
        .sort("count", descending=True)
    )

    print("\nTop 10 Countries by Number of Reviews:")
    display(country_stats.head(10))

    # Create visualization
    fig = px.bar(
        country_stats.head(15).to_pandas(),
        x="country_of_origin",
        y="count",
        title="Top 15 Countries by Number of Reviews",
        labels={"country_of_origin": "Country", "count": "Number of Reviews"},
    )
    fig.show()

    return country_stats


# Text preprocessing utilities
def remove_urls(text: str) -> str:
    """Remove URLs from text."""
    return re.sub(r"http\S+|www\S+", "", text) if isinstance(text, str) else text


def clean_text(text: str) -> str:
    """Clean and standardize text."""
    if isinstance(text, str):
        text = text.lower()
        text = re.sub(r"\[.*?\]", "", text)  # remove content in square brackets
        text = re.sub(r"\w*\d\w*", "", text)  # remove words with numbers
    return text


def remove_punctuation(text: str) -> str:
    """Remove all punctuation from text."""
    return re.sub(r"[^\w\s]", "", text) if isinstance(text, str) else text


def handle_negations(text: str) -> str:
    """Handle common negations in text."""
    negations = {
        "isn't": "is not",
        "aren't": "are not",
        "wasn't": "was not",
        "weren't": "were not",
        "hasn't": "has not",
        "haven't": "have not",
        "hadn't": "had not",
        "doesn't": "does not",
        "don't": "do not",
        "didn't": "did not",
        "won't": "will not",
        "wouldn't": "would not",
        "shan't": "shall not",
        "shouldn't": "should not",
        "can't": "cannot",
        "couldn't": "could not",
        "mustn't": "must not",
        "mightn't": "might not",
        "needn't": "need not",
    }
    if isinstance(text, str):
        for neg in negations:
            text = re.sub(r"\b{}\b".format(neg), negations[neg], text)
    return text


def remove_stopwords(text: str) -> str:
    """Remove common English stopwords."""
    stop_words = set(stopwords.words("english"))
    word_tokens = word_tokenize(text)
    filtered_text = [word for word in word_tokens if word not in stop_words]
    return " ".join(filtered_text)


def lemmatize_text(text: str) -> str:
    """Lemmatize text to reduce words to their base form."""
    lemmatizer = WordNetLemmatizer()
    word_tokens = word_tokenize(text)
    lemmatized_text = [lemmatizer.lemmatize(word) for word in word_tokens]
    return " ".join(lemmatized_text)


def preprocess_text(
    text: str, remove_sw: bool = True, with_punctuation: bool = True
) -> str:
    """
    Preprocess text with configurable options.

    Args:
        text: Input text to preprocess
        remove_sw: Whether to remove stopwords
        with_punctuation: Whether to keep punctuation

    Returns:
        str: Preprocessed text
    """
    text = clean_text(text)
    text = handle_negations(text)
    text = lemmatize_text(text)
    if not with_punctuation:
        text = remove_punctuation(text)
    if remove_sw:
        text = remove_stopwords(text)
    return text


def apply_text_preprocessing(
    df: pl.DataFrame, text_columns: list, for_embeddings: bool = True
) -> pl.DataFrame:
    """
    Apply text preprocessing to specified columns.

    Args:
        df: Input DataFrame
        text_columns: List of columns to preprocess
        for_embeddings: Whether preprocessing is for embeddings (True) or topic modeling (False)

    Returns:
        DataFrame with preprocessed text columns
    """
    print("Applying text preprocessing...")

    for column in text_columns:
        with_punctuation = for_embeddings
        remove_sw = for_embeddings

        df = df.with_columns(
            pl.Series(
                f"processed_{column}",
                df[column]
                .to_pandas()
                .map(
                    lambda text: preprocess_text(
                        text, remove_sw=remove_sw, with_punctuation=with_punctuation
                    )
                    if isinstance(text, str)
                    else ""
                ),
            )
        )

    print("Text preprocessing complete.")
    return df


# Data I/O utilities
def save_parquet(df: pl.DataFrame, path: Union[Path, str], name: str) -> None:
    """Save DataFrame to parquet file."""
    df.write_parquet(path)
    print(f"{name} data saved to {path}")


def load_parquet(path: Union[Path, str], name: str) -> pl.DataFrame:
    """Load DataFrame from parquet file."""
    df = pl.read_parquet(path)
    print(f"{name} DataFrame:")
    display(df.head())
    return df


# Data quality utilities
def check_column_consistency(*dfs: pl.DataFrame) -> None:
    """Check if all DataFrames have the same columns."""
    column_sets = [set(df.columns) for df in dfs]
    if all(column_sets[0] == cols for cols in column_sets):
        print("All datasets have the same columns.")
    else:
        for idx, cols in enumerate(column_sets):
            print(f"Columns in DataFrame {idx+1}: {cols}")


def check_missing_values(df: pl.DataFrame, name: str) -> None:
    """Check for missing values in DataFrame."""
    print(f"\nMissing values in {name} DataFrame:")
    display(df.null_count())


# Add to cleaning.py


def profile_dataset(df: pl.DataFrame, name: str = "Dataset") -> None:
    """
    Perform comprehensive profiling of a DataFrame.

    Args:
        df: DataFrame to profile
        name: Name of the dataset for display purposes
    """
    print(f"\n{name} Overview:")
    print("-" * 50)
    print(f"Number of records: {len(df):,}")
    print(f"Number of features: {len(df.columns):,}")

    # Column information
    print("\nColumn Information:")
    print("-" * 50)

    # Create rows for all columns
    column_info = []
    for col in df.columns:
        dtype = df[col].dtype
        n_missing = df[col].null_count()
        n_unique = df[col].n_unique()
        missing_pct = n_missing / len(df) * 100

        column_info.append(
            {
                "Column": col,
                "Type": str(dtype),
                "Missing": n_missing,
                "Missing %": f"{missing_pct:.1f}%",
                "Unique Values": n_unique,
            }
        )

    # Convert to DataFrame and display full info
    info_df = pl.DataFrame(column_info)

    # Print in a formatted way
    print("\nDetailed Column Analysis:")
    print("-" * 80)
    print(
        f"{'Column':<20} {'Type':<10} {'Missing':>8} {'Missing %':>10} {'Unique Values':>15}"
    )
    print("-" * 80)

    for row in info_df.iter_rows():
        print(f"{row[0]:<20} {row[1]:<10} {row[2]:>8} {row[3]:>10} {row[4]:>15}")


def analyze_numerical_columns(df: pl.DataFrame) -> None:
    """
    Analyze numerical columns in the DataFrame.

    Args:
        df: DataFrame to analyze
    """
    print("\nNumerical Columns Statistics:")
    print("-" * 50)
    numerical_cols = [
        col for col in df.columns if df[col].dtype in [pl.Float64, pl.Int64]
    ]

    if numerical_cols:
        stats = df.select(numerical_cols).describe()
        display(stats)
    else:
        print("No numerical columns found.")


def check_target_variable(df: pl.DataFrame, target_col: str) -> None:
    """
    Check the quality of a target variable.

    Args:
        df: DataFrame containing the target
        target_col: Name of target column
    """
    if target_col not in df.columns:
        print(f"\nWarning: Target column '{target_col}' not found in DataFrame")
        return

    print(f"\nTarget Variable ({target_col}) Analysis:")
    print("-" * 50)

    # Basic stats
    stats = df[target_col].describe()
    print("Basic Statistics:")
    display(stats)

    # Missing values
    n_missing = df[target_col].null_count()
    if n_missing > 0:
        print(f"Warning: Found {n_missing:,} missing values ({n_missing/len(df):.1%})")
    else:
        print("No missing values found in the target column.")

    # Distribution plot
    fig = px.histogram(
        df[target_col].to_pandas(),
        title=f"Distribution of {target_col}",
        template="simple_white",
    )
    fig.show()


def analyze_agtron_values(df: pl.DataFrame) -> None:
    """
    Analyze Agtron values to understand their distribution and format.

    Args:
        df: DataFrame with 'agtron' column
    """
    print("\nAgtron Value Analysis:")
    print("-" * 50)

    # Get unique values and their counts
    agtron_counts = (
        df.group_by("agtron")
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )

    print("\nMost common Agtron values:")
    display(agtron_counts.head(10))

    # Check for different formats
    print("\nValue format analysis:")
    agtron_values = df["agtron"].drop_nulls().unique().to_list()

    # Categorize values
    single_values = []
    range_values = []
    other_format = []

    for val in agtron_values:
        if isinstance(val, (int, float)):
            single_values.append(val)
        elif isinstance(val, str):
            if "/" in val:
                range_values.append(val)
            else:
                other_format.append(val)

    print(f"\nTypes of values found:")
    print(f"- Single numbers: {len(single_values)}")
    print(f"- Range values (x/y): {len(range_values)}")
    print(f"- Other formats: {len(other_format)}")

    if single_values:
        print(f"\nSingle values range: {min(single_values)} - {max(single_values)}")

    if range_values:
        print("\nSample of range values:")
        print(range_values[:5])

    if other_format:
        print("\nSample of other formats:")
        print(other_format[:5])

    # Relationship with roast levels
    if "roast" in df.columns:
        print("\nAgtron values by roast level:")
        roast_agtron = (
            df.group_by("roast")
            .agg(
                [pl.col("agtron").mean().alias("avg_agtron"), pl.count().alias("count")]
            )
            .sort("count", descending=True)
        )
        display(roast_agtron)


def standardize_roast_degree(df: pl.DataFrame) -> pl.DataFrame:
    """
    Standardize roast degree using Agtron values.

    Args:
        df: DataFrame with 'agtron' column
    """
    # First, extract the ground reading (first number)
    df = df.with_columns(
        [
            pl.col("agtron")
            .str.extract(r"(\d+)/")
            .cast(pl.Float64)
            .alias("agtron_ground"),
            pl.col("agtron")
            .str.extract(r"/(\d+)")
            .cast(pl.Float64)
            .alias("agtron_whole"),
        ]
    )

    # Then create roast categories using pl.when/then
    df = df.with_columns(
        [
            pl.when(pl.col("agtron_ground") > 65)
            .then(pl.lit("Light"))
            .when(pl.col("agtron_ground").is_between(55, 65))
            .then(pl.lit("Medium-Light"))
            .when(pl.col("agtron_ground").is_between(45, 55))
            .then(pl.lit("Medium"))
            .when(pl.col("agtron_ground").is_between(35, 45))
            .then(pl.lit("Medium-Dark"))
            .when(pl.col("agtron_ground") < 35)
            .then(pl.lit("Dark"))
            .otherwise(None)
            .alias("roast_by_agtron")
        ]
    )

    return df


def analyze_roast_standardization(df: pl.DataFrame) -> None:
    """Analyze the relationship between Agtron readings and roast levels."""
    print("\nRoast Level Analysis:")
    print("-" * 50)

    # Distribution of standardized roast levels
    roast_dist = (
        df.group_by("roast_by_agtron")
        .agg(
            [
                pl.count().alias("count"),
                pl.col("agtron_ground").mean().alias("avg_ground"),
                pl.col("agtron_whole").mean().alias("avg_whole"),
            ]
        )
        .sort("avg_ground", descending=True)
    )

    print("\nDistribution of Standardized Roast Levels:")
    display(roast_dist)

    # Compare with original roast labels
    comparison = (
        df.group_by(["roast", "roast_by_agtron"])
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
    )

    print("\nComparison with Original Roast Labels:")
    display(comparison)


def drop_irrelevant_columns(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop columns that are not needed for analysis.

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with irrelevant columns removed
    """
    columns_to_drop = [
        "slug",  # URLs have no predictive value
        "all_text",  # Replaced by individual descriptions
        "name",  # Too granular
        "location",  # Captured in roaster variable
        "origin",  # Replaced by standardized country_of_origin
        "review_date",  # Insufficient data for temporal analysis
        "agtron",  # Replaced by ground and whole bean readings
        "with_milk",  # Too many missing values
    ]

    return df.drop(columns_to_drop)


def summarize_column_changes(
    original_df: pl.DataFrame, cleaned_df: pl.DataFrame
) -> None:
    """
    Summarize changes in columns after cleaning.

    Args:
        original_df: DataFrame before dropping columns
        cleaned_df: DataFrame after dropping columns
    """
    print("\nColumn Changes:")
    print("-" * 50)
    print(f"Original columns: {len(original_df.columns)}")
    print(f"Remaining columns: {len(cleaned_df.columns)}")
    print("\nRemaining columns:")
    for col in cleaned_df.columns:
        print(f"• {col}")


# Add to cleaning.py


def analyze_missing_values(df: pl.DataFrame) -> None:
    """
    Analyze missing values in the dataset.

    Args:
        df: DataFrame to analyze
    """
    print("\nMissing Values Analysis:")
    print("-" * 50)

    # Calculate missing values and percentages
    missing_stats = []
    for col in df.columns:
        n_missing = df[col].null_count()
        pct_missing = (n_missing / len(df)) * 100
        missing_stats.append(
            {"Column": col, "Missing": n_missing, "Missing %": f"{pct_missing:.1f}%"}
        )

    # Convert to DataFrame and sort by number of missing values
    missing_df = pl.DataFrame(missing_stats).sort("Missing", descending=True)
    display(
        missing_df.filter(pl.col("Missing") > 0)
    )  # Show only columns with missing values


def analyze_outliers(df: pl.DataFrame, column: str) -> None:
    """
    Analyze outliers in specified column.

    Args:
        df: DataFrame to analyze
        column: Column name to check for outliers
    """
    print(f"\nOutlier Analysis for {column}:")
    print("-" * 50)

    # Calculate basic statistics
    stats = df.select(
        [
            pl.col(column).mean().alias("mean"),
            pl.col(column).median().alias("median"),
            pl.col(column).std().alias("std"),
            pl.col(column).min().alias("min"),
            pl.col(column).max().alias("max"),
        ]
    )

    print("\nBasic Statistics:")
    display(stats)

    # Create box plot
    fig = px.box(df[column].to_pandas(), title=f"Distribution of {column}")
    fig.show()


def clean_dataset(
    df: pl.DataFrame, min_rating: float = 80.0
) -> tuple[pl.DataFrame, dict]:
    """
    Clean dataset by handling missing values and outliers.

    Args:
        df: DataFrame to clean
        min_rating: Minimum rating threshold

    Returns:
        Tuple of (cleaned DataFrame, cleaning statistics)
    """
    initial_rows = len(df)

    # Filter by rating
    df_filtered = df.filter(pl.col("rating") >= min_rating)
    rows_after_rating = len(df_filtered)

    # Drop rows with missing values
    df_cleaned = df_filtered.drop_nulls()
    final_rows = len(df_cleaned)

    # Compile cleaning statistics
    stats = {
        "initial_rows": initial_rows,
        "rows_after_rating_filter": rows_after_rating,
        "final_rows": final_rows,
        "removed_by_rating": initial_rows - rows_after_rating,
        "removed_by_missing": rows_after_rating - final_rows,
        "total_removed": initial_rows - final_rows,
        "total_removed_pct": ((initial_rows - final_rows) / initial_rows) * 100,
    }

    return df_cleaned, stats


def process_and_analyze_text(
    df: pl.DataFrame, desc_columns: list
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """
    Process text data for different analysis types and provide summary.

    Args:
        df: DataFrame with text columns
        desc_columns: List of description columns to process

    Returns:
        Tuple of (embeddings_df, topic_modeling_df, sentiment_df)
    """
    print("\nText Preprocessing Summary:")
    print("-" * 50)

    # Process for embeddings
    df_processed = apply_text_preprocessing(df, desc_columns, for_embeddings=True)
    df_embeddings = df_processed.clone()

    # Process for topic modeling
    for col in desc_columns:
        # Convert the column to pandas series, apply preprocessing, and convert back
        processed_column = (
            df_processed[f"processed_{col}"]
            .to_pandas()
            .map(
                lambda text: preprocess_text(
                    text, remove_sw=False, with_punctuation=False
                )
            )
        )
        df_processed = df_processed.with_columns(
            pl.Series(f"processed_{col}", processed_column)
        )
    df_topic_modeling = df_processed

    # Use same preprocessing as embeddings for sentiment
    df_sentiment = df_embeddings.clone()

    # Analyze results
    print("\nSample of processed text for each purpose:")

    print("\nEmbeddings preprocessing (stopwords removed, punctuation retained):")
    display(
        df_embeddings.select(
            [col for col in df_embeddings.columns if "processed_" in col]
        ).head(2)
    )

    print("\nTopic Modeling preprocessing (stopwords retained, punctuation removed):")
    display(
        df_topic_modeling.select(
            [col for col in df_topic_modeling.columns if "processed_" in col]
        ).head(2)
    )

    # Basic text statistics
    print("\nText Statistics:")
    for df_type, df_processed in [
        ("Embeddings", df_embeddings),
        ("Topic Modeling", df_topic_modeling),
    ]:
        print(f"\n{df_type}:")
        for col in desc_columns:
            processed_col = f"processed_{col}"
            # Fixed: specify split by whitespace
            avg_length = df_processed[processed_col].str.split(by=" ").list.len().mean()
            print(f"• {col} average word count: {avg_length:.1f}")

    return df_embeddings, df_topic_modeling, df_sentiment
