"""
Text preprocessing utilities for coffee review data.
"""

import re
import pandas as pd
import logging
import nltk
import os
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Global flag to check if NLTK data is downloaded
NLTK_DATA_DOWNLOADED = False


def ensure_nltk_data():
    """
    Ensure necessary NLTK data is downloaded.
    """
    global NLTK_DATA_DOWNLOADED

    if NLTK_DATA_DOWNLOADED:
        return

    try:
        nltk_data = ["punkt", "stopwords", "wordnet"]
        for data in nltk_data:
            try:
                nltk.data.find(f"tokenizers/{data}")
                logger.info(f"NLTK data '{data}' already available.")
            except LookupError:
                logger.info(f"Downloading NLTK data: {data}")
                nltk.download(data)
        NLTK_DATA_DOWNLOADED = True
    except Exception as e:
        logger.warning(f"Issue with NLTK data: {e}")


def clean_text(text: str, remove_punctuation: bool = True) -> str:
    """
    Clean text by removing HTML tags, URLs, and optionally punctuation.

    Args:
        text (str): Input text
        remove_punctuation (bool): Whether to remove punctuation marks (default: True)

    Returns:
        str: Cleaned text

    Examples:
        >>> clean_text("Great coffee! Very smooth.")
        "Great coffee Very smooth"
        >>> clean_text("Great coffee! Very smooth.", remove_punctuation=False)
        "Great coffee! Very smooth."
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(
        r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE
    )  # Remove URLs

    if remove_punctuation:
        text = re.sub(
            r"[^a-zA-Z0-9\s]", "", text
        )  # Remove all special characters and punctuation
    else:
        text = re.sub(r"[^\w\s,.!?;]", "", text)  # Keep basic punctuation

    return text


def tokenize_text(text):
    """
    Tokenize text into individual words.

    Args:
        text (str): Input text

    Returns:
        list: List of tokens
    """
    try:
        from nltk.tokenize import word_tokenize

        ensure_nltk_data()  # Ensure NLTK data is available

        if not isinstance(text, str):
            return []
        tokens = word_tokenize(text)
        return tokens
    except ImportError:
        logger.warning(
            "NLTK tokenize is not available. Using basic split for tokenization."
        )
        return text.split() if isinstance(text, str) else []


def remove_stopwords(tokens, keep_stopwords=False):
    """
    Remove common stopwords from token list.

    Args:
        tokens (list): List of tokens
        keep_stopwords (bool): Whether to keep stopwords

    Returns:
        list: Filtered tokens
    """
    if keep_stopwords:
        return tokens

    try:
        from nltk.corpus import stopwords

        ensure_nltk_data()  # Ensure NLTK data is available

        stop_words = set(stopwords.words("english"))
        return [token for token in tokens if token.lower() not in stop_words]
    except ImportError:
        logger.warning("NLTK stopwords not available. Skipping stopword removal.")
        return tokens


def lemmatize_text(tokens):
    """
    Lemmatize tokens to their base form.

    Args:
        tokens (list): List of tokens

    Returns:
        list: Lemmatized tokens
    """
    try:
        from nltk.stem import WordNetLemmatizer

        ensure_nltk_data()  # Ensure NLTK data is available

        lemmatizer = WordNetLemmatizer()
        return [lemmatizer.lemmatize(token.lower()) for token in tokens]
    except ImportError:
        logger.warning("NLTK lemmatizer not available. Skipping lemmatization.")
        return tokens


def preprocess_text(text, remove_stop=True):
    """
    Apply full preprocessing pipeline to text.

    Args:
        text (str): Input text
        remove_stop (bool): Whether to remove stopwords

    Returns:
        str: Preprocessed text
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = [token.lower() for token in tokens]
    tokens = remove_stopwords(tokens, not remove_stop)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


def preprocess_text_for_embeddings(text):
    """
    Preprocess text for embedding-based models and sentiment analysis (thesis methodology).

    Following thesis methodology:
    - Remove stopwords
    - Retain punctuation
    - Clean and tokenize
    - Lemmatize

    Args:
        text (str): Input text

    Returns:
        str: Preprocessed text optimized for embeddings and sentiment analysis
    """
    # Clean text but retain punctuation
    text = clean_text(text, remove_punctuation=False)
    tokens = tokenize_text(text)
    tokens = [token.lower() for token in tokens]
    # Remove stopwords (keep_stopwords=False)
    tokens = remove_stopwords(tokens, keep_stopwords=False)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


def preprocess_text_for_topics(text):
    """
    Preprocess text for topic modeling (thesis methodology).

    Following thesis methodology:
    - Retain stopwords (important for context in topic modeling)
    - Remove punctuation (reduces noise in topic modeling)
    - Clean and tokenize
    - Lemmatize

    Args:
        text (str): Input text

    Returns:
        str: Preprocessed text optimized for topic modeling
    """
    # Clean text and remove punctuation
    text = clean_text(text, remove_punctuation=True)
    tokens = tokenize_text(text)
    tokens = [token.lower() for token in tokens]
    # Keep stopwords (keep_stopwords=True)
    tokens = remove_stopwords(tokens, keep_stopwords=True)
    tokens = lemmatize_text(tokens)
    return " ".join(tokens)


def create_specialized_datasets(df, text_columns=None):
    """
    Create three specialized datasets following thesis methodology.

    Following thesis approach:
    1. Embeddings dataset: For TF-IDF, BERT, GloVe, sentiment analysis
    2. Topic modeling dataset: For LDA and NMF
    3. Sentiment dataset: Same as embeddings (shared requirements)

    Args:
        df (pd.DataFrame): Input DataFrame with text columns
        text_columns (list): List of text columns to process

    Returns:
        tuple: (embeddings_df, topics_df, sentiment_df)
    """
    if text_columns is None:
        text_columns = ["desc_1", "desc_2", "desc_3"]

    logger.info("Creating specialized datasets following thesis methodology")
    logger.info("1. Embeddings dataset: Remove stopwords, retain punctuation")
    logger.info("2. Topic modeling dataset: Retain stopwords, remove punctuation")
    logger.info("3. Sentiment dataset: Same as embeddings")

    # Create copies for each specialized dataset
    embeddings_df = df.copy()
    topics_df = df.copy()
    sentiment_df = df.copy()

    # Process each text column with specialized preprocessing
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Processing column {col} for specialized datasets")

            # Embeddings preprocessing (remove stopwords, retain punctuation)
            embeddings_df[f"processed_{col}_embeddings"] = (
                df[col].fillna("").apply(preprocess_text_for_embeddings)
            )

            # Topic modeling preprocessing (retain stopwords, remove punctuation)
            topics_df[f"processed_{col}_topics"] = (
                df[col].fillna("").apply(preprocess_text_for_topics)
            )

            # Sentiment preprocessing (same as embeddings)
            sentiment_df[f"processed_{col}_sentiment"] = (
                df[col].fillna("").apply(preprocess_text_for_embeddings)
            )
        else:
            logger.warning(f"Column '{col}' not found in DataFrame")

    logger.info("Specialized datasets created successfully")
    return embeddings_df, topics_df, sentiment_df


def extract_country_info(location: str) -> str:
    """
    Extract country name from location string.

    Prioritizes known coffee-producing countries that appear at string start.

    Args:
        location (str): Location string (e.g., "Ethiopia Yirgacheffe")

    Returns:
        str: Extracted country name (e.g., "Ethiopia")

    Examples:
        >>> extract_country_info("Ethiopia Yirgacheffe")
        "Ethiopia"
        >>> extract_country_info("Colombia Huila")
        "Colombia"
        >>> extract_country_info("Jamaica Blue Mountain")
        "Jamaica"
    """
    if pd.isna(location) or not isinstance(location, str):
        return None

    # Common country mappings in coffee data
    country_mapping = {
        "United States": "United States",
        "USA": "United States",
        "US": "United States",
        "Hawaii": "United States",
        "U.S.A": "United States",
        "Sumatra": "Indonesia",
        "Java": "Indonesia",
        "Sulawesi": "Indonesia",
        "Taiwan": "Taiwan",
        "Republic of China": "Taiwan",
    }

    # Try direct mapping first
    if location in country_mapping:
        return country_mapping[location]

    # Known countries that appear first in coffee origin strings
    known_countries = [
        "Ethiopia",
        "Colombia",
        "Kenya",
        "Jamaica",
        "Guatemala",
        "Brazil",
        "Costa Rica",
        "Panama",
        "Yemen",
        "Peru",
        "Indonesia",
        "Mexico",
        "Nicaragua",
        "Honduras",
        "El Salvador",
        "Ecuador",
        "Bolivia",
        "Venezuela",
        "Rwanda",
        "Burundi",
        "Tanzania",
        "Uganda",
        "Malawi",
        "Zimbabwe",
        "India",
        "China",
        "Thailand",
        "Vietnam",
        "Philippines",
    ]

    # Check if any known country appears at the start of the location
    for country in known_countries:
        if location.startswith(country):
            return country

    # Extract first capitalized word as potential country
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", location)
    if words:
        return words[0]

    return location


def standardize_prices(df, price_col="price"):
    """
    Standardize coffee prices to USD per kilogram.

    Args:
        df (pd.DataFrame): DataFrame containing price data
        price_col (str): Name of price column

    Returns:
        pd.DataFrame: DataFrame with standardized prices
    """
    if price_col not in df.columns:
        logger.warning(f"Column {price_col} not found in DataFrame.")
        return df

    # Create a copy of the DataFrame to avoid modifying the original
    result = df.copy()

    # Fill NaN values
    result[price_col] = result[price_col].fillna("")

    # Extract numeric price and unit
    result["price_value"] = result[price_col].str.extract(r"(\d+\.?\d*)").astype(float)
    result["price_unit"] = result[price_col].str.extract(r"(\$/lb|\$/kg|\$/oz|\$)")

    # Convert to standard price per kg
    conversion_rates = {
        "$/lb": 2.20462,  # 1 kg = 2.20462 lbs
        "$/oz": 35.274,  # 1 kg = 35.274 oz
        "$/kg": 1,  # Already in kg
        "$": 2.20462,  # Assume $ is per pound by default
    }

    # Initialize price_standardized column
    result["price_standardized"] = float("nan")

    # Apply conversion rates
    for unit, rate in conversion_rates.items():
        mask = result["price_unit"] == unit
        if mask.any():  # Only apply if there are any matching rows
            result.loc[mask, "price_standardized"] = (
                result.loc[mask, "price_value"] * rate
            )

    return result


def merge_text_columns(df, columns, output_col="merged_text"):
    """
    Merge multiple text columns into one combined text column.

    Args:
        df (pd.DataFrame): DataFrame containing text columns
        columns (list): List of column names to merge
        output_col (str): Name for the output column

    Returns:
        pd.DataFrame: DataFrame with merged text column
    """
    # Make a copy to avoid modifying the original
    result = df.copy()

    # Filter for columns that actually exist
    existing_columns = [col for col in columns if col in result.columns]

    if not existing_columns:
        logger.warning(
            f"None of the specified columns {columns} exist in the DataFrame"
        )
        return result

    # Fill NA values with empty strings for text concatenation
    for col in existing_columns:
        result[col] = result[col].fillna("")

    # Combine text columns
    result[output_col] = result[existing_columns].apply(
        lambda row: " ".join(row).strip(), axis=1
    )

    return result


def load_csv_for_preprocessing(file_path: str = None) -> pd.DataFrame:
    """
    Load CSV data for text preprocessing operations with dual-mode support.

    This function loads CSV data specifically for text preprocessing operations.
    Uses pandas for easier text manipulation and sklearn compatibility.

    Supports dual-mode loading:
    - If file_path is provided: Load from specified file (legacy mode)
    - If file_path is None: Use environment-based dual-mode loader

    Args:
        file_path (str, optional): Path to CSV file. If None, uses dual-mode loader.

    Returns:
        pd.DataFrame: Data optimized for text preprocessing operations

    Note:
        Returns pandas DataFrame (not Polars) for easier text processing.
        Use convert_pandas_to_polars() if you need Polars format afterward.
    """
    try:
        if file_path is not None:
            # Legacy mode: load from specified file path
            logger.info(f"Loading data from specified file: {file_path}")
            data = pd.read_csv(file_path)
        else:
            # Dual-mode: use environment-based loader
            logger.info("Using dual-mode data loader based on ENVIRONMENT variable")
            try:
                from .load_data import load_coffee_data

                # Load with Polars and convert to pandas
                polars_df = load_coffee_data()
                data = polars_df.to_pandas()
                logger.info("Successfully loaded data using dual-mode loader")
            except ImportError as e:
                logger.error(f"Failed to import dual-mode loader: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to load data using dual-mode loader: {e}")
                raise

        logger.info(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def process_raw_data(
    input_file=None,
    output_file=None,
    text_columns=None,
    sample_fraction=None,
    sample_size=None,
):
    """
    Process raw coffee review data and save processed version with dual-mode support.

    Args:
        input_file (str, optional): Path to input file. If None, uses dual-mode loader.
        output_file (str): Path to output file
        text_columns (list): List of text columns to process
        sample_fraction (float): Fraction of data to sample (e.g., 0.1 for 10%)
        sample_size (int): Absolute number of samples to use
    """
    # Ensure output directory exists
    if output_file:
        output_dir = os.path.dirname(output_file)
        os.makedirs(output_dir, exist_ok=True)

    # Load data using dual-mode loader
    df = load_csv_for_preprocessing(input_file)
    if df.empty:
        error_msg = (
            f"Failed to load data from {input_file}"
            if input_file
            else "Failed to load data using dual-mode loader"
        )
        raise ValueError(error_msg)

    original_size = len(df)
    logger.info(f"Original dataset size: {original_size} rows")

    # Apply sampling if requested
    if sample_fraction is not None:
        if not 0 < sample_fraction <= 1:
            raise ValueError("sample_fraction must be between 0 and 1")
        sample_size_calculated = int(original_size * sample_fraction)
        df = df.sample(n=sample_size_calculated, random_state=42)
        logger.info(f"Sampled {sample_fraction * 100:.1f}% of data: {len(df)} rows")
    elif sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if sample_size > original_size:
            logger.warning(
                f"Requested sample_size ({sample_size}) is larger than dataset ({original_size}). Using full dataset."
            )
        else:
            df = df.sample(n=sample_size, random_state=42)
            logger.info(f"Sampled {sample_size} rows from {original_size} total rows")

    # Set default text columns for coffee review data
    if text_columns is None:
        text_columns = ["desc_1", "desc_2", "desc_3"]

    # Extract country information from origin if available
    if "origin" in df.columns:
        df["country_of_origin"] = df["origin"].apply(extract_country_info)
        logger.info("Extracted country of origin information")

    # Standardize prices if available
    if "est_price" in df.columns:
        df = standardize_prices(df, "est_price")
        logger.info("Standardized price information")

    # Process each text column individually
    for col in text_columns:
        if col in df.columns:
            logger.info(f"Preprocessing text column: {col}")
            df[f"processed_{col}"] = (
                df[col].fillna("").apply(lambda x: preprocess_text(x, remove_stop=True))
            )
        else:
            logger.warning(f"Text column '{col}' not found in data")

    # Create a merged text column for compatibility
    existing_text_cols = [col for col in text_columns if col in df.columns]
    if existing_text_cols:
        df = merge_text_columns(df, existing_text_cols, output_col="merged_text")
        df["processed_text"] = df["merged_text"].apply(
            lambda x: preprocess_text(x, remove_stop=True)
        )
        logger.info(f"Merged and processed text columns: {existing_text_cols}")

    # Save processed data
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df
