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


def clean_text(text):
    """
    Clean text by removing HTML tags, URLs, and special characters.

    Args:
        text (str): Input text

    Returns:
        str: Cleaned text
    """
    if not isinstance(text, str):
        return ""

    text = re.sub(r"<.*?>", "", text)  # Remove HTML tags
    text = re.sub(
        r"http\S+|www\S+|https\S+", "", text, flags=re.MULTILINE
    )  # Remove URLs
    text = re.sub(
        r"[^a-zA-Z0-9\s,.!?;]", "", text
    )  # Remove special characters except punctuation
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


def extract_country_info(location):
    """
    Extract country name from location string.

    Args:
        location (str): Location string that may contain country information

    Returns:
        str: Extracted country name or None
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

    # Extract words that look like country names using regex
    words = re.findall(r"\b[A-Z][a-z]{2,}\b", location)
    if words:
        # Return the last capitalized word as the potential country
        return words[-1]

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


def load_coffee_data(file_path):
    """
    Load coffee review data from CSV file.

    Args:
        file_path (str): Path to CSV file

    Returns:
        pd.DataFrame: Loaded data
    """
    try:
        logger.info(f"Loading data from {file_path}")
        data = pd.read_csv(file_path)
        logger.info(f"Loaded {data.shape[0]} rows and {data.shape[1]} columns")
        return data
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        return pd.DataFrame()


def process_raw_data(input_file, output_file, text_columns=None):
    """
    Process raw coffee review data and save processed version.

    Args:
        input_file (str): Path to input file
        output_file (str): Path to output file
        text_columns (list): List of text columns to process
    """
    # Ensure output directory exists
    output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    df = load_coffee_data(input_file)
    if df.empty:
        raise ValueError(f"Failed to load data from {input_file}")

    # Set default text columns if not provided
    if text_columns is None:
        text_columns = ["description", "notes"]

    # Standardize prices if available
    if "price" in df.columns:
        df = standardize_prices(df, "price")
        logger.info("Standardized price information")

    # Extract country information if location is available
    if "location" in df.columns:
        df["country_of_origin"] = df["location"].apply(extract_country_info)
        logger.info("Extracted country of origin information")

    # Merge and preprocess text columns
    df = merge_text_columns(df, text_columns, output_col="merged_text")
    logger.info(f"Merged text columns: {text_columns}")

    # Apply text preprocessing
    df["processed_text"] = df["merged_text"].apply(
        lambda x: preprocess_text(x, remove_stop=True)
    )
    logger.info("Applied text preprocessing")

    # Save processed data
    df.to_csv(output_file, index=False)
    logger.info(f"Saved processed data to {output_file}")

    return df
