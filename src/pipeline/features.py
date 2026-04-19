"""Pipeline step: feature extraction."""

import logging
import traceback

import polars as pl

from features import CoffeeFeatureManager

logger = logging.getLogger(__name__)


def run_feature_extraction(args, config) -> bool:
    """
    Extract NLP features from preprocessed data using CoffeeFeatureManager.

    Fits extractors on combined text, then extracts features separately for
    each description column following thesis methodology.

    Args:
        args: Parsed CLI arguments.
        config: Application configuration object.

    Returns:
        True on success, False on any error.
    """
    logger.info("Starting feature extraction step")
    try:
        processed_data_path = config.paths.get_processed_data_path()
        logger.info(f"Loading preprocessed data from {processed_data_path}")
        df = pl.read_csv(processed_data_path)
        logger.info(f"Loaded data shape: {df.shape}")

        feature_config = {
            "tfidf": {
                "max_features": config.features.tfidf_max_features,
                "ngram_range": config.features.tfidf_ngram_range,
                "models_dir": str(config.paths.models),
            },
            "bert": {
                "batch_size": config.features.bert_batch_size,
                "max_length": config.features.bert_max_length,
            },
            "topics": {
                "n_topics": config.features.n_topics,
                "algorithms": ["lda", "nmf"],
                "models_dir": str(config.paths.models),
            },
            "sentiment": {"batch_size": config.features.bert_batch_size},
            "glove": {"aggregation": "mean"},
        }

        feature_manager = CoffeeFeatureManager(feature_config)

        logger.info("Fitting feature extractors on combined text from all columns...")
        feature_manager.fit(df, config.models.text_columns)

        logger.info("Extracting features separately for each description column...")
        result_df = feature_manager.extract_all_features(df, config.models.text_columns)

        features_data_path = config.paths.get_features_data_path()
        logger.info(f"Saving features to {features_data_path}")
        result_df.write_csv(features_data_path)

        feature_manager.save_extractors(str(config.paths.models))
        feature_manager.print_summary()

        logger.info(f"Feature extraction completed. Final shape: {result_df.shape}")
        return True

    except Exception as e:
        logger.error(f"Feature extraction failed: {e}")
        logger.error(traceback.format_exc())
        return False
