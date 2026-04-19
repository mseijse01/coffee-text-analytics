"""Pipeline step: data preprocessing."""

import logging

from data.preprocessing import process_raw_data

logger = logging.getLogger(__name__)


def run_preprocessing(args, config) -> bool:
    """
    Preprocess raw coffee review data.

    Uses dual-mode loading (local vs MinIO) when input_file matches the
    default config path; otherwise loads from the explicit path.

    Args:
        args: Parsed CLI arguments.
        config: Application configuration object.

    Returns:
        True on success, False on any error.
    """
    logger.info("Starting data preprocessing step")
    try:
        use_dual_mode = args.input_file == str(config.paths.get_raw_data_path())

        if use_dual_mode:
            logger.info("Using dual-mode data loader (environment-based)")
            process_raw_data(
                input_file=None,
                output_file=str(config.paths.get_processed_data_path()),
                text_columns=config.models.text_columns,
                sample_fraction=args.sample_fraction,
                sample_size=args.sample_size,
            )
        else:
            logger.info(f"Using specified input file: {args.input_file}")
            process_raw_data(
                input_file=args.input_file,
                output_file=str(config.paths.get_processed_data_path()),
                text_columns=config.models.text_columns,
                sample_fraction=args.sample_fraction,
                sample_size=args.sample_size,
            )
        return True

    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        return False
