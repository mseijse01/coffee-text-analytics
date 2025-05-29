#!/usr/bin/env python3
"""
Topic Modeling Configuration Validation Script

This script validates that the topic modeling configuration matches the thesis requirements exactly.

Usage:
    python validate_topic_modeling_config.py
"""

import sys
import os
import logging
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.features.topic_extractor import TopicExtractor
from src.config.settings import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def validate_topic_modeling_configuration():
    """
    Validate topic modeling configuration against thesis requirements.

    Returns:
        bool: True if all validations pass, False otherwise
    """
    logger.info("🔍 Starting Topic Modeling Configuration Validation")
    logger.info("=" * 60)

    validation_results = []

    # Test 1: Default Configuration Validation
    logger.info("📋 Test 1: Default Configuration Validation")
    try:
        extractor = TopicExtractor()
        config = extractor.config

        # Check n_topics
        expected_topics = 10
        actual_topics = config["n_topics"]
        if actual_topics == expected_topics:
            logger.info(
                f"✅ PASS: n_topics = {actual_topics} (matches thesis requirement)"
            )
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: n_topics = {actual_topics}, expected {expected_topics}"
            )
            validation_results.append(False)

        # Check algorithms
        expected_algorithms = ["lda", "nmf"]
        actual_algorithms = config["algorithms"]
        if set(actual_algorithms) == set(expected_algorithms):
            logger.info(
                f"✅ PASS: algorithms = {actual_algorithms} (matches thesis requirement)"
            )
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: algorithms = {actual_algorithms}, expected {expected_algorithms}"
            )
            validation_results.append(False)

    except Exception as e:
        logger.error(f"❌ FAIL: Error creating TopicExtractor: {e}")
        validation_results.append(False)

    # Test 2: Feature Names Validation
    logger.info("\n📋 Test 2: Feature Names Validation")
    try:
        extractor = TopicExtractor()
        extractor._create_feature_names()
        feature_names = extractor.get_feature_names()

        # Check LDA feature names
        expected_lda_features = [f"lda_topic_{i}" for i in range(10)]
        lda_features = [name for name in feature_names if name.startswith("lda_")]

        if lda_features == expected_lda_features:
            logger.info(
                f"✅ PASS: LDA feature names correct: {lda_features[:3]}...{lda_features[-1]}"
            )
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: LDA features = {lda_features}, expected {expected_lda_features}"
            )
            validation_results.append(False)

        # Check NMF feature names
        expected_nmf_features = [f"nmf_topic_{i}" for i in range(10)]
        nmf_features = [name for name in feature_names if name.startswith("nmf_")]

        if nmf_features == expected_nmf_features:
            logger.info(
                f"✅ PASS: NMF feature names correct: {nmf_features[:3]}...{nmf_features[-1]}"
            )
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: NMF features = {nmf_features}, expected {expected_nmf_features}"
            )
            validation_results.append(False)

        # Check total feature count
        expected_total = 20  # 10 LDA + 10 NMF
        actual_total = len(feature_names)
        if actual_total == expected_total:
            logger.info(f"✅ PASS: Total features = {actual_total} (10 LDA + 10 NMF)")
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: Total features = {actual_total}, expected {expected_total}"
            )
            validation_results.append(False)

    except Exception as e:
        logger.error(f"❌ FAIL: Error validating feature names: {e}")
        validation_results.append(False)

    # Test 3: Model Configuration Validation
    logger.info("\n📋 Test 3: Model Configuration Validation")
    try:
        extractor = TopicExtractor()

        # Test with sample data
        sample_texts = [
            "This coffee has excellent fruity notes with bright acidity",
            "Dark roast with chocolate and nutty flavors",
            "Light roast with floral aroma and citrus finish",
            "Medium body with caramel sweetness and smooth finish",
            "Complex flavor profile with wine-like characteristics",
        ]

        # Fit the extractor
        extractor.fit(sample_texts)

        # Check LDA model configuration
        if extractor.lda_model_ is not None:
            lda_components = extractor.lda_model_.n_components
            if lda_components == 10:
                logger.info(f"✅ PASS: LDA model has {lda_components} components")
                validation_results.append(True)
            else:
                logger.error(f"❌ FAIL: LDA components = {lda_components}, expected 10")
                validation_results.append(False)
        else:
            logger.error("❌ FAIL: LDA model not created")
            validation_results.append(False)

        # Check NMF model configuration
        if extractor.nmf_model_ is not None:
            nmf_components = extractor.nmf_model_.n_components
            if nmf_components == 10:
                logger.info(f"✅ PASS: NMF model has {nmf_components} components")
                validation_results.append(True)
            else:
                logger.error(f"❌ FAIL: NMF components = {nmf_components}, expected 10")
                validation_results.append(False)
        else:
            logger.error("❌ FAIL: NMF model not created")
            validation_results.append(False)

        # Test feature extraction
        features_df = extractor.extract_features(sample_texts)
        if features_df.shape[1] == 20:  # 10 LDA + 10 NMF
            logger.info(
                f"✅ PASS: Feature extraction produces {features_df.shape[1]} features"
            )
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: Feature extraction produces {features_df.shape[1]} features, expected 20"
            )
            validation_results.append(False)

    except Exception as e:
        logger.error(f"❌ FAIL: Error validating model configuration: {e}")
        validation_results.append(False)

    # Test 4: Global Configuration Validation
    logger.info("\n📋 Test 4: Global Configuration Validation")
    try:
        config = Config()
        global_n_topics = config.features.n_topics

        if global_n_topics == 10:
            logger.info(f"✅ PASS: Global config n_topics = {global_n_topics}")
            validation_results.append(True)
        else:
            logger.error(
                f"❌ FAIL: Global config n_topics = {global_n_topics}, expected 10"
            )
            validation_results.append(False)

    except Exception as e:
        logger.error(f"❌ FAIL: Error validating global configuration: {e}")
        validation_results.append(False)

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("📊 VALIDATION SUMMARY")
    logger.info("=" * 60)

    total_tests = len(validation_results)
    passed_tests = sum(validation_results)
    failed_tests = total_tests - passed_tests

    logger.info(f"Total Tests: {total_tests}")
    logger.info(f"Passed: {passed_tests}")
    logger.info(f"Failed: {failed_tests}")

    if all(validation_results):
        logger.info(
            "🎉 ALL TESTS PASSED - Topic modeling configuration is thesis-compliant!"
        )
        logger.info("\n✅ THESIS COMPLIANCE CONFIRMED:")
        logger.info("   • Both LDA and NMF use exactly 10 topics")
        logger.info("   • Feature naming follows thesis conventions")
        logger.info("   • Model configuration matches thesis methodology")
        logger.info("   • Global configuration is consistent")
        return True
    else:
        logger.error(
            "❌ SOME TESTS FAILED - Topic modeling configuration needs attention"
        )
        return False


def main():
    """Main validation function."""
    try:
        success = validate_topic_modeling_configuration()

        if success:
            logger.info("\n🎯 TASK COMPLETION STATUS:")
            logger.info("✅ Topic Modeling Configuration Verification - COMPLETED")
            logger.info(
                "   This task from THESIS_ALIGNMENT_AUDIT.md is now validated and complete."
            )
            sys.exit(0)
        else:
            logger.error("\n❌ VALIDATION FAILED")
            logger.error(
                "   Topic modeling configuration needs fixes before marking complete."
            )
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ CRITICAL ERROR: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
