#!/usr/bin/env python3
"""
Smart test runner for coffee text analytics — RAM-aware, incremental, coverage-safe.

Runs tests incrementally by file with RAM monitoring to avoid crashes.
Supports safe mode (lightweight tests only) and batch mode (file-by-file).
"""

import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Tuple

import psutil

# Configuration
SAFE_RAM_THRESHOLD_MB = 2500  # Stop if RAM usage exceeds this
TEST_FILES = [
    "tests/test_exceptions.py",  # Lightweight, no ML models
    "tests/test_data_processing.py",  # Data processing core
    "tests/test_models_base.py",  # Model base classes
    "tests/test_categorical_encoder.py",  # Encoding utilities
    "tests/test_config_integration.py",  # Configuration
    "tests/test_feature_manager.py",  # Feature manager (may be heavy)
    "tests/test_feature_selector_contracts.py",  # Feature selection (may be heavy)
    "tests/test_feature_selector_integration.py",  # Feature selection integration
    "tests/test_feature_pipeline_integration.py",  # Full feature pipeline (heavy)
    "tests/test_models_regressors.py",  # Model regressors (may be heavy)
    "tests/test_models_evaluator.py",  # Model evaluation
    "tests/test_mlflow_integration.py",  # MLflow (may be heavy)
    "tests/test_data_preprocessing_integration.py",  # Preprocessing (may be heavy)
    "tests/test_performance.py",  # Performance tests (may be heavy)
    "tests/test_utils_integration.py",  # Utils integration
    "tests/test_visualization_comprehensive.py",  # Visualization (may be heavy)
]

# Lightweight tests that skip heavy ML
LIGHTWEIGHT_TESTS = [
    "tests/test_exceptions.py",
    "tests/test_data_processing.py",
    "tests/test_models_base.py",
    "tests/test_categorical_encoder.py",
    "tests/test_config_integration.py",
]


class RAMMonitor:
    """Monitor RAM usage during test execution."""

    def __init__(self, threshold_mb: int = SAFE_RAM_THRESHOLD_MB):
        self.threshold_mb = threshold_mb
        self.peak_mb = 0
        self.process = psutil.Process()

    def get_ram_mb(self) -> float:
        """Get current process RAM usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024

    def check_and_record(self) -> Tuple[float, bool]:
        """Check RAM and return (current_mb, exceeded_threshold)."""
        current_mb = self.get_ram_mb()
        self.peak_mb = max(self.peak_mb, current_mb)
        exceeded = current_mb > self.threshold_mb
        return current_mb, exceeded

    def report(self):
        """Print RAM usage report."""
        print(
            f"   📊 Peak RAM: {self.peak_mb:.1f} MB (threshold: {self.threshold_mb} MB)"
        )


def run_test_file(
    test_file: str, monitor: RAMMonitor, use_cov: bool = True
) -> Tuple[bool, str]:
    """
    Run a single test file with RAM monitoring.

    Returns:
        (success: bool, output: str)
    """
    print(f"\n   Running {Path(test_file).name}...", end=" ", flush=True)

    cmd = [
        sys.executable,
        "-m",
        "pytest",
        test_file,
        "-q",
        "--tb=line",
        "-v",
    ]

    if use_cov:
        cmd.extend(["--cov=src", "--cov-report=term-missing"])

    start_ram, _ = monitor.check_and_record()

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )

        end_ram, exceeded = monitor.check_and_record()
        ram_increase = end_ram - start_ram

        if exceeded:
            print(f"⚠️  RAM EXCEEDED ({end_ram:.1f} MB > {monitor.threshold_mb} MB)")
            return False, f"RAM exceeded: {end_ram:.1f} MB"

        success = result.returncode == 0
        if success:
            # Count passed tests from output
            passed_count = result.stdout.count(" PASSED")
            print(f"✅ ({passed_count} tests, +{ram_increase:.1f} MB)")
        else:
            print(f"❌ (return code {result.returncode}, +{ram_increase:.1f} MB)")

        return success, result.stdout

    except subprocess.TimeoutExpired:
        print(f"⏱️  TIMEOUT (60s)")
        return False, "Test timeout"
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, str(e)


def run_safe_mode(use_cov: bool = True) -> int:
    """Run only lightweight unit tests (safe mode)."""
    print(f"\n{'=' * 70}")
    print("🧪 SAFE MODE: Lightweight Unit Tests (No Heavy ML)")
    print(f"{'=' * 70}")

    monitor = RAMMonitor()
    results = []
    total_time = 0

    start_time = time.time()

    for test_file in LIGHTWEIGHT_TESTS:
        if not Path(test_file).exists():
            print(f"\n   ⏭️  Skipped {test_file} (not found)")
            continue

        file_start = time.time()
        success, output = run_test_file(test_file, monitor, use_cov=use_cov)
        file_time = time.time() - file_start
        total_time += file_time

        results.append((test_file, success, file_time))

        if not success:
            print(f"   ❌ Failed — stopping to avoid cascade failures")
            break

    end_time = time.time()

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 SAFE MODE SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    monitor.report()

    if passed == total:
        print("\n✓ All safe tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


def run_batch_mode(use_cov: bool = True) -> int:
    """Run all tests in batch (file-by-file with monitoring)."""
    print(f"\n{'=' * 70}")
    print("🔄 BATCH MODE: All Tests (With RAM Monitoring)")
    print(f"{'=' * 70}")

    monitor = RAMMonitor()
    results = []
    total_time = 0
    stopped_early = False

    start_time = time.time()

    for i, test_file in enumerate(TEST_FILES, 1):
        if not Path(test_file).exists():
            print(f"\n{i:2d}. ⏭️  {test_file} (not found)")
            continue

        print(f"\n{i:2d}. ", end="")

        file_start = time.time()
        success, output = run_test_file(test_file, monitor, use_cov=use_cov)
        file_time = time.time() - file_start
        total_time += file_time

        results.append((test_file, success, file_time))

        if not success:
            print(f"   ⚠️  Test failed. Continuing to next file...")
            # Don't stop — continue to gather as much coverage as possible
            # Only stop if RAM exceeded
            if "RAM exceeded" in output:
                print(f"   🛑 Stopping due to RAM limits")
                stopped_early = True
                break

    end_time = time.time()

    # Summary
    print(f"\n{'=' * 70}")
    print("📊 BATCH MODE SUMMARY")
    print(f"{'=' * 70}")

    passed = sum(1 for _, success, _ in results if success)
    total = len(results)

    print(f"✅ Passed: {passed}/{total}")
    print(f"⏱️  Total Time: {total_time:.1f}s")
    monitor.report()

    if stopped_early:
        print(f"\n⚠️  Stopped early due to RAM limits")
        print(f"   Ran {total}/{len(TEST_FILES)} test files")
        return 2  # Partial success
    elif passed == total:
        print(f"\n✓ All tests passed!")
        return 0
    else:
        print(f"\n⚠️  {total - passed} test(s) failed (but continuing to completion)")
        return 1


def main():
    """Main entry point."""
    global SAFE_RAM_THRESHOLD_MB

    import argparse

    parser = argparse.ArgumentParser(
        description="Smart test runner with RAM monitoring"
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run all tests in batch (default: safe mode only)",
    )
    parser.add_argument(
        "--no-cov",
        action="store_true",
        help="Skip coverage reporting (runs faster)",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=SAFE_RAM_THRESHOLD_MB,
        help=f"RAM threshold in MB (default: {SAFE_RAM_THRESHOLD_MB})",
    )

    args = parser.parse_args()

    # Update threshold if provided
    SAFE_RAM_THRESHOLD_MB = args.threshold

    print("☕ Coffee Text Analytics - Smart Test Runner")
    print(f"   RAM Threshold: {args.threshold} MB")
    print(f"   Coverage: {'❌ Disabled' if args.no_cov else '✅ Enabled'}")

    if args.batch:
        exit_code = run_batch_mode(use_cov=not args.no_cov)
    else:
        exit_code = run_safe_mode(use_cov=not args.no_cov)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
